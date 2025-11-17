"""Main pipeline for image denoising baseline comparison."""

import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Import JAX to verify GPU
try:
    import jax
    import jax.numpy as jnp
    print(f"JAX devices: {jax.devices()}")
except ImportError:
    print("Warning: JAX not available")
    sys.exit(1)

# Import our modules
from s1_input.loader import load_image, load_images_from_dir, quantize_to_3bit, dequantize_from_3bit, save_image
from PIL import Image, ImageDraw, ImageFont
from s1_input.noise_gen import add_gaussian_noise, NoiseGenerator
from s2a_baseline.bilateral_filter import bilateral_filter, BilateralFilterJAX
from s3_results.benchmark import GPUPowerMonitor, benchmark_function
from s3_results.metrics import calculate_psnr, calculate_ssim, calculate_efficiency_metrics, calculate_improvement_metrics, export_results_to_json
from s3_results.visualizer import display_metrics, visualize_power_consumption, display_comparison


def verify_gpu():
    """Verify GPU is available."""
    devices = jax.devices()
    # Check for GPU devices - look for 'cuda' or 'gpu' in device string
    # JAX CUDA devices show as "CudaDevice(id=0)" or similar
    gpu_devices = [d for d in devices if 'cuda' in str(d).lower() or 'gpu' in str(d).lower()]
    
    if len(gpu_devices) == 0:
        print("Warning: No GPU devices found. Will use CPU (slower).")
        return False
    else:
        print(f"Found {len(gpu_devices)} GPU device(s): {gpu_devices}")
        return True


def create_comparison_image(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    output_path: Path,
    psnr: float = None,
    ssim: float = None,
    improvement: dict = None,
):
    """
    Create a side-by-side comparison image with labels.
    
    Args:
        clean: Clean reference image
        noisy: Noisy input image
        denoised: Denoised output image
        output_path: Path to save the comparison image
        psnr: Optional PSNR value to display
        ssim: Optional SSIM value to display
    """
    # Convert images to uint8 for PIL
    def to_uint8(img):
        img_clipped = np.clip(img, 0.0, 1.0)
        return (img_clipped * 255.0).astype(np.uint8)
    
    clean_uint8 = to_uint8(clean)
    noisy_uint8 = to_uint8(noisy)
    denoised_uint8 = to_uint8(denoised)
    
    # Get image dimensions
    H, W = clean_uint8.shape
    padding = 10
    label_height = 40
    spacing = 20
    
    # Create combined image: 3 images side by side
    total_width = W * 3 + spacing * 2 + padding * 2
    total_height = H + label_height * 2 + padding * 2
    
    # Create white background
    combined = Image.new('RGB', (total_width, total_height), color='white')
    
    # Paste images
    x_offset = padding
    y_offset = padding + label_height
    
    # Clean image
    clean_pil = Image.fromarray(clean_uint8, mode='L')
    combined.paste(clean_pil, (x_offset, y_offset))
    
    # Noisy image
    x_offset += W + spacing
    noisy_pil = Image.fromarray(noisy_uint8, mode='L')
    combined.paste(noisy_pil, (x_offset, y_offset))
    
    # Denoised image
    x_offset += W + spacing
    denoised_pil = Image.fromarray(denoised_uint8, mode='L')
    combined.paste(denoised_pil, (x_offset, y_offset))
    
    # Add labels
    draw = ImageDraw.Draw(combined)
    
    # Try to use a nice font, fallback to default if not available
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Top labels
    y_label_top = padding + 5
    draw.text((padding + W // 2, y_label_top), "Clean (Reference)", 
              fill='black', anchor='mt', font=font_large)
    draw.text((padding + W + spacing + W // 2, y_label_top), "Noisy (Input)", 
              fill='black', anchor='mt', font=font_large)
    draw.text((padding + 2 * (W + spacing) + W // 2, y_label_top), "Denoised (Output)", 
              fill='black', anchor='mt', font=font_large)
    
    # Bottom labels with metrics
    y_label_bottom = y_offset + H + 10
    draw.text((padding + W // 2, y_label_bottom), "Original", 
              fill='gray', anchor='mt', font=font_small)
    
    # Noisy image label with metrics
    x_noisy = padding + W + spacing + W // 2
    y_current = y_label_bottom
    if improvement and 'psnr_noisy' in improvement:
        draw.text((x_noisy, y_current), f"σ = 15%", 
                  fill='gray', anchor='mt', font=font_small)
        y_current += 18
        draw.text((x_noisy, y_current), f"PSNR: {improvement['psnr_noisy']:.2f} dB", 
                  fill='gray', anchor='mt', font=font_small)
    else:
        draw.text((x_noisy, y_current), "σ = 15%", 
                  fill='gray', anchor='mt', font=font_small)
    
    # Denoised image label with metrics and improvements
    x_metrics = padding + 2 * (W + spacing) + W // 2
    y_current = y_label_bottom
    draw.text((x_metrics, y_current), "Bilateral Filter", 
              fill='gray', anchor='mt', font=font_small)
    
    if psnr is not None:
        y_current += 18  # Line spacing
        improvement_text = f"PSNR: {psnr:.2f} dB"
        if improvement and 'psnr_improvement' in improvement:
            improvement_text += f" (Δ{improvement['psnr_improvement']:+.2f})"
        draw.text((x_metrics, y_current), improvement_text, 
                  fill='gray', anchor='mt', font=font_small)
    
    if ssim is not None:
        y_current += 18  # Line spacing
        ssim_text = f"SSIM: {ssim:.4f}"
        if improvement and 'ssim_improvement' in improvement:
            ssim_text += f" (Δ{improvement['ssim_improvement']:+.4f})"
        draw.text((x_metrics, y_current), ssim_text, 
                  fill='gray', anchor='mt', font=font_small)
    
    if improvement and 'mse_reduction_pct' in improvement:
        y_current += 18  # Line spacing
        draw.text((x_metrics, y_current), f"Noise ↓ {improvement['mse_reduction_pct']:.1f}%", 
                  fill='green', anchor='mt', font=font_small)
    
    # Save
    combined.save(output_path)


def create_toy_example():
    """Create a synthetic 3-bit grayscale image for testing."""
    # Create a simple test pattern
    H, W = 256, 256
    
    # Create gradient pattern
    y_coords, x_coords = np.meshgrid(
        np.linspace(0, 1, H),
        np.linspace(0, 1, W),
        indexing='ij'
    )
    
    # Create pattern with circles
    center_y, center_x = 0.5, 0.5
    radius = 0.3
    dist = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    pattern = np.exp(-dist / radius)
    
    # Quantize to 3-bit
    quantized = quantize_to_3bit(pattern)
    
    # Dequantize back to [0, 1]
    image = dequantize_from_3bit(quantized)
    
    return image


def run_toy_example():
    """Run a toy example to verify everything works."""
    print("\n" + "="*60)
    print("Running Toy Example")
    print("="*60)
    
    # Verify GPU
    has_gpu = verify_gpu()
    
    # Create toy image
    print("\nCreating synthetic 3-bit grayscale image...")
    clean_image = create_toy_example()
    print(f"Image shape: {clean_image.shape}, range: [{clean_image.min():.3f}, {clean_image.max():.3f}]")
    
    # Add noise
    print("\nAdding Gaussian noise (15% std dev)...")
    noisy_image = add_gaussian_noise(clean_image, sigma=0.15, seed=42)
    print(f"Noisy image range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
    
    # Apply bilateral filter (using more aggressive default parameters)
    print("\nApplying bilateral filter...")
    # Default aggressive parameters for better noise reduction:
    # - Larger filter diameter (d=11) for larger neighborhood
    # - Lower sigma_color (30) for tighter intensity matching (more aggressive)
    # - Higher sigma_space (100) for more spatial smoothing
    filter_obj = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)
    print(f"  Filter parameters: d=11, sigma_color=30.0, sigma_space=100.0")
    
    # First run (JIT compilation)
    print("  First run (JIT compilation)...")
    start = datetime.now()
    denoised = filter_obj.denoise(noisy_image)
    first_time = (datetime.now() - start).total_seconds() * 1000
    print(f"  First run time: {first_time:.2f} ms")
    
    # Second run (should be faster)
    print("  Second run (after JIT)...")
    start = datetime.now()
    denoised = filter_obj.denoise(noisy_image)
    second_time = (datetime.now() - start).total_seconds() * 1000
    print(f"  Second run time: {second_time:.2f} ms")
    
    if second_time < first_time:
        print(f"  ✓ JIT compilation working (speedup: {first_time/second_time:.2f}x)")
    else:
        print("  ⚠ JIT may not be working as expected")
    
    # Calculate metrics
    print("\nCalculating quality metrics...")
    psnr = calculate_psnr(clean_image, denoised)
    ssim = calculate_ssim(clean_image, denoised)
    
    # Calculate improvement metrics (noisy vs denoised)
    improvement = calculate_improvement_metrics(clean_image, noisy_image, denoised)
    
    print(f"  PSNR: {psnr:.2f} dB (improvement: {improvement['psnr_improvement']:+.2f} dB)")
    print(f"  SSIM: {ssim:.4f} (improvement: {improvement['ssim_improvement']:+.4f})")
    print(f"  MSE reduction: {improvement['mse_reduction_pct']:.1f}%")
    
    # Test power monitoring
    print("\nTesting power monitoring...")
    monitor = GPUPowerMonitor(gpu_id=0)
    with monitor:
        _ = filter_obj.denoise(noisy_image)
    
    print(f"  Energy: {monitor.total_energy_j:.3f} J")
    print(f"  Avg Power: {monitor.avg_power_w:.2f} W")
    print(f"  Duration: {monitor.duration_s:.3f} s")
    
    # Save results
    output_dir = Path("s0_data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual images
    save_image(clean_image, output_dir / "toy_clean.png")
    save_image(noisy_image, output_dir / "toy_noisy.png")
    save_image(denoised, output_dir / "toy_denoised.png")
    
    # Create combined comparison image with labels
    create_comparison_image(
        clean_image,
        noisy_image,
        denoised,
        output_dir / "toy_comparison.png",
        psnr=psnr,
        ssim=ssim,
        improvement=improvement,
    )
    print(f"\nSaved images to {output_dir}/")
    print(f"  - Individual images: toy_clean.png, toy_noisy.png, toy_denoised.png")
    print(f"  - Comparison: toy_comparison.png")
    
    print("\n✓ Toy example completed successfully!")
    return True


def benchmark_image(
    image_path: str,
    noise_level: float = 0.15,
    n_warmup: int = 5,
    n_runs: int = 20,
    save_images: bool = False,
    filter_d: int = 11,
    sigma_color: float = 30.0,
    sigma_space: float = 100.0,
):
    """Benchmark denoising on a single image."""
    print("\n" + "="*60)
    print(f"Benchmarking: {image_path}")
    print("="*60)
    
    # Load image
    print("\nLoading image...")
    clean_image = load_image(image_path, target_size=(256, 256))
    print(f"Image shape: {clean_image.shape}")
    
    # Quantize to 3-bit
    quantized = quantize_to_3bit(clean_image)
    clean_3bit = dequantize_from_3bit(quantized)
    
    # Add noise
    print(f"\nAdding Gaussian noise (sigma={noise_level})...")
    noisy_image = add_gaussian_noise(clean_3bit, sigma=noise_level, seed=42)
    
    # Create filter with configurable parameters
    filter_obj = BilateralFilterJAX(
        d=filter_d,
        sigma_color=sigma_color,
        sigma_space=sigma_space
    )
    print(f"  Filter parameters: d={filter_d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
    
    # Benchmark
    print("\nRunning benchmark...")
    def denoise_fn():
        return filter_obj.denoise(noisy_image)
    
    results = benchmark_function(
        denoise_fn,
        n_warmup=n_warmup,
        n_runs=n_runs,
        gpu_id=0,
    )
    
    # Get one denoised result for metrics
    denoised = denoise_fn()
    
    # Calculate quality metrics
    psnr = calculate_psnr(clean_3bit, denoised)
    ssim = calculate_ssim(clean_3bit, denoised)
    
    # Calculate improvement metrics (noisy vs denoised)
    improvement = calculate_improvement_metrics(clean_3bit, noisy_image, denoised)
    
    # Calculate efficiency metrics
    efficiency = calculate_efficiency_metrics(psnr, results['mean_energy_j'], clean_3bit.shape)
    
    # Combine results
    full_results = {
        **results,
        'psnr': psnr,
        'ssim': ssim,
        **improvement,
        **efficiency,
        'image_path': str(image_path),
        'noise_level': noise_level,
    }
    
    # Display results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    display_metrics(full_results)
    
    # Save results
    if save_images:
        output_dir = Path("s0_data/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_image(clean_3bit, output_dir / f"clean_{timestamp}.png")
        save_image(noisy_image, output_dir / f"noisy_{timestamp}.png")
        save_image(denoised, output_dir / f"denoised_{timestamp}.png")
        
        # Create comparison image
        create_comparison_image(
            clean_3bit,
            noisy_image,
            denoised,
            output_dir / f"comparison_{timestamp}.png",
            psnr=psnr,
            ssim=ssim,
            improvement=improvement,
        )
        print(f"\nSaved images to {output_dir}/")
        print(f"  - Individual images: clean_{timestamp}.png, noisy_{timestamp}.png, denoised_{timestamp}.png")
        print(f"  - Comparison: comparison_{timestamp}.png")
    
    # Export JSON
    json_path = Path("s0_data/results") / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    export_results_to_json(full_results, json_path)
    print(f"\nResults saved to {json_path}")
    
    return full_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Image denoising baseline pipeline")
    parser.add_argument(
        '--toy',
        action='store_true',
        help='Run toy example to verify setup'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image'
    )
    parser.add_argument(
        '--noise-level',
        type=float,
        default=0.15,
        help='Noise level (std dev as fraction, default 0.15 = 15%%)'
    )
    parser.add_argument(
        '--n-warmup',
        type=int,
        default=5,
        help='Number of warmup runs (default: 5)'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=20,
        help='Number of benchmark runs (default: 20)'
    )
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='Save output images'
    )
    
    args = parser.parse_args()
    
    if args.toy:
        success = run_toy_example()
        sys.exit(0 if success else 1)
    elif args.image:
        benchmark_image(
            args.image,
            noise_level=args.noise_level,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
            save_images=args.save_images,
            filter_d=args.filter_d,
            sigma_color=args.sigma_color,
            sigma_space=args.sigma_space,
        )
    else:
        parser.print_help()
        print("\nUse --toy to run a toy example or --image <path> to benchmark an image.")


if __name__ == "__main__":
    main()

