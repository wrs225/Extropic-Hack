#!/usr/bin/env python3
"""
Main entry point for the denoising pipeline.

This script runs basic tests and examples using the pipeline infrastructure.
Place your input images in data/input/ and results will be saved to data/output/.
"""

from pathlib import Path
import numpy as np
import sys
import argparse

# Add parent directory to path to allow importing from src-temp if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src-temp"))

from pipeline import DenoisingPipeline, ImageDataStore, ComparisonVisualizer


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.15, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to an image."""
    np.random.seed(seed)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0.0, 1.0)
    return noisy


# Example 1: Using the existing Bilateral Filter
def example_bilateral_filter():
    """Example using bilateral filter with the pipeline."""
    try:
        from s2a_baseline.bilateral_filter import BilateralFilterJAX
    except ImportError:
        print("Warning: BilateralFilterJAX not found. Skipping bilateral filter example.")
        print("  (This is expected if you haven't set up the baseline algorithm yet)")
        return None
    
    # Initialize pipeline (sets up JAX GPU connection)
    print("\n" + "="*60)
    print("Example 1: Bilateral Filter")
    print("="*60)
    pipeline = DenoisingPipeline(gpu_id=0)
    datastore = ImageDataStore()
    visualizer = ComparisonVisualizer()
    
    # Create test image (or load from file)
    # For demo, create a simple test pattern
    H, W = 256, 256
    clean = np.ones((H, W)) * 0.5
    clean[64:192, 64:192] = 0.8  # White square
    
    # Add noise
    noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)
    
    # Create bilateral filter denoiser
    bilateral = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)
    
    # Process through pipeline
    print("Processing with Bilateral Filter...")
    results = pipeline.process(
        denoiser=bilateral,
        clean_image=clean,
        noisy_image=noisy,
        metadata={
            'algorithm': 'BilateralFilter',
            'd': 11,
            'sigma_color': 30.0,
            'sigma_space': 100.0,
        }
    )
    
    # Save results
    exp_dir = datastore.save_results(results, experiment_name="bilateral_example")
    
    # Create comparison image
    visualizer.create_comparison(
        clean=results['clean'],
        noisy=results['noisy'],
        denoised=results['denoised'],
        output_path=exp_dir / "comparison.png",
        metrics=results['metrics'],
        algorithm_name="Bilateral Filter",
    )
    
    # Print results
    print(f"\n✓ Results saved to: {exp_dir}")
    print(f"  PSNR improvement: {results['metrics']['psnr_improvement']:.2f} dB")
    print(f"  SSIM improvement: {results['metrics']['ssim_improvement']:.4f}")
    print(f"  Energy: {results['power']['mean_energy_j']:.3f} J")
    print(f"  Time: {results['performance']['mean_time_ms']:.2f} ms")
    
    return results


# Example 2: Custom denoiser implementation
class SimpleGaussianDenoiser:
    """Simple example denoiser using Gaussian blur."""
    
    def __init__(self, sigma=1.0):
        self.sigma = sigma
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Simple Gaussian blur denoising."""
        try:
            from scipy import ndimage
            denoised = ndimage.gaussian_filter(image, sigma=self.sigma)
            return np.clip(denoised, 0.0, 1.0)
        except ImportError:
            # Fallback: simple averaging filter
            from scipy.signal import convolve2d
            kernel = np.ones((5, 5)) / 25.0
            denoised = convolve2d(image, kernel, mode='same', boundary='symm')
            return np.clip(denoised, 0.0, 1.0)
    
    def denoise_batch(self, images: list) -> list:
        return [self.denoise(img) for img in images]


def example_custom_denoiser():
    """Example using a custom denoiser."""
    print("\n" + "="*60)
    print("Example 2: Custom Gaussian Denoiser")
    print("="*60)
    
    pipeline = DenoisingPipeline(gpu_id=0)
    datastore = ImageDataStore()
    visualizer = ComparisonVisualizer()
    
    # Load or create test image
    clean = np.ones((256, 256)) * 0.5
    clean[64:192, 64:192] = 0.8
    noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)
    
    # Create custom denoiser
    denoiser = SimpleGaussianDenoiser(sigma=1.5)
    
    # Process
    print("Processing with Gaussian Blur...")
    results = pipeline.process(
        denoiser=denoiser,
        clean_image=clean,
        noisy_image=noisy,
        metadata={'algorithm': 'SimpleGaussian', 'sigma': 1.5},
    )
    
    # Save
    exp_dir = datastore.save_results(results, "gaussian_example")
    
    # Create comparison
    visualizer.create_comparison(
        clean=results['clean'],
        noisy=results['noisy'],
        denoised=results['denoised'],
        output_path=exp_dir / "comparison.png",
        metrics=results['metrics'],
        algorithm_name="Gaussian Blur",
    )
    
    print(f"\n✓ Results saved to: {exp_dir}")
    print(f"  PSNR improvement: {results['metrics']['psnr_improvement']:.2f} dB")
    print(f"  SSIM improvement: {results['metrics']['ssim_improvement']:.4f}")
    
    return results


# Example 3: Load from input directory
def example_load_from_input():
    """Example loading images from data/input/ directory."""
    print("\n" + "="*60)
    print("Example 3: Load from Input Directory")
    print("="*60)
    
    pipeline = DenoisingPipeline(gpu_id=0)
    datastore = ImageDataStore()
    visualizer = ComparisonVisualizer()
    
    # Check for input images
    input_dir = datastore.input_dir
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    
    if len(image_files) == 0:
        print(f"  No images found in {input_dir}/")
        print(f"  Place your input images (PNG/JPG) in {input_dir}/ to test with real images")
        return None
    
    # Load first image
    image_path = image_files[0]
    print(f"  Loading: {image_path.name}")
    clean = datastore.load_image(image_path)
    
    # Add noise
    noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)
    
    # Use simple denoiser
    denoiser = SimpleGaussianDenoiser(sigma=1.5)
    
    # Process
    results = pipeline.process(
        denoiser=denoiser,
        clean_image=clean,
        noisy_image=noisy,
        metadata={'algorithm': 'Gaussian', 'input_file': image_path.name},
    )
    
    # Save
    exp_dir = datastore.save_results(results, f"input_{image_path.stem}")
    
    # Create comparison
    visualizer.create_comparison(
        clean=results['clean'],
        noisy=results['noisy'],
        denoised=results['denoised'],
        output_path=exp_dir / "comparison.png",
        metrics=results['metrics'],
        algorithm_name="Gaussian Blur",
    )
    
    print(f"\n✓ Results saved to: {exp_dir}")
    return results


# Example 4: Comparing multiple algorithms
def example_compare_algorithms():
    """Compare multiple denoising algorithms."""
    print("\n" + "="*60)
    print("Example 4: Algorithm Comparison")
    print("="*60)
    
    pipeline = DenoisingPipeline(gpu_id=0)
    datastore = ImageDataStore()
    
    # Create test image
    clean = np.ones((256, 256)) * 0.5
    clean[64:192, 64:192] = 0.8
    noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)
    
    algorithms = []
    
    # Test 1: Bilateral Filter (if available)
    try:
        from s2a_baseline.bilateral_filter import BilateralFilterJAX
        bilateral = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)
        results_bilateral = pipeline.process(
            denoiser=bilateral,
            clean_image=clean,
            noisy_image=noisy,
            metadata={'algorithm': 'BilateralFilter'},
        )
        algorithms.append(('Bilateral Filter', results_bilateral))
    except ImportError:
        print("  BilateralFilter not available, skipping...")
    
    # Test 2: Custom denoiser
    gaussian = SimpleGaussianDenoiser(sigma=1.5)
    results_gaussian = pipeline.process(
        denoiser=gaussian,
        clean_image=clean,
        noisy_image=noisy,
        metadata={'algorithm': 'Gaussian'},
    )
    algorithms.append(('Gaussian Blur', results_gaussian))
    
    # Compare results
    print("\n" + "-"*60)
    print("Comparison Results:")
    print("-"*60)
    for name, results in algorithms:
        print(f"\n{name}:")
        print(f"  PSNR: {results['metrics']['psnr_denoised']:.2f} dB (improvement: {results['metrics']['psnr_improvement']:+.2f} dB)")
        print(f"  SSIM: {results['metrics']['ssim_denoised']:.4f} (improvement: {results['metrics']['ssim_improvement']:+.4f})")
        print(f"  Energy: {results['power']['mean_energy_j']:.3f} J")
        print(f"  Time: {results['performance']['mean_time_ms']:.2f} ms")
    
    return algorithms


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Denoising pipeline - run basic tests and examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run all examples
  python main.py --bilateral         # Run bilateral filter example
  python main.py --custom           # Run custom denoiser example
  python main.py --input            # Load from data/input/ directory
  python main.py --compare          # Compare multiple algorithms
        """
    )
    parser.add_argument('--bilateral', action='store_true', help='Run bilateral filter example')
    parser.add_argument('--custom', action='store_true', help='Run custom denoiser example')
    parser.add_argument('--input', action='store_true', help='Load images from data/input/')
    parser.add_argument('--compare', action='store_true', help='Compare multiple algorithms')
    
    args = parser.parse_args()
    
    # If no specific example requested, run all
    if not any([args.bilateral, args.custom, args.input, args.compare]):
        print("="*60)
        print("Denoising Pipeline - Running All Examples")
        print("="*60)
        print("\nDirectory Structure:")
        print(f"  Input images:  data/input/")
        print(f"  Output results: data/output/")
        print(f"  Raw images:    data/raw/")
        print("\nRunning examples...")
        
        example_bilateral_filter()
        example_custom_denoiser()
        example_load_from_input()
        example_compare_algorithms()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        print(f"\nCheck results in: data/output/")
    else:
        # Run specific examples
        if args.bilateral:
            example_bilateral_filter()
        if args.custom:
            example_custom_denoiser()
        if args.input:
            example_load_from_input()
        if args.compare:
            example_compare_algorithms()


if __name__ == '__main__':
    main()

