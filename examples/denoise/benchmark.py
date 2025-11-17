"""
Benchmark script for 4-bit denoising with standard test images:
- Cameraman (256×256): Very sharp edges + large flat areas
- Barbara (512×512): Woven cloth patterns - hardest test
- Boat (512×512): Mix of straight edges + natural textures
- Peppers (512×512): Smooth gradients + sharp reflections
"""

import os
import sys
import jax
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import urllib.request

from . import MultiLayerDenoiser, add_noise, compute_metrics


# Standard test image URLs (grayscale)
# Using alternative sources that are more accessible
TEST_IMAGES = {
    'cameraman': {
        'url': 'https://sipi.usc.edu/database/preview/misc/4.2.03.png',
        'size': (256, 256),
        'description': 'Very sharp edges + large flat areas'
    },
    'barbara': {
        'url': 'https://www.hlevkin.com/06testimages/barbara.bmp',
        'size': (512, 512),
        'description': 'Woven cloth patterns - hardest test'
    },
    'boat': {
        'url': 'https://sipi.usc.edu/database/preview/misc/4.2.07.png',
        'size': (512, 512),
        'description': 'Mix of straight edges + natural textures'
    },
    'peppers': {
        'url': 'https://sipi.usc.edu/database/preview/misc/4.2.06.png',
        'size': (512, 512),
        'description': 'Smooth gradients + sharp reflections'
    },
}


def create_synthetic_test_image(name, size):
    """Create a synthetic test image with different characteristics."""
    h, w = size
    img = np.zeros((h, w), dtype=np.uint8)

    if name == 'cameraman':
        # Sharp edges + flat areas
        # Large circle
        y, x = np.ogrid[:h, :w]
        center = h // 2, w // 2
        mask = (x - center[1])**2 + (y - center[0])**2 <= (h // 3)**2
        img[mask] = 200
        # Rectangles
        img[h//4:h//2, w//4:w//2] = 100
        img[h//2:3*h//4, w//2:3*w//4] = 150

    elif name == 'barbara':
        # Checkerboard patterns (woven cloth simulation)
        tile_size = 8
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                if ((i // tile_size) + (j // tile_size)) % 2 == 0:
                    img[i:i+tile_size, j:j+tile_size] = 180
                else:
                    img[i:i+tile_size, j:j+tile_size] = 80
        # Add some gradients
        img[:h//2, :] = np.clip(img[:h//2, :] + np.linspace(0, 30, w).astype(np.uint8), 0, 255)

    elif name == 'boat':
        # Mix of edges and textures
        # Diagonal lines
        for i in range(h):
            img[i, :] = np.clip(i % 50 * 5, 0, 255).astype(np.uint8)
        # Add rectangles for boat hull
        img[h//3:2*h//3, w//4:3*w//4] = 120
        # Add some noise texture
        texture = np.random.randint(0, 30, (h, w), dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)

    elif name == 'peppers':
        # Smooth gradients + sharp features
        y, x = np.ogrid[:h, :w]
        # Radial gradient
        img = np.clip(np.sqrt((x - w//2)**2 + (y - h//2)**2) * 255 / (h//2), 0, 255).astype(np.uint8)
        # Add some sharp circles (peppers)
        for cy, cx, r, intensity in [(h//4, w//4, h//8, 200), (3*h//4, 3*w//4, h//8, 150)]:
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            img[mask] = intensity

    return img


def download_image(name, info, data_dir='data'):
    """Download a test image if not already present, or create synthetic if download fails."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f'{name}.png')

    if os.path.exists(filepath):
        print(f"  {name}: already downloaded")
        return filepath

    print(f"  {name}: downloading from {info['url']}...", end='', flush=True)
    try:
        urllib.request.urlretrieve(info['url'], filepath)
        print(" done")
        return filepath
    except Exception as e:
        print(f" failed: {e}")
        print(f"    Creating synthetic {name} image instead...", end='', flush=True)
        synthetic_img = create_synthetic_test_image(name, info['size'])
        Image.fromarray(synthetic_img).save(filepath)
        print(" done")
        return filepath


def load_and_prepare_image(filepath, target_size=None):
    """Load image and convert to 4-bit grayscale."""
    img = Image.open(filepath)

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Resize if needed
    if target_size is not None:
        img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Quantize to 4-bit (0-15)
    img_4bit = (img_array >> 4).astype(np.uint8)

    return img_4bit


def run_benchmark(image_name, img_4bit, noise_level=0.15, n_warmup=20, n_samples=10):
    """Run benchmark on a single image."""
    h, w = img_4bit.shape

    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {image_name.upper()}")
    print(f"{'='*70}")
    print(f"Size: {h}x{w}")
    print(f"Noise level: {noise_level}")
    print(f"Unique values: {len(np.unique(img_4bit))}")

    # Add noise
    print("\n1. Adding noise...")
    np.random.seed(42)
    noisy_img = add_noise(img_4bit, noise_level=noise_level)
    print(f"   Noisy unique values: {len(np.unique(noisy_img))}")

    # Create denoiser
    print("\n2. Initializing denoiser...")
    denoiser = MultiLayerDenoiser(
        h=h, w=w, num_bits=4,
        n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=5
    )

    # Denoise
    print("\n3. Denoising...")
    key = jax.random.PRNGKey(42)
    denoised_img, individual_imgs, denoise_time = denoiser.denoise_image(
        noisy_img, key=key, warm_up=True, verbose=True
    )

    # Metrics
    print("\n4. Computing metrics...")
    metrics = compute_metrics(img_4bit, noisy_img, denoised_img)

    print(f"   Noisy PSNR:     {metrics['psnr_noisy']:.2f} dB")
    print(f"   Denoised PSNR:  {metrics['psnr_denoised']:.2f} dB")
    print(f"   Improvement:    {metrics['psnr_improvement']:.2f} dB")

    # Save results
    print("\n5. Saving results...")
    # Get absolute path to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, 'build', f'benchmark_{image_name}')
    os.makedirs(output_dir, exist_ok=True)

    # Save comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = 15

    axes[0].imshow(img_4bit, cmap='gray', vmin=0, vmax=vmax)
    axes[0].set_title(f'{image_name.capitalize()} - Original (4-bit)')
    axes[0].axis('off')

    axes[1].imshow(noisy_img, cmap='gray', vmin=0, vmax=vmax)
    axes[1].set_title(f'Noisy\nPSNR: {metrics["psnr_noisy"]:.2f} dB')
    axes[1].axis('off')

    axes[2].imshow(denoised_img, cmap='gray', vmin=0, vmax=vmax)
    axes[2].set_title(f'Denoised\nPSNR: {metrics["psnr_denoised"]:.2f} dB (+{metrics["psnr_improvement"]:.2f} dB)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir}/comparison.png")
    plt.close()

    # Save individual images
    Image.fromarray((img_4bit * 17).astype(np.uint8)).save(f'{output_dir}/original.png')
    Image.fromarray((noisy_img * 17).astype(np.uint8)).save(f'{output_dir}/noisy.png')
    Image.fromarray((denoised_img * 17).astype(np.uint8)).save(f'{output_dir}/denoised.png')

    # Save detail crops (center 128x128)
    crop_size = 128
    ch, cw = h // 2, w // 2
    crop = slice(ch - crop_size//2, ch + crop_size//2), slice(cw - crop_size//2, cw + crop_size//2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_4bit[crop], cmap='gray', vmin=0, vmax=vmax)
    axes[0].set_title('Original (detail)')
    axes[0].axis('off')

    axes[1].imshow(noisy_img[crop], cmap='gray', vmin=0, vmax=vmax)
    axes[1].set_title('Noisy (detail)')
    axes[1].axis('off')

    axes[2].imshow(denoised_img[crop], cmap='gray', vmin=0, vmax=vmax)
    axes[2].set_title('Denoised (detail)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/detail_crop.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir}/detail_crop.png")
    plt.close()

    return {
        'name': image_name,
        'size': (h, w),
        'time': denoise_time,
        **metrics
    }


def main():
    print("="*70)
    print("4-BIT DENOISING BENCHMARK - STANDARD TEST IMAGES")
    print("="*70)

    # Download images
    print("\nDownloading test images...")
    image_files = {}
    for name, info in TEST_IMAGES.items():
        filepath = download_image(name, info)
        if filepath:
            image_files[name] = filepath

    if not image_files:
        print("\nError: No images could be downloaded")
        return

    # Run benchmarks
    results = []

    for name, filepath in image_files.items():
        info = TEST_IMAGES[name]
        print(f"\n\nLoading {name}...")
        print(f"  Description: {info['description']}")

        # Load and prepare image
        img_4bit = load_and_prepare_image(filepath)
        print(f"  Size: {img_4bit.shape}")

        # Run benchmark
        result = run_benchmark(name, img_4bit, noise_level=0.15)
        results.append(result)

    # Summary
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\n{'Image':<15} {'Size':<12} {'Noisy PSNR':>12} {'Denoised PSNR':>14} {'Improvement':>12} {'Time (s)':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r['name']:<15} {str(r['size']):<12} {r['psnr_noisy']:>11.2f} dB {r['psnr_denoised']:>13.2f} dB {r['psnr_improvement']:>11.2f} dB {r['time']:>9.3f}s")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)

    # Save summary to file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_file = os.path.join(project_root, 'build', 'benchmark_summary.txt')
    with open(output_file, 'w') as f:
        f.write("4-BIT DENOISING BENCHMARK RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Image':<15} {'Size':<12} {'Noisy PSNR':>12} {'Denoised PSNR':>14} {'Improvement':>12} {'Time (s)':>10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['name']:<15} {str(r['size']):<12} {r['psnr_noisy']:>11.2f} dB {r['psnr_denoised']:>13.2f} dB {r['psnr_improvement']:>11.2f} dB {r['time']:>9.3f}s\n")

    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()
