"""
Demo script for 4-bit greyscale denoising with Multi-Layer Ising Machine
"""

import os
import jax
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from . import MultiLayerDenoiser, add_noise, compute_metrics
from .utils import create_test_image


def main():
    print("="*70)
    print("UNIFIED 4-BIT DENOISING - ALL BIT PLANES IN ONE ISING MACHINE")
    print("="*70)

    # Parameters
    H, W = 256, 256
    NUM_BITS = 4

    print(f"Image size: {H}x{W}")
    print(f"Bit depth: {NUM_BITS} bits")
    print(f"Weights: α_b = 0.5 * 2^b, β_b = 1.0 * 2^b")
    print(f"Neighborhood: 4-connected (no inter-layer connections)")
    print("="*70)

    # Create test image
    print("\n1. Creating test image...")
    np.random.seed(42)
    original_img = create_test_image(size=H)
    print(f"   Unique values: {len(np.unique(original_img))}, range: [{original_img.min()}, {original_img.max()}]")

    # Add noise
    print("\n2. Adding Gaussian noise...")
    noisy_img = add_noise(original_img, noise_level=0.15)
    print(f"   Unique values: {len(np.unique(noisy_img))}")

    # Create denoiser
    print("\n3. Initializing denoiser...")
    denoiser = MultiLayerDenoiser(h=H, w=W, num_bits=NUM_BITS, n_warmup=20, n_samples=10, steps_per_sample=5)

    # Denoise
    print("\n4. Denoising...")
    key = jax.random.PRNGKey(42)
    denoised_img, individual_imgs, denoise_time = denoiser.denoise_image(noisy_img, key=key, warm_up=True, verbose=True)

    print(f"   Denoised unique values: {len(np.unique(denoised_img))}")
    print(f"   Got {len(individual_imgs)} individual samples")

    # Metrics
    print("\n5. Computing metrics...")
    metrics = compute_metrics(original_img, noisy_img, denoised_img)

    print(f"   Noisy PSNR:     {metrics['psnr_noisy']:.2f} dB")
    print(f"   Denoised PSNR:  {metrics['psnr_denoised']:.2f} dB (averaged)")
    print(f"   Improvement:    {metrics['psnr_improvement']:.2f} dB")

    # Compute metrics for individual samples
    print("\n   Individual sample PSNRs:")
    for i, sample_img in enumerate(individual_imgs):
        sample_metrics = compute_metrics(original_img, noisy_img, sample_img)
        print(f"     Sample {i}: {sample_metrics['psnr_denoised']:.2f} dB")

    # Visualize
    print("\n6. Saving results...")

    # Main comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = 15

    axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=vmax)
    axes[0].set_title('Original (4-bit)')
    axes[0].axis('off')

    axes[1].imshow(noisy_img, cmap='gray', vmin=0, vmax=vmax)
    axes[1].set_title(f'Noisy\nPSNR: {metrics["psnr_noisy"]:.2f} dB')
    axes[1].axis('off')

    axes[2].imshow(denoised_img, cmap='gray', vmin=0, vmax=vmax)
    axes[2].set_title(f'Denoised (Averaged)\nPSNR: {metrics["psnr_denoised"]:.2f} dB (+{metrics["psnr_improvement"]:.2f} dB)')
    axes[2].axis('off')

    # Create build folder
    os.makedirs('../../build', exist_ok=True)

    plt.tight_layout()
    plt.savefig('../../build/demo_result.png', dpi=150, bbox_inches='tight')
    print("   Saved: build/demo_result.png")

    # Save images
    Image.fromarray((original_img * 17).astype(np.uint8)).save('../../build/demo_original.png')
    Image.fromarray((noisy_img * 17).astype(np.uint8)).save('../../build/demo_noisy.png')
    Image.fromarray((denoised_img * 17).astype(np.uint8)).save('../../build/demo_denoised.png')
    print("   Saved: build/demo_original.png, demo_noisy.png, demo_denoised.png")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"  Time: {denoise_time:.3f}s")
    print(f"  PSNR improvement: {metrics['psnr_improvement']:.2f} dB")
    print(f"  Greyscale levels: {len(np.unique(denoised_img))}/16")
    print()


if __name__ == "__main__":
    main()
