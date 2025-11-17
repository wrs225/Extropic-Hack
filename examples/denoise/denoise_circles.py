#!/usr/bin/env python3
"""
Circles synthetic image denoising demo using MultiLayerDenoiser.
Creates a test pattern with circles and rectangles, adds salt & pepper noise,
and creates a comparison gif showing original | noisy | denoised.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add src to path to import denoise module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from denoise import MultiLayerDenoiser, add_salt_pepper_noise

def create_test_image(size=256):
    """Create a 4-bit greyscale test image with circles and rectangles."""
    img = np.zeros((size, size), dtype=np.uint8)

    center = size // 2
    y, x = np.ogrid[:size, :size]

    # Create patterns with 4-bit values (0-15)
    circles = [
        (center, center, size//3, 14),
        (size//4, size//4, size//8, 10),
        (3*size//4, 3*size//4, size//8, 6),
    ]

    for cy, cx, r, intensity in circles:
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = intensity

    # Rectangles
    img[20:60, 20:80] = 12
    img[size-60:size-20, size-80:size-20] = 8

    return img

def scale_to_8bit(img_4bit):
    """Convert 4-bit image back to 8-bit for display."""
    return (img_4bit * 17).astype(np.uint8)

def create_comparison(original, noisy, denoised):
    """Create three-way comparison image: original | noisy | denoised."""
    h, w = original.shape
    comparison = np.zeros((h, w * 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w:2*w] = noisy
    comparison[:, 2*w:] = denoised
    return comparison

def main():
    # Configuration
    output_gif = '/home/will/Extropic-Hack/data/circles_denoised.gif'
    noise_level = 0.10  # 10% salt & pepper noise

    print("="*60)
    print("Circles Synthetic Image Denoising Demo")
    print("="*60)
    print(f"Output gif: {output_gif}")
    print(f"Noise level: {noise_level*100}%")
    print("="*60)

    # Create synthetic test image
    print("\nCreating synthetic test image with circles and rectangles...")
    img_4bit = create_test_image(size=256)
    print(f"Image size: {img_4bit.shape}")

    # Add salt & pepper noise
    print(f"Adding {noise_level*100}% salt & pepper noise...")
    noisy_4bit = add_salt_pepper_noise(img_4bit, noise_level=noise_level)

    # Initialize denoiser
    print("\nInitializing MultiLayerDenoiser...")
    denoiser = MultiLayerDenoiser(
        h=256,
        w=256,
        num_bits=4,
        n_warmup=20,
        n_samples=10,
        steps_per_sample=5,
        alpha_coef=0.5,
        beta_coef=1.0,
        temperature=1.0
    )

    # Denoise
    print("Denoising image...")
    denoised_4bit, _, denoise_time = denoiser.denoise_image(
        noisy_4bit,
        warm_up=True,
        verbose=True
    )
    print(f"Denoising time: {denoise_time:.3f}s")

    # Convert to 8-bit for visualization
    original_8bit = scale_to_8bit(img_4bit)
    noisy_8bit = scale_to_8bit(noisy_4bit)
    denoised_8bit = scale_to_8bit(denoised_4bit)

    # Create comparison frames
    comparison = create_comparison(original_8bit, noisy_8bit, denoised_8bit)

    # Create gif with multiple frames showing the same comparison (for visibility)
    print("\nCreating comparison gif...")
    frames = []
    comparison_pil = Image.fromarray(comparison).convert('RGB')

    # Add text labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison_pil)
    w = comparison.shape[1] // 3

    # Simple text labels (using default font)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    draw.text((w//2 - 30, 10), "Original", fill=(255, 255, 255), font=font)
    draw.text((w + w//2 - 20, 10), "Noisy", fill=(255, 255, 255), font=font)
    draw.text((2*w + w//2 - 35, 10), "Denoised", fill=(255, 255, 255), font=font)

    # Create a single frame that loops
    frames = [comparison_pil]

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:] if len(frames) > 1 else [],
        duration=2000,  # 2 seconds per frame
        loop=0
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Denoising time: {denoise_time:.3f}s")
    print(f"Output saved to: {output_gif}")
    print(f"{'='*60}")
    print("Done!")

if __name__ == "__main__":
    main()
