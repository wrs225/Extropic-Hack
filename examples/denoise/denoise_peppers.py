#!/usr/bin/env python3
"""
Peppers image denoising demo using MultiLayerDenoiser.
Adds salt & pepper noise and creates a comparison gif showing original | noisy | denoised.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add src to path to import denoise module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from denoise import MultiLayerDenoiser, add_salt_pepper_noise

def load_and_prepare_image(filepath, target_size=256):
    """Load image, convert to grayscale, resize, and quantize to 4-bit."""
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_8bit = np.array(img)
    img_4bit = (img_8bit / 17).astype(np.uint8)  # Quantize to 4-bit
    return img_4bit

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
    input_image = '/home/will/Extropic-Hack/data/peppers.jpeg'
    output_gif = '/home/will/Extropic-Hack/data/peppers_denoised.gif'
    noise_level = 0.10  # 10% salt & pepper noise

    print("="*60)
    print("Peppers Image Denoising Demo")
    print("="*60)
    print(f"Input image: {input_image}")
    print(f"Output gif: {output_gif}")
    print(f"Noise level: {noise_level*100}%")
    print("="*60)

    # Load and prepare image
    print("\nLoading and preparing image...")
    img_4bit = load_and_prepare_image(input_image)
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
