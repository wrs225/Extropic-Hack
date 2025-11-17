#!/usr/bin/env python3
"""
Video denoising script using MultiLayerDenoiser.
Adds salt & pepper noise to video frames and denoises them, creating a side-by-side comparison gif.
"""

import os
import sys
import numpy as np
from PIL import Image
import time

# Add src to path to import denoise module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from denoise import MultiLayerDenoiser, add_salt_pepper_noise

def load_pgm_frame(filepath):
    """Load a PGM frame and convert to numpy array."""
    img = Image.open(filepath)
    return np.array(img)

def quantize_to_4bit(img_8bit):
    """Convert 8-bit image to 4-bit (0-15 range)."""
    return (img_8bit / 17).astype(np.uint8)

def scale_to_8bit(img_4bit):
    """Convert 4-bit image back to 8-bit for display."""
    return (img_4bit * 17).astype(np.uint8)

def create_side_by_side(noisy, denoised):
    """Create side-by-side comparison image."""
    h, w = noisy.shape
    comparison = np.zeros((h, w * 2), dtype=np.uint8)
    comparison[:, :w] = noisy
    comparison[:, w:] = denoised
    return comparison

def main():
    # Configuration
    video_dir = '/home/will/Extropic-Hack/data/will-video/processed'
    output_gif = '/home/will/Extropic-Hack/data/will-video/denoised_comparison.gif'
    noise_level = 0.10  # 10% salt & pepper noise

    print("="*60)
    print("Video Denoising with Salt & Pepper Noise")
    print("="*60)
    print(f"Video directory: {video_dir}")
    print(f"Output gif: {output_gif}")
    print(f"Noise level: {noise_level*100}%")
    print("="*60)

    # Load all frame paths
    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.pgm')])
    print(f"\nFound {len(frame_files)} frames")

    if len(frame_files) == 0:
        print("Error: No PGM frames found!")
        return

    # Initialize denoiser (256x256, 4-bit)
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
    print("Denoiser initialized!")

    # Process frames
    comparison_frames = []
    total_time = 0

    for i, frame_file in enumerate(frame_files):
        print(f"\n[{i+1}/{len(frame_files)}] Processing {frame_file}...")

        # Load frame
        frame_path = os.path.join(video_dir, frame_file)
        img_8bit = load_pgm_frame(frame_path)

        # Convert to 4-bit
        img_4bit = quantize_to_4bit(img_8bit)

        # Add salt & pepper noise
        noisy_4bit = add_salt_pepper_noise(img_4bit, noise_level=noise_level)

        # Denoise (warm_up only on first frame for JIT compilation)
        warm_up = (i == 0)
        denoised_4bit, _, denoise_time = denoiser.denoise_image(
            noisy_4bit,
            warm_up=warm_up,
            verbose=False
        )
        total_time += denoise_time

        print(f"  Denoising time: {denoise_time:.3f}s")

        # Convert back to 8-bit for visualization
        noisy_8bit = scale_to_8bit(noisy_4bit)
        denoised_8bit = scale_to_8bit(denoised_4bit)

        # Create side-by-side comparison
        comparison = create_side_by_side(noisy_8bit, denoised_8bit)

        # Convert to PIL Image (RGB for gif)
        comparison_pil = Image.fromarray(comparison).convert('RGB')
        comparison_frames.append(comparison_pil)

    # Create gif
    print(f"\n{'='*60}")
    print(f"Creating gif with {len(comparison_frames)} frames...")
    comparison_frames[0].save(
        output_gif,
        save_all=True,
        append_images=comparison_frames[1:],
        duration=100,  # 100ms per frame (~10 fps)
        loop=0
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames processed: {len(frame_files)}")
    print(f"Total denoising time: {total_time:.2f}s")
    print(f"Average time per frame: {total_time/len(frame_files):.3f}s")
    print(f"Output saved to: {output_gif}")
    print(f"{'='*60}")
    print("Done!")

if __name__ == "__main__":
    main()
