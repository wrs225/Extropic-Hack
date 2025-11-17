"""
Utility functions for image denoising
"""

import numpy as np


def image_to_bitplanes(img, num_bits=4):
    """Decompose 4-bit image into binary bit planes."""
    bitplanes = []
    for bit in range(num_bits):
        plane = (img >> bit) & 1
        bitplanes.append(plane.astype(np.bool_))
    return bitplanes


def bitplanes_to_image(bitplanes):
    """Recombine binary bit planes into image."""
    img = np.zeros_like(bitplanes[0], dtype=np.uint8)
    for bit, plane in enumerate(bitplanes):
        img += (plane.astype(np.uint8) << bit)
    return img


def add_noise(image, noise_level=0.15):
    """Add Gaussian noise to 4-bit image."""
    img_float = image.astype(np.float32) / 15.0
    noise = np.random.normal(0, noise_level, img_float.shape)
    noisy = np.clip(img_float + noise, 0, 1)
    return (noisy * 15).astype(np.uint8)


def add_salt_pepper_noise(image, noise_level=0.15):
    """Add salt and pepper noise to 4-bit image.

    Args:
        image: 4-bit image (values 0-15)
        noise_level: Probability of a pixel being corrupted (half salt, half pepper)

    Returns:
        Noisy image with salt (15) and pepper (0) noise
    """
    noisy = image.copy()

    # Generate random mask for noise
    noise_mask = np.random.random(image.shape) < noise_level

    # Split noise into salt and pepper
    salt_mask = noise_mask & (np.random.random(image.shape) < 0.5)
    pepper_mask = noise_mask & ~salt_mask

    # Apply salt (max value = 15 for 4-bit)
    noisy[salt_mask] = 15

    # Apply pepper (min value = 0)
    noisy[pepper_mask] = 0

    return noisy


def compute_metrics(original, noisy, denoised):
    """Compute PSNR metrics."""
    original_norm = original.astype(np.float32) / 15.0
    noisy_norm = noisy.astype(np.float32) / 15.0
    denoised_norm = denoised.astype(np.float32) / 15.0

    mse_noisy = np.mean((original_norm - noisy_norm) ** 2)
    mse_denoised = np.mean((original_norm - denoised_norm) ** 2)

    psnr_noisy = -10 * np.log10(mse_noisy) if mse_noisy > 0 else float('inf')
    psnr_denoised = -10 * np.log10(mse_denoised) if mse_denoised > 0 else float('inf')

    return {
        'psnr_noisy': psnr_noisy,
        'psnr_denoised': psnr_denoised,
        'psnr_improvement': psnr_denoised - psnr_noisy
    }


def create_test_image(size=256):
    """Create a 4-bit greyscale test image."""
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
