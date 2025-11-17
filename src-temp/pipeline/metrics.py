"""Quality metrics calculation for pipeline."""

import numpy as np
from typing import Dict, Any


def calculate_psnr(image1: np.ndarray, image2: np.ndarray, max_value: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    
    sigma1_sq = np.var(image1)
    sigma2_sq = np.var(image2)
    sigma12 = np.mean((image1 - mu1) * (image2 - mu2))
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / (denominator + 1e-8)
    return float(ssim)


def calculate_all_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate all quality metrics.
    
    Args:
        clean: Clean reference image
        noisy: Noisy input image
        denoised: Denoised output image
    
    Returns:
        Dictionary with all metrics
    """
    # PSNR
    psnr_noisy = calculate_psnr(clean, noisy)
    psnr_denoised = calculate_psnr(clean, denoised)
    psnr_improvement = psnr_denoised - psnr_noisy
    
    # SSIM
    ssim_noisy = calculate_ssim(clean, noisy)
    ssim_denoised = calculate_ssim(clean, denoised)
    ssim_improvement = ssim_denoised - ssim_noisy
    
    # MSE
    mse_noisy = np.mean((clean - noisy) ** 2)
    mse_denoised = np.mean((clean - denoised) ** 2)
    if mse_noisy > 0:
        mse_reduction_pct = ((mse_noisy - mse_denoised) / mse_noisy) * 100.0
    else:
        mse_reduction_pct = 0.0
    
    return {
        'psnr_noisy': float(psnr_noisy),
        'psnr_denoised': float(psnr_denoised),
        'psnr_improvement': float(psnr_improvement),
        'ssim_noisy': float(ssim_noisy),
        'ssim_denoised': float(ssim_denoised),
        'ssim_improvement': float(ssim_improvement),
        'mse_noisy': float(mse_noisy),
        'mse_denoised': float(mse_denoised),
        'mse_reduction_pct': float(mse_reduction_pct),
    }

