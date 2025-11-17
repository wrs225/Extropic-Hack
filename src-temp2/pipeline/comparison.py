"""Comparison visualization for pipeline results."""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ComparisonVisualizer:
    """Create side-by-side comparison images with metrics."""
    
    def create_comparison(
        self,
        clean: np.ndarray,
        noisy: np.ndarray,
        denoised: np.ndarray,
        output_path: Path,
        metrics: Optional[Dict[str, Any]] = None,
        algorithm_name: str = "Denoiser",
    ) -> None:
        """
        Create side-by-side comparison image with labels.
        
        Args:
            clean: Clean reference image
            noisy: Noisy input image
            denoised: Denoised output image
            output_path: Path to save comparison image
            metrics: Optional metrics dictionary
            algorithm_name: Name of denoising algorithm
        """
        # Convert to uint8
        def to_uint8(img):
            img_clipped = np.clip(img, 0.0, 1.0)
            return (img_clipped * 255.0).astype(np.uint8)
        
        clean_uint8 = to_uint8(clean)
        noisy_uint8 = to_uint8(noisy)
        denoised_uint8 = to_uint8(denoised)
        
        # Get dimensions
        H, W = clean_uint8.shape
        padding = 10
        label_height = 40
        spacing = 20
        
        # Create combined image
        total_width = W * 3 + spacing * 2 + padding * 2
        total_height = H + label_height * 2 + padding * 2
        
        combined = Image.new('RGB', (total_width, total_height), color='white')
        
        # Paste images
        x_offset = padding
        y_offset = padding + label_height
        
        combined.paste(Image.fromarray(clean_uint8, mode='L'), (x_offset, y_offset))
        x_offset += W + spacing
        combined.paste(Image.fromarray(noisy_uint8, mode='L'), (x_offset, y_offset))
        x_offset += W + spacing
        combined.paste(Image.fromarray(denoised_uint8, mode='L'), (x_offset, y_offset))
        
        # Add labels
        draw = ImageDraw.Draw(combined)
        
        # Try to load fonts
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
        draw.text((padding + 2 * (W + spacing) + W // 2, y_label_top), f"{algorithm_name} (Output)", 
                  fill='black', anchor='mt', font=font_large)
        
        # Bottom labels with metrics
        y_label_bottom = y_offset + H + 10
        draw.text((padding + W // 2, y_label_bottom), "Original", 
                  fill='gray', anchor='mt', font=font_small)
        
        # Noisy metrics
        x_noisy = padding + W + spacing + W // 2
        y_current = y_label_bottom
        if metrics and 'psnr_noisy' in metrics:
            draw.text((x_noisy, y_current), f"PSNR: {metrics['psnr_noisy']:.2f} dB", 
                      fill='gray', anchor='mt', font=font_small)
        else:
            draw.text((x_noisy, y_current), "Noisy", 
                      fill='gray', anchor='mt', font=font_small)
        
        # Denoised metrics
        x_metrics = padding + 2 * (W + spacing) + W // 2
        y_current = y_label_bottom
        draw.text((x_metrics, y_current), algorithm_name, 
                  fill='gray', anchor='mt', font=font_small)
        
        if metrics:
            if 'psnr_denoised' in metrics:
                y_current += 18
                improvement_text = f"PSNR: {metrics['psnr_denoised']:.2f} dB"
                if 'psnr_improvement' in metrics:
                    improvement_text += f" (Δ{metrics['psnr_improvement']:+.2f})"
                draw.text((x_metrics, y_current), improvement_text, 
                          fill='gray', anchor='mt', font=font_small)
            
            if 'ssim_denoised' in metrics:
                y_current += 18
                ssim_text = f"SSIM: {metrics['ssim_denoised']:.4f}"
                if 'ssim_improvement' in metrics:
                    ssim_text += f" (Δ{metrics['ssim_improvement']:+.4f})"
                draw.text((x_metrics, y_current), ssim_text, 
                          fill='gray', anchor='mt', font=font_small)
            
            if 'mse_reduction_pct' in metrics:
                y_current += 18
                draw.text((x_metrics, y_current), f"Noise ↓ {metrics['mse_reduction_pct']:.1f}%", 
                          fill='green', anchor='mt', font=font_small)
        
        # Save
        combined.save(output_path)

