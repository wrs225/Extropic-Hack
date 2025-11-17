"""Core pipeline infrastructure - agnostic to specific denoising algorithms."""

from typing import Optional, Dict, Any, Protocol
import numpy as np
import jax


# Protocol for denoising algorithms - any class with these methods will work
class DenoiserInterface(Protocol):
    """
    Protocol for denoising algorithms.
    
    Any denoising algorithm should implement this interface to work with the pipeline.
    Simply implement a class with `denoise()` method (and optionally `denoise_batch()`).
    """
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise a single image.
        
        Args:
            image: Input noisy image, shape (H, W), values in [0, 1]
        
        Returns:
            Denoised image, same shape and range as input
        """
        ...
    
    def denoise_batch(self, images: list) -> list:
        """
        Denoise a batch of images (optional, can default to sequential).
        
        Args:
            images: List of input noisy images
        
        Returns:
            List of denoised images
        """
        ...


class DenoisingPipeline:
    """
    Agnostic pipeline for image denoising with GPU monitoring.
    
    This pipeline handles:
    - Data loading and preprocessing
    - GPU verification and JAX setup
    - Running denoising algorithms
    - Power monitoring
    - Metrics calculation
    - Result storage and visualization
    """
    
    def __init__(self, gpu_id: int = 0):
        """
        Initialize pipeline.
        
        Args:
            gpu_id: GPU device ID to use
        """
        self.gpu_id = gpu_id
        self._verify_gpu()
        self._setup_jax()
    
    def _verify_gpu(self) -> bool:
        """Verify GPU is available."""
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'cuda' in str(d).lower() or 'gpu' in str(d).lower()]
        
        if len(gpu_devices) == 0:
            print("Warning: No GPU devices found. Will use CPU (slower).")
            return False
        else:
            print(f"Found {len(gpu_devices)} GPU device(s): {gpu_devices}")
            return True
    
    def _setup_jax(self):
        """Setup JAX for GPU usage."""
        # Ensure JAX uses GPU
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        
        # Set default device if GPU available
        gpu_devices = [d for d in devices if 'cuda' in str(d).lower() or 'gpu' in str(d).lower()]
        if len(gpu_devices) > 0:
            # JAX will automatically use GPU for operations
            print(f"Using GPU device: {gpu_devices[0]}")
    
    def process(
        self,
        denoiser: Any,  # DenoiserInterface - using Any for flexibility
        clean_image: np.ndarray,
        noisy_image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.
        
        Args:
            denoiser: Denoising algorithm with denoise() method
            clean_image: Clean reference image
            noisy_image: Noisy input image
            metadata: Optional metadata (algorithm name, parameters, etc.)
        
        Returns:
            Dictionary with results including denoised image and metrics
        """
        from .benchmark import PipelineBenchmark
        from .metrics import calculate_all_metrics
        
        # Run denoising with power monitoring
        benchmark = PipelineBenchmark(gpu_id=self.gpu_id)
        result = benchmark.run(denoiser.denoise, noisy_image)
        
        # Get denoised image
        denoised = result['output']
        
        # Calculate all metrics
        metrics = calculate_all_metrics(clean_image, noisy_image, denoised)
        
        # Combine results
        return {
            'denoised': denoised,
            'clean': clean_image,
            'noisy': noisy_image,
            'metrics': metrics,
            'power': result['power'],
            'performance': result['performance'],
            'metadata': metadata or {},
        }
    
    def process_batch(
        self,
        denoiser: Any,  # DenoiserInterface
        clean_images: list,
        noisy_images: list,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list:
        """
        Process a batch of images.
        
        Args:
            denoiser: Denoising algorithm
            clean_images: List of clean reference images
            noisy_images: List of noisy input images
            metadata: Optional metadata
        
        Returns:
            List of result dictionaries
        """
        results = []
        for clean, noisy in zip(clean_images, noisy_images):
            result = self.process(denoiser, clean, noisy, metadata)
            results.append(result)
        return results

