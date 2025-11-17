"""Example usage of the agnostic pipeline."""

from pathlib import Path
import numpy as np
from pipeline import DenoisingPipeline, ImageDataStore, ComparisonVisualizer
from s1_input.noise_gen import add_gaussian_noise


# Example 1: Using the existing Bilateral Filter
def example_bilateral_filter():
    """Example using bilateral filter with the pipeline."""
    from s2a_baseline.bilateral_filter import BilateralFilterJAX
    
    # Initialize pipeline (sets up JAX GPU connection)
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
    print(f"\nResults saved to: {exp_dir}")
    print(f"PSNR improvement: {results['metrics']['psnr_improvement']:.2f} dB")
    print(f"SSIM improvement: {results['metrics']['ssim_improvement']:.4f}")
    print(f"Energy: {results['power']['mean_energy_j']:.3f} J")
    print(f"Time: {results['performance']['mean_time_ms']:.2f} ms")
    
    return results


# Example 2: Custom denoiser implementation
class SimpleGaussianDenoiser:
    """Simple example denoiser using Gaussian blur."""
    
    def __init__(self, sigma=1.0):
        self.sigma = sigma
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Simple Gaussian blur denoising."""
        from scipy import ndimage
        denoised = ndimage.gaussian_filter(image, sigma=self.sigma)
        return np.clip(denoised, 0.0, 1.0)
    
    def denoise_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        return [self.denoise(img) for img in images]


def example_custom_denoiser():
    """Example using a custom denoiser."""
    pipeline = DenoisingPipeline(gpu_id=0)
    datastore = ImageDataStore()
    
    # Load or create test image
    clean = np.random.rand(256, 256).astype(np.float32)
    noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)
    
    # Create custom denoiser
    denoiser = SimpleGaussianDenoiser(sigma=1.5)
    
    # Process
    results = pipeline.process(
        denoiser=denoiser,
        clean_image=clean,
        noisy_image=noisy,
        metadata={'algorithm': 'SimpleGaussian', 'sigma': 1.5},
    )
    
    # Save
    exp_dir = datastore.save_results(results, "gaussian_example")
    print(f"Results: {exp_dir}")
    
    return results


# Example 3: Comparing multiple algorithms
def example_compare_algorithms():
    """Compare multiple denoising algorithms."""
    pipeline = DenoisingPipeline(gpu_id=0)
    datastore = ImageDataStore()
    
    # Create test image
    clean = np.ones((256, 256)) * 0.5
    clean[64:192, 64:192] = 0.8
    noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)
    
    algorithms = []
    
    # Test 1: Bilateral Filter
    from s2a_baseline.bilateral_filter import BilateralFilterJAX
    bilateral = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)
    results_bilateral = pipeline.process(
        denoiser=bilateral,
        clean_image=clean,
        noisy_image=noisy,
        metadata={'algorithm': 'BilateralFilter'},
    )
    algorithms.append(('Bilateral Filter', results_bilateral))
    
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
    print("\n" + "="*60)
    print("Algorithm Comparison")
    print("="*60)
    for name, results in algorithms:
        print(f"\n{name}:")
        print(f"  PSNR: {results['metrics']['psnr_denoised']:.2f} dB (improvement: {results['metrics']['psnr_improvement']:+.2f} dB)")
        print(f"  SSIM: {results['metrics']['ssim_denoised']:.4f} (improvement: {results['metrics']['ssim_improvement']:+.4f})")
        print(f"  Energy: {results['power']['mean_energy_j']:.3f} J")
        print(f"  Time: {results['performance']['mean_time_ms']:.2f} ms")
    
    return algorithms


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'bilateral':
            example_bilateral_filter()
        elif sys.argv[1] == 'custom':
            example_custom_denoiser()
        elif sys.argv[1] == 'compare':
            example_compare_algorithms()
        else:
            print("Usage: python example_usage.py [bilateral|custom|compare]")
    else:
        # Run all examples
        print("Running bilateral filter example...")
        example_bilateral_filter()
        
        print("\n" + "="*60)
        print("Running custom denoiser example...")
        example_custom_denoiser()
        
        print("\n" + "="*60)
        print("Running comparison example...")
        example_compare_algorithms()

