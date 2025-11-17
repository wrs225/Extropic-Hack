# Pipeline Usage Guide

## Overview

The pipeline is an **agnostic infrastructure** for comparing different image denoising algorithms. It handles:
- ✅ Image data loading and storage
- ✅ GPU power monitoring (via JAX)
- ✅ Quality metrics calculation
- ✅ Side-by-side comparison visualization
- ✅ Result organization and export

## Quick Start

### 1. Implement Your Denoiser

Create a class that implements the `DenoiserInterface`:

```python
import numpy as np
from pipeline.core import DenoiserInterface

class MyDenoiser(DenoiserInterface):
    """Your custom denoising algorithm."""
    
    def __init__(self, param1=10, param2=20):
        """Initialize your algorithm."""
        self.param1 = param1
        self.param2 = param2
        # Setup your model/algorithm here
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise a single image.
        
        Args:
            image: Input noisy image, shape (H, W), values in [0, 1]
        
        Returns:
            Denoised image, same shape and range
        """
        # Your denoising logic here
        # Must return numpy array with same shape, values in [0, 1]
        denoised = ...  # Your algorithm
        return denoised
    
    def denoise_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Optional: Denoise batch of images (can default to sequential).
        """
        return [self.denoise(img) for img in images]
```

### 2. Use the Pipeline

```python
from pathlib import Path
from pipeline import DenoisingPipeline, ImageDataStore, ComparisonVisualizer
from my_denoiser import MyDenoiser

# Initialize pipeline (automatically sets up JAX GPU connection)
pipeline = DenoisingPipeline(gpu_id=0)

# Initialize data store
datastore = ImageDataStore(base_dir=Path("s0_data"))

# Load images
clean = datastore.load_image(Path("input/clean.png"))
noisy = datastore.load_image(Path("input/noisy.png"))

# Create your denoiser
denoiser = MyDenoiser(param1=10, param2=20)

# Process through pipeline
results = pipeline.process(
    denoiser=denoiser,
    clean_image=clean,
    noisy_image=noisy,
    metadata={
        'algorithm': 'MyDenoiser',
        'param1': 10,
        'param2': 20,
    }
)

# Save results
exp_dir = datastore.save_results(results, experiment_name="my_denoiser_test")

# Create comparison image
visualizer = ComparisonVisualizer()
visualizer.create_comparison(
    clean=results['clean'],
    noisy=results['noisy'],
    denoised=results['denoised'],
    output_path=exp_dir / "comparison.png",
    metrics=results['metrics'],
    algorithm_name="MyDenoiser",
)

print(f"Results saved to: {exp_dir}")
print(f"PSNR improvement: {results['metrics']['psnr_improvement']:.2f} dB")
print(f"Energy: {results['power']['mean_energy_j']:.3f} J")
```

## Example: Plugging in Bilateral Filter

```python
from pipeline import DenoisingPipeline, ImageDataStore
from s2a_baseline.bilateral_filter import BilateralFilterJAX

# Setup
pipeline = DenoisingPipeline(gpu_id=0)
datastore = ImageDataStore()

# Load images
clean = datastore.load_image(Path("test_image.png"))
noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)

# Create bilateral filter denoiser
bilateral = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)

# Process
results = pipeline.process(
    denoiser=bilateral,
    clean_image=clean,
    noisy_image=noisy,
    metadata={'algorithm': 'BilateralFilter', 'd': 11},
)

# Save
datastore.save_results(results, experiment_name="bilateral")
```

## Example: Plugging in ISling

```python
from pipeline import DenoisingPipeline
from s2b_isling import ISlingDenoiser  # Your ISling implementation

# Setup
pipeline = DenoisingPipeline(gpu_id=0)

# Create ISling denoiser
isling = ISlingDenoiser(config_path="isling_config.json")

# Process (same interface!)
results = pipeline.process(
    denoiser=isling,
    clean_image=clean,
    noisy_image=noisy,
    metadata={'algorithm': 'ISling'},
)
```

## Comparing Multiple Algorithms

```python
from pipeline import DenoisingPipeline, ImageDataStore
from s2a_baseline.bilateral_filter import BilateralFilterJAX
from s2b_isling import ISlingDenoiser

pipeline = DenoisingPipeline(gpu_id=0)
datastore = ImageDataStore()

# Load test image
clean = datastore.load_image(Path("test.png"))
noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)

# Test bilateral filter
bilateral = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)
results_bilateral = pipeline.process(
    denoiser=bilateral,
    clean_image=clean,
    noisy_image=noisy,
    metadata={'algorithm': 'BilateralFilter'},
)
datastore.save_results(results_bilateral, "bilateral_comparison")

# Test ISling
isling = ISlingDenoiser()
results_isling = pipeline.process(
    denoiser=isling,
    clean_image=clean,
    noisy_image=noisy,
    metadata={'algorithm': 'ISling'},
)
datastore.save_results(results_isling, "isling_comparison")

# Compare metrics
print("Bilateral Filter:")
print(f"  PSNR: {results_bilateral['metrics']['psnr_denoised']:.2f} dB")
print(f"  Energy: {results_bilateral['power']['mean_energy_j']:.3f} J")

print("ISling:")
print(f"  PSNR: {results_isling['metrics']['psnr_denoised']:.2f} dB")
print(f"  Energy: {results_isling['power']['mean_energy_j']:.3f} J")
```

## Results Structure

Each experiment saves:
```
s0_data/results/experiment_20250116_143022/
├── clean.png              # Clean reference image
├── noisy.png              # Noisy input image
├── denoised.png           # Denoised output
├── comparison.png         # Side-by-side comparison (if created)
├── metrics.json           # Quality metrics
└── results.json           # Full results (metrics + power + performance)
```

## Metrics Available

The pipeline automatically calculates:

**Quality Metrics:**
- `psnr_noisy`: PSNR of noisy image vs clean
- `psnr_denoised`: PSNR of denoised image vs clean
- `psnr_improvement`: Improvement in dB
- `ssim_noisy`: SSIM of noisy image
- `ssim_denoised`: SSIM of denoised image
- `ssim_improvement`: Improvement in SSIM
- `mse_reduction_pct`: Percentage of noise reduction

**Power Metrics:**
- `mean_energy_j`: Average energy consumption (Joules)
- `std_energy_j`: Standard deviation
- `mean_power_w`: Average power draw (Watts)
- `std_power_w`: Standard deviation

**Performance Metrics:**
- `mean_time_ms`: Average processing time (milliseconds)
- `std_time_ms`: Standard deviation
- `n_runs`: Number of benchmark runs

## GPU Connection

The pipeline automatically:
1. Verifies GPU availability via JAX
2. Sets up JAX to use GPU for all operations
3. Monitors GPU power during denoising
4. Ensures your denoiser runs on GPU (if using JAX)

**Important**: Your denoiser should use JAX arrays to ensure GPU usage:
```python
import jax.numpy as jnp

def denoise(self, image: np.ndarray) -> np.ndarray:
    # Convert to JAX array (will use GPU)
    jax_image = jnp.array(image)
    
    # Your JAX operations here (automatically on GPU)
    result = your_jax_function(jax_image)
    
    # Convert back to numpy
    return np.array(result)
```

## Command-Line Interface

You can also create a CLI script:

```python
# pipeline_cli.py
import argparse
from pathlib import Path
from pipeline import DenoisingPipeline, ImageDataStore
from my_denoiser import MyDenoiser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', required=True, help='Clean reference image')
    parser.add_argument('--noisy', required=True, help='Noisy input image')
    parser.add_argument('--algorithm', default='bilateral', help='Algorithm name')
    args = parser.parse_args()
    
    # Setup
    pipeline = DenoisingPipeline(gpu_id=0)
    datastore = ImageDataStore()
    
    # Load images
    clean = datastore.load_image(Path(args.clean))
    noisy = datastore.load_image(Path(args.noisy))
    
    # Create denoiser based on algorithm
    if args.algorithm == 'bilateral':
        from s2a_baseline.bilateral_filter import BilateralFilterJAX
        denoiser = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)
    elif args.algorithm == 'isling':
        from s2b_isling import ISlingDenoiser
        denoiser = ISlingDenoiser()
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Process
    results = pipeline.process(
        denoiser=denoiser,
        clean_image=clean,
        noisy_image=noisy,
        metadata={'algorithm': args.algorithm},
    )
    
    # Save
    exp_dir = datastore.save_results(results, experiment_name=args.algorithm)
    print(f"Results saved to: {exp_dir}")

if __name__ == '__main__':
    main()
```

## Tips

1. **Always use JAX arrays** in your denoiser to ensure GPU usage
2. **Warmup runs** are handled automatically by the pipeline
3. **Metrics are calculated automatically** - just provide clean, noisy, and denoised images
4. **Power monitoring** happens automatically during denoising
5. **Results are organized** by timestamp and experiment name
6. **Comparison images** are created automatically with all metrics

## Next Steps

1. Implement your denoiser following `DenoiserInterface`
2. Use `DenoisingPipeline.process()` to run it
3. Compare results using saved metrics JSON files
4. Visualize using comparison images

