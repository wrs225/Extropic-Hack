# Agnostic Image Denoising Pipeline

A clean, modular pipeline infrastructure for comparing image denoising algorithms with GPU power monitoring.

## Features

- 🔌 **Plug-and-Play**: Easy to swap different denoising algorithms
- ⚡ **GPU-First**: Automatic JAX GPU setup and power monitoring
- 📊 **Comprehensive Metrics**: Quality (PSNR, SSIM) + Energy + Performance
- 🖼️ **Visualization**: Automatic side-by-side comparison images
- 💾 **Data Management**: Organized storage with timestamped experiments
- 🔬 **Reproducible**: Fixed seeds, JSON export, full metadata

## Architecture

```
pipeline/
├── core.py           # DenoisingPipeline and DenoiserInterface
├── datastore.py      # ImageDataStore for loading/saving
├── benchmark.py      # GPUPowerMonitor and PipelineBenchmark
├── metrics.py        # Quality metrics calculation
├── comparison.py     # ComparisonVisualizer
└── USAGE.md          # Detailed usage guide
```

## Quick Example

```python
from pipeline import DenoisingPipeline, ImageDataStore
from my_denoiser import MyDenoiser

# Setup
pipeline = DenoisingPipeline(gpu_id=0)
datastore = ImageDataStore()

# Load images
clean = datastore.load_image(Path("clean.png"))
noisy = datastore.load_image(Path("noisy.png"))

# Create your denoiser
denoiser = MyDenoiser()

# Process (automatically handles GPU monitoring, metrics, etc.)
results = pipeline.process(
    denoiser=denoiser,
    clean_image=clean,
    noisy_image=noisy,
    metadata={'algorithm': 'MyDenoiser'},
)

# Save results
exp_dir = datastore.save_results(results, "my_experiment")
print(f"Results: {exp_dir}")
print(f"PSNR improvement: {results['metrics']['psnr_improvement']:.2f} dB")
print(f"Energy: {results['power']['mean_energy_j']:.3f} J")
```

## Implementing a Denoiser

Your denoiser just needs to implement:

```python
class MyDenoiser:
    def denoise(self, image: np.ndarray) -> np.ndarray:
        # Input: (H, W) array, values in [0, 1]
        # Output: (H, W) array, values in [0, 1]
        return denoised_image
```

That's it! The pipeline handles everything else.

See `USAGE.md` for detailed examples and documentation.

