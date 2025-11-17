# Agnostic Denoising Pipeline - Summary

## What We Built

A **clean, modular pipeline infrastructure** that separates algorithm implementation from benchmarking infrastructure. This allows you to easily plug in different denoising algorithms (bilateral filter, ISling, or any custom algorithm) and compare them fairly.

## Key Components

### 1. **Pipeline Core** (`pipeline/core.py`)
- `DenoisingPipeline`: Main orchestrator
- `DenoiserInterface`: Protocol for algorithms (just needs `denoise()` method)
- Automatic JAX GPU setup and verification
- Handles all the infrastructure so you just focus on your algorithm

### 2. **Data Store** (`pipeline/datastore.py`)
- `ImageDataStore`: Manages image loading/saving
- Organized storage with timestamped experiments
- Automatic result organization

### 3. **Benchmarking** (`pipeline/benchmark.py`)
- `GPUPowerMonitor`: Context manager for power monitoring
- `PipelineBenchmark`: Runs algorithms with power/performance tracking
- Automatic warmup runs for JIT compilation
- Statistical analysis (mean, std dev)

### 4. **Metrics** (`pipeline/metrics.py`)
- Quality metrics: PSNR, SSIM, MSE reduction
- Improvement metrics: Shows how much better denoised is vs noisy
- All calculated automatically

### 5. **Visualization** (`pipeline/comparison.py`)
- `ComparisonVisualizer`: Creates side-by-side comparison images
- Automatic labeling with metrics
- Perfect for demos and presentations

## Architecture

```
pipeline/
├── core.py           # DenoisingPipeline - main orchestrator
├── datastore.py      # ImageDataStore - data management
├── benchmark.py       # GPU power monitoring & benchmarking
├── metrics.py        # Quality metrics calculation
├── comparison.py     # Visualization
├── USAGE.md          # Detailed usage guide
└── example_usage.py  # Working examples
```

## How to Use (Plug & Play)

### Step 1: Implement Your Denoiser

```python
class MyDenoiser:
    def denoise(self, image: np.ndarray) -> np.ndarray:
        # Your algorithm here
        # Input: (H, W) array, values [0, 1]
        # Output: (H, W) array, values [0, 1]
        return denoised_image
```

### Step 2: Use the Pipeline

```python
from pipeline import DenoisingPipeline, ImageDataStore

# Setup (automatically connects to GPU via JAX)
pipeline = DenoisingPipeline(gpu_id=0)
datastore = ImageDataStore()

# Load images
clean = datastore.load_image(Path("clean.png"))
noisy = datastore.load_image(Path("noisy.png"))

# Create your denoiser
denoiser = MyDenoiser()

# Process (handles everything: GPU monitoring, metrics, etc.)
results = pipeline.process(
    denoiser=denoiser,
    clean_image=clean,
    noisy_image=noisy,
    metadata={'algorithm': 'MyDenoiser'},
)

# Save results
exp_dir = datastore.save_results(results, "my_experiment")
```

That's it! The pipeline handles:
- ✅ GPU power monitoring
- ✅ Quality metrics (PSNR, SSIM, MSE reduction)
- ✅ Performance metrics (time, energy)
- ✅ Result storage
- ✅ Comparison visualization

## Key Features

### 1. **GPU-First Design**
- Automatic JAX GPU setup
- Real-time power monitoring via pynvml/nvidia-ml-py
- Ensures your algorithm runs on GPU (if using JAX)

### 2. **Agnostic Interface**
- Any algorithm with `denoise(image)` method works
- No need to modify pipeline code
- Easy to swap algorithms

### 3. **Comprehensive Metrics**
- Quality: PSNR, SSIM, MSE reduction
- Energy: Joules, Watts, energy/pixel
- Performance: Latency, throughput

### 4. **Organized Results**
- Timestamped experiment directories
- JSON export for easy comparison
- Side-by-side visualization

## Example: Comparing Algorithms

```python
# Test Bilateral Filter
bilateral = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)
results_bilateral = pipeline.process(bilateral, clean, noisy, 
                                     metadata={'algorithm': 'Bilateral'})

# Test ISling
isling = ISlingDenoiser()
results_isling = pipeline.process(isling, clean, noisy,
                                   metadata={'algorithm': 'ISling'})

# Compare
print(f"Bilateral: {results_bilateral['metrics']['psnr_improvement']:.2f} dB, "
      f"{results_bilateral['power']['mean_energy_j']:.3f} J")
print(f"ISling: {results_isling['metrics']['psnr_improvement']:.2f} dB, "
      f"{results_isling['power']['mean_energy_j']:.3f} J")
```

## What We Learned

1. **JAX GPU Integration**: Use `static_argnums` for compile-time constants
2. **Power Monitoring**: Context manager pattern with pynvml/nvidia-ml-py
3. **Metrics**: Always compare noisy vs denoised (improvement metrics)
4. **Visualization**: Side-by-side comparisons are essential for demos
5. **Modularity**: Separate infrastructure from algorithms for flexibility

## File Structure

```
src-temp/
├── pipeline/              # NEW: Agnostic pipeline infrastructure
│   ├── core.py           # DenoisingPipeline
│   ├── datastore.py      # ImageDataStore
│   ├── benchmark.py      # GPU monitoring
│   ├── metrics.py        # Quality metrics
│   ├── comparison.py    # Visualization
│   ├── USAGE.md          # Usage guide
│   └── example_usage.py  # Examples
├── s2a_baseline/         # Bilateral filter (example algorithm)
├── s2b_isling/           # ISling (your algorithm)
└── LEARNINGS.md          # What we learned
```

## Next Steps

1. **Implement ISling**: Create `s2b_isling/isling_denoiser.py` with `denoise()` method
2. **Use Pipeline**: Run both algorithms through the same pipeline
3. **Compare**: Use saved JSON files and comparison images
4. **Iterate**: Tune parameters and compare results

The pipeline is ready to use - just implement your denoiser and plug it in!

