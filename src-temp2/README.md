# Agnostic Denoising Pipeline

A clean, modular pipeline infrastructure for comparing image denoising algorithms with GPU power monitoring.

## Quick Start

```bash
# Run all examples
python main.py

# Run specific examples
python main.py --bilateral    # Bilateral filter example
python main.py --custom       # Custom denoiser example
python main.py --input        # Load from data/input/
python main.py --compare       # Compare multiple algorithms
```

## Directory Structure

```
src-temp2/
├── pipeline/              # Pipeline infrastructure
│   ├── core.py           # DenoisingPipeline
│   ├── datastore.py      # ImageDataStore
│   ├── benchmark.py      # GPU monitoring
│   ├── metrics.py        # Quality metrics
│   └── comparison.py     # Visualization
├── data/
│   ├── input/            # ← Place your input images here
│   ├── output/           # ← Results are saved here
│   └── raw/              # Optional: raw images
├── main.py               # Main entry point (run this!)
└── README.md             # This file
```

## Usage

### 1. Place Input Images

Put your images in `data/input/`:
```bash
cp your_image.png data/input/
```

### 2. Run Pipeline

```bash
python main.py --input
```

### 3. Check Results

Results are saved to `data/output/experiment_*/`:
- `comparison.png` - Visual comparison
- `metrics.json` - Quality metrics
- `results.json` - Full results with GPU power

## Implementing Your Own Denoiser

Just implement a class with a `denoise()` method:

```python
class MyDenoiser:
    def denoise(self, image: np.ndarray) -> np.ndarray:
        # Your algorithm here
        # Input: (H, W) array, values [0, 1]
        # Output: (H, W) array, values [0, 1]
        return denoised_image
```

Then use it with the pipeline:

```python
from pipeline import DenoisingPipeline, ImageDataStore

pipeline = DenoisingPipeline(gpu_id=0)
datastore = ImageDataStore()

# Load images
clean = datastore.load_image(Path("data/input/your_image.png"))
noisy = add_gaussian_noise(clean, sigma=0.15)

# Process
denoiser = MyDenoiser()
results = pipeline.process(denoiser, clean, noisy, 
                          metadata={'algorithm': 'MyDenoiser'})

# Save
datastore.save_results(results, "my_experiment")
```

## Features

- ✅ **GPU-First**: Automatic JAX GPU setup and power monitoring
- ✅ **Plug-and-Play**: Easy to swap different algorithms
- ✅ **Comprehensive Metrics**: Quality (PSNR, SSIM) + Energy + Performance
- ✅ **Visualization**: Automatic side-by-side comparison images
- ✅ **Organized Storage**: Timestamped experiment directories

## See Also

- `pipeline/USAGE.md` - Detailed usage guide
- `pipeline/README.md` - Pipeline architecture
- `LEARNINGS.md` - What we learned from implementation

