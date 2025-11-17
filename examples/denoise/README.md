# 4-bit Greyscale Denoising with Multi-Layer Ising Machine

This example demonstrates image denoising using a multi-layer Ising machine on 4-bit greyscale images.

## Overview

The denoiser processes all 4 bit planes simultaneously in a single unified Ising model:
- **4 layers**, one per bit plane
- **4-connected** grid within each layer
- **No inter-layer connections**
- **Bit-plane specific weights**: α_b = 0.5 × 2^b (fidelity), β_b = 1.0 × 2^b (smoothness)
- **JIT-compiled** with JAX for fast inference

## Files

- `denoiser.py` - Main `MultiLayerDenoiser` class
- `utils.py` - Utility functions (bit plane conversion, metrics, etc.)
- `demo.py` - Simple demonstration with synthetic test image
- `benchmark.py` - Comprehensive benchmark with standard test images

## Usage

### Run Demo

```bash
cd examples/denoise
python demo.py
```

### Run Benchmark

The benchmark tests on standard images:
- **Cameraman (256×256)**: Sharp edges + large flat areas
- **Barbara (512×512)**: Woven cloth patterns (hardest test)
- **Boat (512×512)**: Mix of straight edges + natural textures
- **Peppers (512×512)**: Smooth gradients + sharp reflections

```bash
cd examples/denoise
python benchmark.py
```

Results are saved to `build/benchmark_*/` directories.

## How It Works

### Bit Plane Decomposition

4-bit images (values 0-15) are decomposed into 4 binary bit planes:
- Bit 0 (LSB): represents ±1
- Bit 1: represents ±2
- Bit 2: represents ±4
- Bit 3 (MSB): represents ±8

### Energy-Based Denoising

Each bit plane is denoised using an Ising model that balances:
1. **Data fidelity**: Stay close to noisy observations
2. **Spatial smoothness**: Match neighboring pixels

The energy function for bit plane b is:

```
E_b = -α_b Σᵢ hᵢ sᵢ - β_b Σ_{ij} wᵢⱼ sᵢ sⱼ
```

where:
- `hᵢ` = noisy observation at pixel i
- `sᵢ` = denoised spin at pixel i
- `wᵢⱼ` = edge weight (smoothness coupling)
- `α_b = 0.5 × 2^b` (higher bits have stronger data fidelity)
- `β_b = 1.0 × 2^b` (higher bits have stronger smoothness)

### Sampling

The model uses Gibbs sampling with a checkerboard update scheme:
- Warm-up phase to reach equilibrium
- Multiple samples collected
- Samples averaged and thresholded to produce final result

## Performance

Typical performance on a GPU:
- **256×256 image**: ~0.1s (after JIT compilation)
- **512×512 image**: ~0.3s (after JIT compilation)

PSNR improvements of 3-8 dB depending on noise level and image characteristics.

## Requirements

- JAX (with GPU support recommended)
- thrml >= 0.1.3
- NumPy
- Pillow
- Matplotlib
