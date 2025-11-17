# Implementation Learnings and Best Practices

## Key Learnings from Baseline Implementation

### 1. JAX GPU Integration
- **JIT Compilation**: Must use `static_argnums` for compile-time constants (like filter diameter `d`)
- **Python Conditionals**: Cannot use `if` statements inside `@jax.jit` decorated functions - normalize values before JIT
- **Device Detection**: Use string matching (`'cuda' in str(device)`) rather than `device_kind` attribute for compatibility
- **GPU Verification**: Always verify GPU availability before benchmarking

### 2. Bilateral Filter Parameters
- **More Aggressive Denoising**:
  - Lower `sigma_color` (20-30) = tighter intensity matching = more aggressive
  - Higher `sigma_space` (100-150) = more spatial smoothing
  - Larger `d` (11-13) = larger neighborhood = more smoothing
- **Parameter Trade-offs**: More aggressive = less noise but may blur details

### 3. Power Monitoring
- **pynvml vs nvidia-ml-py**: Use `nvidia-ml-py` (newer package, same API)
- **Context Manager Pattern**: Use `with GPUPowerMonitor():` for clean resource management
- **Sampling Rate**: 100ms (0.1s) is good balance between accuracy and overhead
- **Energy Calculation**: Integrate power over time using trapezoidal rule

### 4. Metrics and Quality Assessment
- **Improvement Metrics**: Always compare noisy vs denoised (not just absolute values)
- **PSNR Improvement**: Shows dB gain from denoising
- **SSIM Improvement**: Shows structural similarity gain
- **MSE Reduction**: Percentage of noise reduction
- **Visual Comparison**: Side-by-side images with metrics are essential for demos

### 5. Dependency Management with uv
- **Path Dependencies**: Cannot use TOML table syntax in dependency list - use direct JAX dependency
- **Isolation**: Create separate `pyproject.toml` in subdirectory for isolated dependencies
- **JAX Sharing**: Specify same version constraint to allow uv to reuse existing installation

### 6. Image Processing Pipeline
- **3-bit Quantization**: Convert to 3-bit (0-7) for processing, use float32 internally
- **Normalization**: Always normalize to [0, 1] for processing, denormalize for display
- **Reproducible Noise**: Use fixed seeds for reproducible experiments
- **Batch Processing**: Support both single images and video sequences (frame-by-frame)

### 7. Common Pitfalls Avoided
- ✅ Correct image order: clean → noisy → denoised (not swapped)
- ✅ Normalize sigma values before JIT compilation
- ✅ Use `static_argnums` for compile-time constants
- ✅ Handle multiline text in PIL by drawing lines separately
- ✅ Export all metrics to JSON for easy comparison

## Architecture Insights

### Pipeline Structure
```
s0_data/        - Data storage (raw images, results)
s1_input/      - Data loading and preprocessing (agnostic)
s2a_baseline/   - Specific algorithm implementation
s2b_isling/     - Alternative algorithm implementation
s3_results/     - Benchmarking, metrics, visualization (agnostic)
```

### Key Design Principles
1. **Separation of Concerns**: Pipeline infrastructure separate from algorithms
2. **Agnostic Interfaces**: Algorithms implement standard interface
3. **GPU-First**: All processing assumes GPU availability with fallback
4. **Comprehensive Metrics**: Quality + Energy + Performance
5. **Reproducibility**: Fixed seeds, timestamped results, JSON export

## Performance Characteristics

### Typical Baseline Performance (H200, 256x256, 3-bit)
- **Processing Time**: 15-30 ms per image
- **Energy**: 1-3 Joules per image
- **Power**: 80-120 Watts during processing
- **PSNR Improvement**: 3-8 dB typical
- **SSIM Improvement**: 0.1-0.3 typical

### Optimization Opportunities
- JIT compilation critical for performance
- Batch processing can improve throughput
- Larger filters = more computation but better quality
- Parameter tuning is image-dependent

