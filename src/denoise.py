"""
4-bit Greyscale Denoising with UNIFIED Multi-Layer Ising Machine

All 4 bit planes processed simultaneously in a single Ising model.
- 4 layers, one per bit plane
- 4-connected within each layer
- No inter-layer connections
- Bit-plane specific weights: α_b = 0.5 * 2^b, β_b = 1.0 * 2^b
- JIT-compiled, averaged samples
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram


# Image parameters
H, W = 256, 256
NUM_BITS = 4  # 4-bit greyscale
NUM_LAYERS = NUM_BITS


def create_test_image(size=256):
    """Create a 4-bit greyscale test image."""
    img = np.zeros((size, size), dtype=np.uint8)

    center = size // 2
    y, x = np.ogrid[:size, :size]

    # Create patterns with 4-bit values (0-15)
    circles = [
        (center, center, size//3, 14),
        (size//4, size//4, size//8, 10),
        (3*size//4, 3*size//4, size//8, 6),
    ]

    for cy, cx, r, intensity in circles:
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = intensity

    # Rectangles
    img[20:60, 20:80] = 12
    img[size-60:size-20, size-80:size-20] = 8

    return img


def add_noise(image, noise_level=0.15):
    """Add Gaussian noise to 4-bit image."""
    img_float = image.astype(np.float32) / 15.0
    noise = np.random.normal(0, noise_level, img_float.shape)
    noisy = np.clip(img_float + noise, 0, 1)
    return (noisy * 15).astype(np.uint8)


def image_to_bitplanes(img, num_bits=4):
    """Decompose 4-bit image into binary bit planes."""
    bitplanes = []
    for bit in range(num_bits):
        plane = (img >> bit) & 1
        bitplanes.append(plane.astype(np.bool_))
    return bitplanes


def bitplanes_to_image(bitplanes):
    """Recombine binary bit planes into image."""
    img = np.zeros_like(bitplanes[0], dtype=np.uint8)
    for bit, plane in enumerate(bitplanes):
        img += (plane.astype(np.uint8) << bit)
    return img


def build_multilayer_4connected_graph(h, w, num_layers):
    """
    Build multi-layer 4-connected graph.
    - Each layer is a 4-connected grid
    - No connections between layers
    """
    # Create nodes for all layers
    all_nodes = []
    for layer in range(num_layers):
        layer_nodes = [[SpinNode() for _ in range(w)] for _ in range(h)]
        all_nodes.append(layer_nodes)

    flat_nodes = sum([sum(layer, []) for layer in all_nodes], [])

    # Build edges (4-connected within each layer only)
    edges = []

    for layer in range(num_layers):
        layer_offset = layer * h * w

        def idx(i, j): return layer_offset + i * w + j

        for i in range(h):
            for j in range(w):
                u = idx(i, j)
                # Right neighbor
                if j + 1 < w:
                    edges.append((flat_nodes[u], flat_nodes[idx(i, j+1)]))
                # Down neighbor
                if i + 1 < h:
                    edges.append((flat_nodes[u], flat_nodes[idx(i+1, j)]))

    return all_nodes, flat_nodes, edges


def build_multilayer_checkerboard(flat_nodes, h, w, num_layers):
    """Build checkerboard blocks across all layers."""
    even_idx = jnp.array([layer*h*w + i*w + j
                          for layer in range(num_layers)
                          for i in range(h)
                          for j in range(w)
                          if (i + j) % 2 == 0])
    odd_idx = jnp.array([layer*h*w + i*w + j
                         for layer in range(num_layers)
                         for i in range(h)
                         for j in range(w)
                         if (i + j) % 2 == 1])

    free_blocks = [
        Block([flat_nodes[int(k)] for k in even_idx]),
        Block([flat_nodes[int(k)] for k in odd_idx])
    ]

    return free_blocks, even_idx, odd_idx


class MultiLayerDenoiser:
    """Encapsulated 4-bit denoiser using multi-layer Ising machine."""

    def __init__(self, h=256, w=256, num_bits=4, n_warmup=20, n_samples=10, steps_per_sample=5):
        self.h = h
        self.w = w
        self.num_bits = num_bits
        self.num_layers = num_bits

        # Build unified multi-layer graph
        print("Building unified 4-layer, 4-connected graph...")
        self.all_nodes, self.flat_nodes, self.edges = build_multilayer_4connected_graph(h, w, self.num_layers)
        self.free_blocks, self.even_idx, self.odd_idx = build_multilayer_checkerboard(
            self.flat_nodes, h, w, self.num_layers
        )

        self.clamped_blocks = []
        self.state_clamp = []

        print(f"  Total nodes: {len(self.flat_nodes)} ({self.num_layers} layers × {h*w})")
        print(f"  Total edges: {len(self.edges)} (4-connected within layers)")
        print(f"  Even nodes: {len(self.even_idx)}, Odd nodes: {len(self.odd_idx)}")

        # Sampling schedule
        self.schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=steps_per_sample)

        # Create JIT-compiled denoising function
        self._denoise_jit = jax.jit(self._denoise_all_bitplanes)

    def _denoise_all_bitplanes(self, key, noisy_bitplanes_array):
        """
        Denoise all 4 bit planes simultaneously.

        Uses bit-plane specific weights:
        - α_b = 0.5 * 2^b (fidelity)
        - β_b = 1.0 * 2^b (smoothness)

        Args:
            key: JAX random key
            noisy_bitplanes_array: Shape (4, H, W) binary bit planes

        Returns:
            Denoised bit planes, shape (4, H, W)
        """
        h, w = self.h, self.w
        num_layers = self.num_layers

        # Flatten and concatenate all bit planes
        all_noisy_flat = []
        for layer in range(num_layers):
            noisy_flat = noisy_bitplanes_array[layer].flatten()
            all_noisy_flat.append(noisy_flat)

        all_noisy_flat = jnp.concatenate(all_noisy_flat)  # Shape: (4*H*W,)

        # Bit-plane specific weights
        # α_b = 0.5 * 2^b, β_b = 1.0 * 2^b
        alpha_values = jnp.array([0.5 * (2**b) for b in range(self.num_bits)])
        beta_values = jnp.array([1.0 * (2**b) for b in range(self.num_bits)])

        # Build biases for all layers (fidelity to noisy input)
        biases_list = []
        for layer in range(num_layers):
            layer_start = layer * h * w
            layer_end = (layer + 1) * h * w
            # Convert {0,1} to spins {-1,+1}
            noisy_spins = (all_noisy_flat[layer_start:layer_end].astype(jnp.float32) * 2.0 - 1.0)
            layer_biases = (alpha_values[layer] * noisy_spins).astype(jnp.float32)
            biases_list.append(layer_biases)

        biases = jnp.concatenate(biases_list)

        # Build edge weights (smoothness)
        # Each edge belongs to a specific layer, so assign appropriate β_b
        # Since edges are built layer-by-layer in order, we can assign weights accordingly
        edges_per_layer = len(self.edges) // num_layers  # Assuming equal edges per layer
        weights_list = []

        for layer in range(num_layers):
            layer_weights = jnp.ones((edges_per_layer,), dtype=jnp.float32) * beta_values[layer]
            weights_list.append(layer_weights)

        weights = jnp.concatenate(weights_list)

        # Temperature
        beta_inv = jnp.array(1.0, dtype=jnp.float32)

        # Create unified Ising model
        model = IsingEBM(self.flat_nodes, self.edges, biases, weights, beta_inv)
        program = IsingSamplingProgram(model, self.free_blocks, self.clamped_blocks)

        # Warm-start from noisy data (x = y)
        init_state_free = [
            all_noisy_flat[self.even_idx],
            all_noisy_flat[self.odd_idx],
        ]

        nodes_to_sample = [Block(self.flat_nodes)]

        # Sample
        samples_list = sample_states(
            key, program, self.schedule, init_state_free, self.state_clamp, nodes_to_sample
        )

        # Return both individual samples AND averaged
        all_samples = samples_list[0]  # Shape: (n_samples, 4*H*W)

        # Average over samples
        averaged = jnp.mean(all_samples, axis=0)  # Average over samples
        denoised_all_flat_avg = (averaged > 0.5).astype(jnp.bool_)  # Threshold

        # Reshape averaged to individual bit planes
        denoised_bitplanes_avg = []
        for layer in range(num_layers):
            layer_start = layer * h * w
            layer_end = (layer + 1) * h * w
            denoised_plane = denoised_all_flat_avg[layer_start:layer_end].reshape(h, w)
            denoised_bitplanes_avg.append(denoised_plane)

        # Also return individual samples (last 5 samples)
        individual_samples = []
        for sample_idx in range(max(0, all_samples.shape[0] - 5), all_samples.shape[0]):
            sample_bitplanes = []
            for layer in range(num_layers):
                layer_start = layer * h * w
                layer_end = (layer + 1) * h * w
                sample_plane = all_samples[sample_idx, layer_start:layer_end].reshape(h, w)
                sample_bitplanes.append(sample_plane)
            individual_samples.append(jnp.stack(sample_bitplanes))

        return jnp.stack(denoised_bitplanes_avg), individual_samples

    def denoise_image(self, noisy_img, key=None, warm_up=True):
        """Denoise 4-bit image using unified multi-layer Ising model."""
        if key is None:
            key = jax.random.PRNGKey(42)

        # Decompose into bit planes
        noisy_bitplanes = image_to_bitplanes(noisy_img, self.num_bits)

        # Stack into array
        noisy_bitplanes_array = jnp.stack([jnp.array(bp) for bp in noisy_bitplanes])

        # Warm-up call to trigger JIT compilation (discarded)
        if warm_up:
            print("  Warming up JIT (compiling)...", end='', flush=True)
            warmup_start = time.time()
            _ = self._denoise_jit(key, noisy_bitplanes_array)
            _[0][0].block_until_ready()  # Force computation
            warmup_time = time.time() - warmup_start
            print(f" {warmup_time:.3f}s")

        print("  Denoising all 4 bit planes (JIT compiled)...", end='', flush=True)
        denoise_start = time.time()

        # Denoise all at once - get averaged AND individual samples (now instant!)
        denoised_bitplanes_array, individual_samples = self._denoise_jit(key, noisy_bitplanes_array)
        denoised_bitplanes_array[0].block_until_ready()  # Force computation

        denoise_time = time.time() - denoise_start
        print(f" {denoise_time:.3f}s")

        # Convert averaged back to list
        denoised_bitplanes = [np.array(denoised_bitplanes_array[i]) for i in range(self.num_bits)]

        # Recombine averaged
        denoised_img = bitplanes_to_image(denoised_bitplanes)

        # Also recombine individual samples
        individual_imgs = []
        for sample_bitplanes_array in individual_samples:
            sample_bitplanes = [np.array(sample_bitplanes_array[i]) for i in range(self.num_bits)]
            sample_img = bitplanes_to_image(sample_bitplanes)
            individual_imgs.append(sample_img)

        return denoised_img, individual_imgs, denoise_time


def compute_metrics(original, noisy, denoised):
    """Compute PSNR metrics."""
    original_norm = original.astype(np.float32) / 15.0
    noisy_norm = noisy.astype(np.float32) / 15.0
    denoised_norm = denoised.astype(np.float32) / 15.0

    mse_noisy = np.mean((original_norm - noisy_norm) ** 2)
    mse_denoised = np.mean((original_norm - denoised_norm) ** 2)

    psnr_noisy = -10 * np.log10(mse_noisy) if mse_noisy > 0 else float('inf')
    psnr_denoised = -10 * np.log10(mse_denoised) if mse_denoised > 0 else float('inf')

    return {
        'psnr_noisy': psnr_noisy,
        'psnr_denoised': psnr_denoised,
        'psnr_improvement': psnr_denoised - psnr_noisy
    }


def main():
    print("="*70)
    print("UNIFIED 4-BIT DENOISING - ALL BIT PLANES IN ONE ISING MACHINE")
    print("="*70)
    print(f"Image size: {H}x{W}")
    print(f"Bit depth: {NUM_BITS} bits")
    print(f"Weights: α_b = 0.5 * 2^b, β_b = 1.0 * 2^b")
    print(f"Neighborhood: 4-connected (no inter-layer connections)")
    print(f"Sampling: {schedule.n_warmup} warmup + {schedule.n_samples} samples × {schedule.steps_per_sample} steps")
    print("="*70)

    # Create test image
    print("\n1. Creating test image...")
    np.random.seed(42)
    original_img = create_test_image(size=H)
    print(f"   Unique values: {len(np.unique(original_img))}, range: [{original_img.min()}, {original_img.max()}]")

    # Add noise
    print("\n2. Adding Gaussian noise...")
    noisy_img = add_noise(original_img, noise_level=0.15)
    print(f"   Unique values: {len(np.unique(noisy_img))}")

    # Denoise
    print("\n3. Denoising...")
    key = jax.random.PRNGKey(42)

    denoised_img, individual_imgs, denoise_time = denoise_4bit_image(key, noisy_img)

    print(f"   Denoised unique values: {len(np.unique(denoised_img))}")
    print(f"   Got {len(individual_imgs)} individual samples")

    # Metrics
    print("\n4. Computing metrics...")
    metrics = compute_metrics(original_img, noisy_img, denoised_img)

    print(f"   Noisy PSNR:     {metrics['psnr_noisy']:.2f} dB")
    print(f"   Denoised PSNR:  {metrics['psnr_denoised']:.2f} dB (averaged)")
    print(f"   Improvement:    {metrics['psnr_improvement']:.2f} dB")

    # Compute metrics for individual samples
    print("\n   Individual sample PSNRs:")
    for i, sample_img in enumerate(individual_imgs):
        sample_metrics = compute_metrics(original_img, noisy_img, sample_img)
        print(f"     Sample {i}: {sample_metrics['psnr_denoised']:.2f} dB")

    # Visualize
    print("\n5. Saving results...")

    # Main comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = 15

    axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=vmax)
    axes[0].set_title('Original (4-bit)')
    axes[0].axis('off')

    axes[1].imshow(noisy_img, cmap='gray', vmin=0, vmax=vmax)
    axes[1].set_title(f'Noisy\nPSNR: {metrics["psnr_noisy"]:.2f} dB')
    axes[1].axis('off')

    axes[2].imshow(denoised_img, cmap='gray', vmin=0, vmax=vmax)
    axes[2].set_title(f'Denoised (Averaged)\nPSNR: {metrics["psnr_denoised"]:.2f} dB (+{metrics["psnr_improvement"]:.2f} dB)')
    axes[2].axis('off')

    # Create build folder
    os.makedirs('build', exist_ok=True)

    plt.tight_layout()
    plt.savefig('build/result_unified.png', dpi=150, bbox_inches='tight')
    print("   Saved: build/result_unified.png")

    # Save images
    Image.fromarray((original_img * 17).astype(np.uint8)).save('build/original.png')
    Image.fromarray((noisy_img * 17).astype(np.uint8)).save('build/noisy.png')
    Image.fromarray((denoised_img * 17).astype(np.uint8)).save('build/denoised.png')
    print("   Saved: build/original.png, noisy.png, denoised.png")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"  Time: {denoise_time:.3f}s")
    print(f"  PSNR improvement: {metrics['psnr_improvement']:.2f} dB")
    print(f"  Greyscale levels: {len(np.unique(denoised_img))}/16")
    print()


if __name__ == "__main__":
    main()
