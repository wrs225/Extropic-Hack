"""
Multi-Layer Ising Machine Denoiser for 4-bit Greyscale Images

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
import time

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from .utils import image_to_bitplanes, bitplanes_to_image


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

    def __init__(self, h=256, w=256, num_bits=4, n_warmup=20, n_samples=10, steps_per_sample=5,
                 alpha_coef=0.5, beta_coef=1.0, temperature=1.0):
        """
        Args:
            h: Image height
            w: Image width
            num_bits: Number of bits (4 for 4-bit greyscale)
            n_warmup: Number of warmup steps for sampling
            n_samples: Number of samples to collect
            steps_per_sample: Steps between samples
            alpha_coef: Fidelity coefficient (α_b = alpha_coef * 2^b). Higher = more faithful to noisy input
            beta_coef: Smoothness coefficient (β_b = beta_coef * 2^b). Higher = more smoothing
            temperature: Temperature parameter (beta_inv). Higher = less aggressive optimization
        """
        self.h = h
        self.w = w
        self.num_bits = num_bits
        self.num_layers = num_bits
        self.alpha_coef = alpha_coef
        self.beta_coef = beta_coef
        self.temperature = temperature

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

    def _denoise_all_bitplanes(self, key, noisy_bitplanes_array, alpha_coef, beta_coef, temperature):
        """
        Denoise all 4 bit planes simultaneously.

        Uses bit-plane specific weights:
        - α_b = alpha_coef * 2^b (fidelity)
        - β_b = beta_coef * 2^b (smoothness)

        Args:
            key: JAX random key
            noisy_bitplanes_array: Shape (4, H, W) binary bit planes
            alpha_coef: Fidelity coefficient
            beta_coef: Smoothness coefficient
            temperature: Temperature parameter

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
        # α_b = alpha_coef * 2^b, β_b = beta_coef * 2^b
        alpha_values = jnp.array([alpha_coef * (2**b) for b in range(self.num_bits)])
        beta_values = jnp.array([beta_coef * (2**b) for b in range(self.num_bits)])

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
        beta_inv = jnp.array(temperature, dtype=jnp.float32)

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

    def denoise_image(self, noisy_img, key=None, warm_up=True, verbose=True,
                     alpha_coef=None, beta_coef=None, temperature=None):
        """Denoise 4-bit image using unified multi-layer Ising model.

        Args:
            noisy_img: Noisy input image
            key: JAX random key
            warm_up: Whether to do warmup JIT compilation
            verbose: Whether to print progress
            alpha_coef: Override fidelity coefficient (uses default if None)
            beta_coef: Override smoothness coefficient (uses default if None)
            temperature: Override temperature (uses default if None)
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        # Use instance defaults if not overridden
        alpha_coef = alpha_coef if alpha_coef is not None else self.alpha_coef
        beta_coef = beta_coef if beta_coef is not None else self.beta_coef
        temperature = temperature if temperature is not None else self.temperature

        # Decompose into bit planes
        noisy_bitplanes = image_to_bitplanes(noisy_img, self.num_bits)

        # Stack into array
        noisy_bitplanes_array = jnp.stack([jnp.array(bp) for bp in noisy_bitplanes])

        # Warm-up call to trigger JIT compilation (discarded)
        if warm_up:
            if verbose:
                print("  Warming up JIT (compiling)...", end='', flush=True)
            warmup_start = time.time()
            _ = self._denoise_jit(key, noisy_bitplanes_array, alpha_coef, beta_coef, temperature)
            _[0][0].block_until_ready()  # Force computation
            warmup_time = time.time() - warmup_start
            if verbose:
                print(f" {warmup_time:.3f}s")

        if verbose:
            print("  Denoising all 4 bit planes (JIT compiled)...", end='', flush=True)
        denoise_start = time.time()

        # Denoise all at once - get averaged AND individual samples (now instant!)
        denoised_bitplanes_array, individual_samples = self._denoise_jit(key, noisy_bitplanes_array, alpha_coef, beta_coef, temperature)
        denoised_bitplanes_array[0].block_until_ready()  # Force computation

        denoise_time = time.time() - denoise_start
        if verbose:
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
