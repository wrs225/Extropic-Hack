"""
Parameter Sweep for Cameraman Image with Different Salt & Pepper Noise Levels

Tests salt and pepper noise levels: 0.05, 0.10, and 0.15
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import add_noise, add_salt_pepper_noise, compute_metrics, image_to_bitplanes, bitplanes_to_image

# Import the graph building functions and denoiser class
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram


def build_multilayer_4connected_graph(h, w, num_layers):
    """Build multi-layer 4-connected graph."""
    all_nodes = []
    for layer in range(num_layers):
        layer_nodes = [[SpinNode() for _ in range(w)] for _ in range(h)]
        all_nodes.append(layer_nodes)

    flat_nodes = sum([sum(layer, []) for layer in all_nodes], [])

    edges = []
    for layer in range(num_layers):
        layer_offset = layer * h * w
        def idx(i, j): return layer_offset + i * w + j

        for i in range(h):
            for j in range(w):
                u = idx(i, j)
                if j + 1 < w:
                    edges.append((flat_nodes[u], flat_nodes[idx(i, j+1)]))
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
        self.h = h
        self.w = w
        self.num_bits = num_bits
        self.num_layers = num_bits
        self.alpha_coef = alpha_coef
        self.beta_coef = beta_coef
        self.temperature = temperature

        # Build unified multi-layer graph
        self.all_nodes, self.flat_nodes, self.edges = build_multilayer_4connected_graph(h, w, self.num_layers)
        self.free_blocks, self.even_idx, self.odd_idx = build_multilayer_checkerboard(
            self.flat_nodes, h, w, self.num_layers
        )

        self.clamped_blocks = []
        self.state_clamp = []

        # Sampling schedule
        self.schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=steps_per_sample)

        # Create JIT-compiled denoising function
        self._denoise_jit = jax.jit(self._denoise_all_bitplanes)

    def _denoise_all_bitplanes(self, key, noisy_bitplanes_array, alpha_coef, beta_coef, temperature):
        """Denoise all 4 bit planes simultaneously."""
        h, w = self.h, self.w
        num_layers = self.num_layers

        # Flatten and concatenate all bit planes
        all_noisy_flat = []
        for layer in range(num_layers):
            noisy_flat = noisy_bitplanes_array[layer].flatten()
            all_noisy_flat.append(noisy_flat)

        all_noisy_flat = jnp.concatenate(all_noisy_flat)

        # Bit-plane specific weights
        alpha_values = jnp.array([alpha_coef * (2**b) for b in range(self.num_bits)])
        beta_values = jnp.array([beta_coef * (2**b) for b in range(self.num_bits)])

        # Build biases for all layers (fidelity to noisy input)
        biases_list = []
        for layer in range(num_layers):
            layer_start = layer * h * w
            layer_end = (layer + 1) * h * w
            noisy_spins = (all_noisy_flat[layer_start:layer_end].astype(jnp.float32) * 2.0 - 1.0)
            layer_biases = (alpha_values[layer] * noisy_spins).astype(jnp.float32)
            biases_list.append(layer_biases)

        biases = jnp.concatenate(biases_list)

        # Build edge weights (smoothness)
        edges_per_layer = len(self.edges) // num_layers
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

        # Warm-start from noisy data
        init_state_free = [
            all_noisy_flat[self.even_idx],
            all_noisy_flat[self.odd_idx],
        ]

        nodes_to_sample = [Block(self.flat_nodes)]

        # Sample
        samples_list = sample_states(
            key, program, self.schedule, init_state_free, self.state_clamp, nodes_to_sample
        )

        all_samples = samples_list[0]

        # Average over samples
        averaged = jnp.mean(all_samples, axis=0)
        denoised_all_flat_avg = (averaged > 0.5).astype(jnp.bool_)

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
        """Denoise 4-bit image using unified multi-layer Ising model."""
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
            _[0][0].block_until_ready()
            warmup_time = time.time() - warmup_start
            if verbose:
                print(f" {warmup_time:.3f}s")

        if verbose:
            print("  Denoising all 4 bit planes (JIT compiled)...", end='', flush=True)
        denoise_start = time.time()

        # Denoise all at once
        denoised_bitplanes_array, individual_samples = self._denoise_jit(key, noisy_bitplanes_array, alpha_coef, beta_coef, temperature)
        denoised_bitplanes_array[0].block_until_ready()

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


def load_cameraman_image(cameraman_path):
    """Load and convert cameraman image to 4-bit greyscale."""
    img = Image.open(cameraman_path)

    # Convert to greyscale
    img_grey = img.convert('L')

    # Resize to 256x256 for faster processing
    img_grey = img_grey.resize((256, 256), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img_grey)

    # Convert to 4-bit (0-15)
    img_4bit = (img_array / 255.0 * 15).astype(np.uint8)

    return img_4bit


def run_parameter_sweep(noisy_img, original_img, noise_level, output_dir):
    """Run parameter sweep for a specific noise level."""

    os.makedirs(output_dir, exist_ok=True)

    # Define parameter ranges
    alpha_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    beta_values = [0.3, 0.5, 0.7, 1.0, 1.5]

    results = []
    h, w = noisy_img.shape

    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP - CAMERAMAN IMAGE - SALT & PEPPER NOISE {noise_level}")
    print(f"{'='*70}")
    print(f"Testing {len(alpha_values)} × {len(beta_values)} = {len(alpha_values) * len(beta_values)} parameter combinations")
    print(f"Alpha (fidelity) values: {alpha_values}")
    print(f"Beta (smoothness) values: {beta_values}")
    print(f"{'='*70}")

    total_combinations = len(alpha_values) * len(beta_values)
    current = 0

    # JAX key for reproducibility
    key = jax.random.PRNGKey(42)

    # Initialize denoiser ONCE
    print("\nInitializing denoiser (building graph + JIT compilation)...")
    denoiser = MultiLayerDenoiser(
        h=h, w=w,
        n_warmup=20,
        n_samples=10,
        steps_per_sample=200,
        alpha_coef=0.5,
        beta_coef=1.0,
        temperature=1.0
    )
    print("Denoiser initialized!\n")

    # Run sweep
    for alpha in alpha_values:
        for beta in beta_values:
            current += 1
            print(f"[{current}/{total_combinations}] α={alpha:.1f}, β={beta:.1f}...", end='', flush=True)

            # Denoise with these parameters
            denoised_img, individual_imgs, denoise_time = denoiser.denoise_image(
                noisy_img,
                key=key,
                warm_up=(current == 1),
                verbose=False,
                alpha_coef=alpha,
                beta_coef=beta,
                temperature=1.0
            )

            # Compute metrics
            metrics = compute_metrics(original_img, noisy_img, denoised_img)

            # Store results
            result = {
                'alpha': float(alpha),
                'beta': float(beta),
                'psnr_noisy': float(metrics['psnr_noisy']),
                'psnr_denoised': float(metrics['psnr_denoised']),
                'psnr_improvement': float(metrics['psnr_improvement']),
                'denoise_time': float(denoise_time),
            }
            results.append(result)

            print(f" → PSNR: {metrics['psnr_denoised']:.2f} dB (+{metrics['psnr_improvement']:.2f} dB)")

            # Save this result's image
            img_filename = f"{output_dir}/denoised_a{alpha:.1f}_b{beta:.1f}.png"
            Image.fromarray((denoised_img * 17).astype(np.uint8)).save(img_filename)

    # Find best parameters
    best_result = max(results, key=lambda x: x['psnr_denoised'])

    print(f"\n{'='*70}")
    print("SWEEP COMPLETE!")
    print(f"{'='*70}")
    print(f"Best parameters for noise level {noise_level}:")
    print(f"  alpha_coef: {best_result['alpha']:.1f}")
    print(f"  beta_coef: {best_result['beta']:.1f}")
    print(f"  PSNR improvement: {best_result['psnr_improvement']:.2f} dB")
    print(f"  Final PSNR: {best_result['psnr_denoised']:.2f} dB")
    print(f"{'='*70}")

    # Save results to JSON
    results_file = f"{output_dir}/sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'noise_level': noise_level,
            'results': results,
            'best': best_result,
            'alpha_values': alpha_values,
            'beta_values': beta_values
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Create heatmap
    create_heatmap(results, alpha_values, beta_values, noise_level, output_dir)

    return results, best_result


def create_heatmap(results, alpha_values, beta_values, noise_level, output_dir):
    """Create heatmap visualization."""

    psnr_matrix = np.zeros((len(beta_values), len(alpha_values)))
    for result in results:
        alpha_idx = alpha_values.index(result['alpha'])
        beta_idx = beta_values.index(result['beta'])
        psnr_matrix[beta_idx, alpha_idx] = result['psnr_improvement']

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    im = ax.imshow(psnr_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(alpha_values)))
    ax.set_yticks(range(len(beta_values)))
    ax.set_xticklabels([f"{a:.1f}" for a in alpha_values])
    ax.set_yticklabels([f"{b:.1f}" for b in beta_values])
    ax.set_xlabel('Alpha (Fidelity Coefficient)', fontsize=12)
    ax.set_ylabel('Beta (Smoothness Coefficient)', fontsize=12)
    ax.set_title(f'PSNR Improvement (dB) - Cameraman - Salt&Pepper Noise={noise_level}', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(beta_values)):
        for j in range(len(alpha_values)):
            text = ax.text(j, i, f'{psnr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, label='PSNR Improvement (dB)')
    plt.tight_layout()

    heatmap_file = f"{output_dir}/heatmap_noise{noise_level}.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_file}")
    plt.close()


def main():
    """Run parameter sweeps for cameraman image with different salt & pepper noise levels."""

    # Load cameraman image
    print("Loading cameraman image...")
    cameraman_path = "/home/will/Extropic-Hack/data/cameraman.png"
    original_img = load_cameraman_image(cameraman_path)
    print(f"Loaded cameraman image: {original_img.shape}, range: [{original_img.min()}, {original_img.max()}]")

    # Save original
    os.makedirs("build/cameraman_sweeps", exist_ok=True)
    Image.fromarray((original_img * 17).astype(np.uint8)).save("build/cameraman_sweeps/original.png")

    # Test different noise levels
    noise_levels = [0.05, 0.10, 0.15]
    all_results = {}

    for noise_level in noise_levels:
        print(f"\n\n{'#'*70}")
        print(f"# TESTING NOISE LEVEL: {noise_level}")
        print(f"{'#'*70}")

        # Add salt and pepper noise
        np.random.seed(42)
        noisy_img = add_salt_pepper_noise(original_img, noise_level=noise_level)

        # Save noisy image
        noisy_path = f"build/cameraman_sweeps/noisy_{noise_level}.png"
        Image.fromarray((noisy_img * 17).astype(np.uint8)).save(noisy_path)
        print(f"Noisy image saved: {noisy_path}")

        # Run sweep
        output_dir = f"build/cameraman_sweeps/noise_{noise_level}"
        results, best_result = run_parameter_sweep(noisy_img, original_img, noise_level, output_dir)

        all_results[noise_level] = {
            'results': results,
            'best': best_result
        }

    # Summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY - CAMERAMAN IMAGE")
    print(f"{'='*70}")
    for noise_level in noise_levels:
        best = all_results[noise_level]['best']
        print(f"\nNoise level {noise_level}:")
        print(f"  Best α={best['alpha']:.1f}, β={best['beta']:.1f}")
        print(f"  PSNR: {best['psnr_denoised']:.2f} dB (+{best['psnr_improvement']:.2f} dB)")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
