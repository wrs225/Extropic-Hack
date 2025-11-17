"""
Parameter Sweep for Ising Denoiser

Sweeps over alpha_coef and beta_coef to find optimal smoothing parameters.
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

from utils import create_test_image, add_noise, compute_metrics, image_to_bitplanes, bitplanes_to_image

# Import the graph building functions and denoiser class
import jax.numpy as jnp
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


def run_parameter_sweep(noisy_img, original_img, output_dir="build/parameter_sweep"):
    """
    Run a parameter sweep over alpha_coef and beta_coef.

    Args:
        noisy_img: Noisy input image
        original_img: Original clean image (for computing metrics)
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define parameter ranges
    # alpha_coef: fidelity to noisy input (try 0.3 to 1.5)
    # beta_coef: smoothness (try 0.3 to 1.5)
    alpha_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    beta_values = [0.3, 0.5, 0.7, 1.0, 1.5]

    results = []
    h, w = noisy_img.shape

    print("="*70)
    print("PARAMETER SWEEP FOR ISING DENOISER")
    print("="*70)
    print(f"Testing {len(alpha_values)} × {len(beta_values)} = {len(alpha_values) * len(beta_values)} parameter combinations")
    print(f"Alpha (fidelity) values: {alpha_values}")
    print(f"Beta (smoothness) values: {beta_values}")
    print("="*70)

    total_combinations = len(alpha_values) * len(beta_values)
    current = 0

    # JAX key for reproducibility
    key = jax.random.PRNGKey(42)

    # Initialize denoiser ONCE - this builds the graph and JIT compiles
    print("\nInitializing denoiser (building graph + JIT compilation)...")
    denoiser = MultiLayerDenoiser(
        h=h, w=w,
        n_warmup=20,
        n_samples=10,
        steps_per_sample=5,
        alpha_coef=0.5,  # Default values, we'll override these
        beta_coef=1.0,
        temperature=1.0
    )
    print("Denoiser initialized!\n")

    # Run sweep - just change parameters, no re-initialization!
    for alpha in alpha_values:
        for beta in beta_values:
            current += 1
            print(f"[{current}/{total_combinations}] Testing alpha={alpha:.1f}, beta={beta:.1f}...", end='', flush=True)

            # Denoise with these parameters (no re-initialization!)
            denoised_img, individual_imgs, denoise_time = denoiser.denoise_image(
                noisy_img,
                key=key,
                warm_up=(current == 1),  # Only warm up once on first call
                verbose=False,
                alpha_coef=alpha,
                beta_coef=beta,
                temperature=1.0
            )

            # Compute metrics
            metrics = compute_metrics(original_img, noisy_img, denoised_img)

            # Store results (convert to Python floats for JSON serialization)
            result = {
                'alpha': float(alpha),
                'beta': float(beta),
                'psnr_noisy': float(metrics['psnr_noisy']),
                'psnr_denoised': float(metrics['psnr_denoised']),
                'psnr_improvement': float(metrics['psnr_improvement']),
                'denoise_time': float(denoise_time),
            }
            results.append(result)

            print(f" → PSNR: {metrics['psnr_denoised']:.2f} dB (+{metrics['psnr_improvement']:.2f} dB) in {denoise_time:.3f}s")

            # Save this result's image
            img_filename = f"{output_dir}/denoised_a{alpha:.1f}_b{beta:.1f}.png"
            Image.fromarray((denoised_img * 17).astype(np.uint8)).save(img_filename)

    # Find best parameters
    best_result = max(results, key=lambda x: x['psnr_denoised'])

    print("\n" + "="*70)
    print("SWEEP COMPLETE!")
    print("="*70)
    print(f"Best parameters:")
    print(f"  alpha_coef: {best_result['alpha']:.1f}")
    print(f"  beta_coef: {best_result['beta']:.1f}")
    print(f"  PSNR improvement: {best_result['psnr_improvement']:.2f} dB")
    print(f"  Final PSNR: {best_result['psnr_denoised']:.2f} dB")
    print("="*70)

    # Save results to JSON
    results_file = f"{output_dir}/sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'best': best_result,
            'alpha_values': alpha_values,
            'beta_values': beta_values
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Create visualization
    visualize_sweep_results(results, alpha_values, beta_values, output_dir)

    return results, best_result


def visualize_sweep_results(results, alpha_values, beta_values, output_dir):
    """Create heatmap visualization of sweep results."""

    # Create PSNR improvement matrix
    psnr_matrix = np.zeros((len(beta_values), len(alpha_values)))

    for result in results:
        alpha_idx = alpha_values.index(result['alpha'])
        beta_idx = beta_values.index(result['beta'])
        psnr_matrix[beta_idx, alpha_idx] = result['psnr_improvement']

    # Create heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PSNR Improvement heatmap
    im1 = axes[0].imshow(psnr_matrix, cmap='RdYlGn', aspect='auto')
    axes[0].set_xticks(range(len(alpha_values)))
    axes[0].set_yticks(range(len(beta_values)))
    axes[0].set_xticklabels([f"{a:.1f}" for a in alpha_values])
    axes[0].set_yticklabels([f"{b:.1f}" for b in beta_values])
    axes[0].set_xlabel('Alpha (Fidelity Coefficient)', fontsize=12)
    axes[0].set_ylabel('Beta (Smoothness Coefficient)', fontsize=12)
    axes[0].set_title('PSNR Improvement (dB)', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(beta_values)):
        for j in range(len(alpha_values)):
            text = axes[0].text(j, i, f'{psnr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im1, ax=axes[0], label='PSNR Improvement (dB)')

    # Final PSNR heatmap
    psnr_final_matrix = np.zeros((len(beta_values), len(alpha_values)))
    for result in results:
        alpha_idx = alpha_values.index(result['alpha'])
        beta_idx = beta_values.index(result['beta'])
        psnr_final_matrix[beta_idx, alpha_idx] = result['psnr_denoised']

    im2 = axes[1].imshow(psnr_final_matrix, cmap='viridis', aspect='auto')
    axes[1].set_xticks(range(len(alpha_values)))
    axes[1].set_yticks(range(len(beta_values)))
    axes[1].set_xticklabels([f"{a:.1f}" for a in alpha_values])
    axes[1].set_yticklabels([f"{b:.1f}" for b in beta_values])
    axes[1].set_xlabel('Alpha (Fidelity Coefficient)', fontsize=12)
    axes[1].set_ylabel('Beta (Smoothness Coefficient)', fontsize=12)
    axes[1].set_title('Final PSNR (dB)', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(beta_values)):
        for j in range(len(alpha_values)):
            text = axes[1].text(j, i, f'{psnr_final_matrix[i, j]:.2f}',
                              ha="center", va="center", color="white", fontsize=9)

    plt.colorbar(im2, ax=axes[1], label='Final PSNR (dB)')

    plt.tight_layout()
    heatmap_file = f"{output_dir}/parameter_sweep_heatmap.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_file}")
    plt.close()

    # Create comparison plot: beta/alpha ratio vs PSNR improvement
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ratios = [r['beta'] / r['alpha'] for r in results]
    improvements = [r['psnr_improvement'] for r in results]

    scatter = ax.scatter(ratios, improvements, c=improvements, cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Beta/Alpha Ratio (Smoothness/Fidelity)', fontsize=12)
    ax.set_ylabel('PSNR Improvement (dB)', fontsize=12)
    ax.set_title('Effect of Beta/Alpha Ratio on Denoising Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='PSNR Improvement (dB)')

    ratio_file = f"{output_dir}/ratio_analysis.png"
    plt.savefig(ratio_file, dpi=150, bbox_inches='tight')
    print(f"Ratio analysis saved to: {ratio_file}")
    plt.close()


def main():
    """Run parameter sweep on test image."""

    # Create test image
    print("Creating test image...")
    np.random.seed(42)
    original_img = create_test_image(size=256)

    # Add noise
    print("Adding noise...")
    noisy_img = add_noise(original_img, noise_level=0.15)

    # Save original and noisy
    os.makedirs("build/parameter_sweep", exist_ok=True)
    Image.fromarray((original_img * 17).astype(np.uint8)).save("build/parameter_sweep/original.png")
    Image.fromarray((noisy_img * 17).astype(np.uint8)).save("build/parameter_sweep/noisy.png")

    # Run sweep
    results, best_result = run_parameter_sweep(noisy_img, original_img)

    print("\nYou can now use the best parameters in your denoiser:")
    print(f"  denoiser = MultiLayerDenoiser(alpha_coef={best_result['alpha']:.1f}, beta_coef={best_result['beta']:.1f})")


if __name__ == "__main__":
    main()
