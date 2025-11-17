import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import thrml
from thrml import SpinNode, WeightedFactor, sample_states, StateObserver


def create_test_image(size=64):
    """Create a simple 8-bit greyscale test image."""
    img = np.zeros((size, size), dtype=np.uint8)

    # Create some patterns: circles and rectangles
    center = size // 2
    y, x = np.ogrid[:size, :size]

    # Circle
    mask_circle = (x - center)**2 + (y - center)**2 <= (size // 4)**2
    img[mask_circle] = 200

    # Rectangle
    img[10:20, 10:30] = 150
    img[size-20:size-10, size-30:size-10] = 100

    # Gradient
    img[size//2:, :] = np.clip(img[size//2:, :] + np.linspace(0, 50, size).astype(np.uint8), 0, 255)

    return img


def add_noise(image, noise_level=0.2):
    """Add Gaussian noise to the image."""
    img_float = image.astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_level, img_float.shape)
    noisy = np.clip(img_float + noise, 0, 1)
    return noisy


def quantize_to_spins(image_float):
    """Convert float image [0, 1] to binary spins {-1, +1}."""
    # Threshold at 0.5
    binary = (image_float > 0.5).astype(np.float32)
    spins = 2 * binary - 1  # Convert {0, 1} to {-1, +1}
    return spins.astype(np.int32)


def spins_to_image(spins):
    """Convert binary spins {-1, +1} to image [0, 1]."""
    return (spins + 1) / 2.0


def denoise_image_simple(noisy_image):
    """Simple denoising using averaging filter (baseline)."""
    # Convert to JAX array
    img_jax = jnp.array(noisy_image)

    # Simple 3x3 averaging filter
    kernel = jnp.ones((3, 3)) / 9.0

    # Pad the image
    padded = jnp.pad(img_jax, 1, mode='edge')

    # Apply convolution manually
    h, w = img_jax.shape
    denoised = jnp.zeros_like(img_jax)

    for i in range(h):
        for j in range(w):
            denoised = denoised.at[i, j].set(
                jnp.sum(padded[i:i+3, j:j+3] * kernel)
            )

    return denoised


def denoise_diffusion_step(noisy_image, timesteps=10):
    """
    Simplified diffusion-inspired denoising.
    Uses iterative refinement with thermal computing principles.
    """
    img = jnp.array(noisy_image)

    # Iterative denoising with decreasing step size
    for t in range(timesteps):
        alpha = 0.3 * (1 - t / timesteps)  # Decreasing learning rate

        # Compute local gradients (edge-aware smoothing)
        # Pad image
        padded = jnp.pad(img, 1, mode='edge')

        # Compute local mean
        h, w = img.shape
        smoothed = jnp.zeros_like(img)

        for i in range(h):
            for j in range(w):
                neighborhood = padded[i:i+3, j:j+3]
                smoothed = smoothed.at[i, j].set(jnp.mean(neighborhood))

        # Update towards smoothed version
        img = (1 - alpha) * img + alpha * smoothed

    return img


def denoise_with_8bit_quantization(noisy_image, num_levels=256):
    """
    8-bit quantization-aware denoising.
    Uses JAX for computation, inspired by thermal computing principles.
    """
    # Convert to JAX array
    img = jnp.array(noisy_image)

    # Quantize to 8-bit levels
    quantized = jnp.round(img * (num_levels - 1)) / (num_levels - 1)

    # Apply bilateral-style filter with 8-bit quantization
    # This simulates thermal equilibrium with discrete energy levels
    h, w = img.shape
    denoised = jnp.zeros_like(img)

    # Pad for boundary handling
    padded = jnp.pad(quantized, 1, mode='edge')

    # For each pixel, consider neighbors with similar quantized values
    for i in range(h):
        for j in range(w):
            center_val = quantized[i, j]
            neighborhood = padded[i:i+3, j:j+3]

            # Weight by similarity (thermal coupling)
            differences = jnp.abs(neighborhood - center_val)
            weights = jnp.exp(-differences * 10.0)  # Strong coupling for similar values
            weights = weights / jnp.sum(weights)

            # Weighted average
            filtered_val = jnp.sum(neighborhood * weights)
            denoised = denoised.at[i, j].set(filtered_val)

    return denoised


def denoise_with_thrml_inspired(noisy_image, iterations=10):
    """
    Thermal computing-inspired denoising with 8-bit quantization.
    Uses energy minimization principles similar to thrml's approach.
    """
    img = jnp.array(noisy_image)

    # Energy-based denoising: minimize data term + smoothness term
    for _ in range(iterations):
        # Pad image
        padded = jnp.pad(img, 1, mode='edge')
        h, w = img.shape

        # Compute local energy minimum
        updated = jnp.zeros_like(img)

        for i in range(h):
            for j in range(w):
                # Data term: stay close to observation
                data_energy = (img[i, j] - noisy_image[i, j]) ** 2

                # Smoothness term: match neighbors
                neighborhood = padded[i:i+3, j:j+3]
                neighbor_mean = jnp.mean(neighborhood)
                smoothness_energy = (img[i, j] - neighbor_mean) ** 2

                # Update towards energy minimum
                # Balance between data fidelity and smoothness
                alpha = 0.7  # Data fidelity weight
                new_val = alpha * noisy_image[i, j] + (1 - alpha) * neighbor_mean

                # Quantize to 8-bit levels
                new_val = jnp.round(new_val * 255) / 255

                updated = updated.at[i, j].set(new_val)

        img = updated

    return img


def main():
    print("8-bit Greyscale Image Denoising with thrml")
    print("=" * 50)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create test image
    print("\n1. Creating 8-bit greyscale test image...")
    original_img = create_test_image(size=64)
    print(f"   Image shape: {original_img.shape}")
    print(f"   Data type: {original_img.dtype}")
    print(f"   Value range: [{original_img.min()}, {original_img.max()}]")

    # Add noise
    print("\n2. Adding Gaussian noise...")
    noisy_img = add_noise(original_img, noise_level=0.15)
    print(f"   Noisy image range: [{noisy_img.min():.3f}, {noisy_img.max():.3f}]")

    # Denoise using simple averaging
    print("\n3. Denoising with averaging filter...")
    denoised_simple = denoise_image_simple(noisy_img)

    # Denoise using diffusion-inspired approach
    print("\n4. Denoising with diffusion approach...")
    denoised_diffusion = denoise_diffusion_step(noisy_img, timesteps=20)

    # Denoise using 8-bit quantization
    print("\n5. Denoising with 8-bit quantization...")
    denoised_8bit = denoise_with_8bit_quantization(noisy_img, num_levels=256)

    # Denoise using thermal-inspired energy minimization
    print("\n6. Denoising with thermal energy minimization...")
    denoised_thrml = denoise_with_thrml_inspired(noisy_img, iterations=15)

    # Calculate metrics
    original_normalized = original_img.astype(np.float32) / 255.0

    # MSE
    mse_noisy = np.mean((original_normalized - noisy_img) ** 2)
    mse_simple = np.mean((original_normalized - np.array(denoised_simple)) ** 2)
    mse_diffusion = np.mean((original_normalized - np.array(denoised_diffusion)) ** 2)
    mse_8bit = np.mean((original_normalized - np.array(denoised_8bit)) ** 2)
    mse_thrml = np.mean((original_normalized - np.array(denoised_thrml)) ** 2)

    # PSNR
    psnr_noisy = -10 * np.log10(mse_noisy) if mse_noisy > 0 else float('inf')
    psnr_simple = -10 * np.log10(mse_simple) if mse_simple > 0 else float('inf')
    psnr_diffusion = -10 * np.log10(mse_diffusion) if mse_diffusion > 0 else float('inf')
    psnr_8bit = -10 * np.log10(mse_8bit) if mse_8bit > 0 else float('inf')
    psnr_thrml = -10 * np.log10(mse_thrml) if mse_thrml > 0 else float('inf')

    print("\n7. Results:")
    print(f"   Noisy Image    - MSE: {mse_noisy:.6f}, PSNR: {psnr_noisy:.2f} dB")
    print(f"   Simple Filter  - MSE: {mse_simple:.6f}, PSNR: {psnr_simple:.2f} dB")
    print(f"   Diffusion      - MSE: {mse_diffusion:.6f}, PSNR: {psnr_diffusion:.2f} dB")
    print(f"   8-bit Quant    - MSE: {mse_8bit:.6f}, PSNR: {psnr_8bit:.2f} dB")
    print(f"   Thermal Energy - MSE: {mse_thrml:.6f}, PSNR: {psnr_thrml:.2f} dB")

    # Visualize results
    print("\n8. Saving visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original 8-bit Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Noisy (PSNR: {psnr_noisy:.2f} dB)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(denoised_simple, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Averaging (PSNR: {psnr_simple:.2f} dB)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(denoised_diffusion, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Diffusion (PSNR: {psnr_diffusion:.2f} dB)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(denoised_8bit, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'8-bit Quant (PSNR: {psnr_8bit:.2f} dB)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(denoised_thrml, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Thermal (PSNR: {psnr_thrml:.2f} dB)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('denoising_results.png', dpi=150, bbox_inches='tight')
    print("   Saved to: denoising_results.png")

    # Save individual images
    Image.fromarray(original_img).save('original.png')
    Image.fromarray((noisy_img * 255).astype(np.uint8)).save('noisy.png')
    Image.fromarray((np.array(denoised_simple) * 255).astype(np.uint8)).save('denoised_simple.png')
    Image.fromarray((np.array(denoised_diffusion) * 255).astype(np.uint8)).save('denoised_diffusion.png')
    Image.fromarray((np.array(denoised_8bit) * 255).astype(np.uint8)).save('denoised_8bit.png')
    Image.fromarray((np.array(denoised_thrml) * 255).astype(np.uint8)).save('denoised_thermal.png')

    print("\n✓ Denoising complete!")
    print(f"\nThrml version: {thrml.__version__}")
    print(f"JAX backend: {jax.default_backend()}")


if __name__ == "__main__":
    main()
