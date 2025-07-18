import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.LODOPAB import LODOPAB
from runners.diffusion import get_beta_schedule
import torch.nn.functional as F

# Global dictionary to store attention maps
attention_maps = {}

def attention_hook(name):
    """Create a hook function to capture attention weights"""
    def hook(module, input, output):
        # Capture attention weights if available
        if hasattr(module, 'attn'):
            # For self-attention modules, extract attention weights
            attention_maps[name] = module.attn.detach().cpu()
        elif isinstance(output, tuple) and len(output) > 1:
            # Some attention modules return (output, attention_weights)
            if output[1] is not None:
                attention_maps[name] = output[1].detach().cpu()
    return hook

def register_attention_hooks(model):
    """Register hooks on attention modules to capture attention maps"""
    hooks = []
    for name, module in model.named_modules():
        # Look for attention modules
        if any(keyword in name.lower() for keyword in ['attn', 'attention']):
            hook = module.register_forward_hook(attention_hook(name))
            hooks.append(hook)
            print(f"Registered attention hook on: {name}")
    return hooks

def remove_hooks(hooks):
    """Remove all registered hooks"""
    for hook in hooks:
        hook.remove()

def visualize_pixel_intensity_histogram(save_dir, bins=256, sample_size=None):
    """
    Calculate and display histograms for pixel intensities across the training dataset.
    
    Args:
        save_dir (str): Directory to save the histogram plots
        bins (int): Number of bins for the histogram
        sample_size (int): Number of samples to use (None for entire dataset)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load training dataset
    dataset = LODOPAB(dataroot='', img_size=128, split='train')
    
    # Determine sample size
    total_samples = len(dataset)
    if sample_size is None or sample_size > total_samples:
        sample_size = total_samples
    
    print(f"Analyzing pixel intensities for {sample_size} samples from training dataset...")
    
    # Collect pixel values
    ld_pixels = []  # Low-dose pixels
    fd_pixels = []  # Full-dose pixels
    
    for i in range(sample_size):
        if i % 100 == 0:
            print(f"Processing sample {i+1}/{sample_size}")
        
        sample = dataset[i]
        ld_image = sample['LD'].numpy()
        fd_image = sample['FD'].numpy()
        
        # Flatten and collect pixels
        ld_pixels.extend(ld_image.flatten())
        fd_pixels.extend(fd_image.flatten())
    
    # Convert to numpy arrays
    ld_pixels = np.array(ld_pixels)
    fd_pixels = np.array(fd_pixels)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot histograms
    axes[0, 0].hist(ld_pixels, bins=bins, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_title('Low-Dose CT Pixel Intensity Distribution')
    axes[0, 0].set_xlabel('Pixel Intensity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(fd_pixels, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_title('Full-Dose CT Pixel Intensity Distribution')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overlay comparison
    axes[1, 0].hist(ld_pixels, bins=bins, alpha=0.5, color='red', label='Low-Dose', density=True)
    axes[1, 0].hist(fd_pixels, bins=bins, alpha=0.5, color='blue', label='Full-Dose', density=True)
    axes[1, 0].set_title('Normalized Pixel Intensity Comparison')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics table
    stats_text = f"""
    Low-Dose Statistics:
    Mean: {np.mean(ld_pixels):.4f}
    Std: {np.std(ld_pixels):.4f}
    Min: {np.min(ld_pixels):.4f}
    Max: {np.max(ld_pixels):.4f}
    
    Full-Dose Statistics:
    Mean: {np.mean(fd_pixels):.4f}
    Std: {np.std(fd_pixels):.4f}
    Min: {np.min(fd_pixels):.4f}
    Max: {np.max(fd_pixels):.4f}
    
    Samples analyzed: {sample_size}
    Total pixels: {len(ld_pixels):,}
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Dataset Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pixel_intensity_histogram.png"), dpi=150, bbox_inches='tight')
    # plt.savefig(os.path.join(save_dir, "pixel_intensity_histogram.pdf"), bbox_inches='tight')
    
    print(f"Pixel intensity histogram saved in: {save_dir}")
    print("Statistics:")
    print(f"Low-Dose - Mean: {np.mean(ld_pixels):.4f}, Std: {np.std(ld_pixels):.4f}")
    print(f"Full-Dose - Mean: {np.mean(fd_pixels):.4f}, Std: {np.std(fd_pixels):.4f}")

def visualize_forward_diffusion_process(image, num_steps, beta_start, beta_end, schedule_type, save_dir):
    """Visualize the forward diffusion process in a single plot."""
    os.makedirs(save_dir, exist_ok=True)

    # Normalize the input image to range (-1, 1)
    image_min, image_max = image.min(), image.max()
    image = (image - image_min) / (image_max - image_min + 1e-8) * 2 - 1

    # Convert image to tensor
    x = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()

    # Generate betas using Fast-DDPM logic
    betas = get_beta_schedule(beta_schedule=schedule_type, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_steps)
    betas = torch.tensor(betas, dtype=torch.float32)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Create a figure for visualization
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))

    # Forward diffusion process
    for t in range(num_steps):
        alpha_t = alphas_cumprod[t]
        noise = torch.randn_like(x)
        x_noised = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

        # Save intermediate steps
        step_image = x_noised.squeeze().cpu().numpy()
        step_image = np.clip((step_image + 1) / 2, 0, 1)
        axes[t].imshow(step_image, cmap='gray', vmin=0, vmax=1)
        axes[t].axis('off')
        axes[t].set_title(f"Step {t+1}")

        # Update x for next step
        x = x_noised

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "forward_diffusion_process.png"), dpi=150, bbox_inches='tight')
    print(f"Forward diffusion process visualized and saved in: {save_dir}")

def visualize_denoising_attention_progression(image, num_steps, beta_start, beta_end, schedule_type, save_dir, attention_steps=[2, 4, 6, 8]):
    """
    Visualize how attention patterns change during different steps of the denoising process.
    
    Args:
        image: Input low-dose image
        num_steps: Total number of denoising steps
        beta_start, beta_end, schedule_type: Diffusion parameters
        save_dir: Directory to save visualizations
        attention_steps: Which denoising steps to show attention for
    """
    global attention_maps
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure image is 2D
    if len(image.shape) == 3:
        image = image.squeeze()
    
    # Normalize the input image to range (-1, 1)
    image_min, image_max = image.min(), image.max()
    image_normalized = (image - image_min) / (image_max - image_min + 1e-8) * 2 - 1
    
    # Convert image to tensor
    x = torch.tensor(image_normalized).unsqueeze(0).unsqueeze(0).float()
    
    # Generate betas using Fast-DDPM logic
    betas = get_beta_schedule(beta_schedule=schedule_type, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_steps)
    betas = torch.tensor(betas, dtype=torch.float32)
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Create figure for progression visualization
    num_attention_steps = len(attention_steps)
    fig, axes = plt.subplots(3, num_attention_steps + 1, figsize=((num_attention_steps + 1) * 3, 9))
    
    # Show original image in first column (ensure 2D)
    original_display = np.clip((image_normalized + 1) / 2, 0, 1)
    axes[0, 0].imshow(original_display, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_display, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    axes[2, 0].imshow(original_display, cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title('Original Image')
    axes[2, 0].axis('off')
    
    # Simulate forward diffusion and capture attention at specific steps
    x_current = x.clone()
    
    for col_idx, step in enumerate(attention_steps):
        attention_maps.clear()
        
        # Apply noise up to this step
        alpha_t = alphas_cumprod[step - 1]  # step-1 because of 0-indexing
        noise = torch.randn_like(x)
        x_noised = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        
        # Show noisy image at this step
        step_image = x_noised.squeeze().cpu().numpy()
        step_image_display = np.clip((step_image + 1) / 2, 0, 1)
        
        axes[0, col_idx + 1].imshow(step_image_display, cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx + 1].set_title(f'Step {step}/{num_steps}\n(Noisy Image)')
        axes[0, col_idx + 1].axis('off')
        
        # Simulate attention maps (placeholder - in real implementation, this would come from the model)
        # For demonstration, create synthetic attention maps that focus on different regions
        attention_map = create_synthetic_attention_map(step_image_display, step, num_steps)
        
        # Show raw attention map
        im1 = axes[1, col_idx + 1].imshow(attention_map, cmap='hot', vmin=0, vmax=1)
        axes[1, col_idx + 1].set_title(f'Attention Map\nStep {step}')
        axes[1, col_idx + 1].axis('off')
        
        # Show attention overlay
        axes[2, col_idx + 1].imshow(step_image_display, cmap='gray', vmin=0, vmax=1)
        axes[2, col_idx + 1].imshow(attention_map, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        axes[2, col_idx + 1].set_title(f'Attention Overlay\nStep {step}')
        axes[2, col_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "denoising_attention_progression.png"), dpi=150, bbox_inches='tight')
    print(f"Denoising attention progression saved in: {save_dir}")

def create_synthetic_attention_map(image, step, total_steps):
    """
    Create a synthetic attention map for demonstration purposes.
    In real implementation, this would be replaced with actual attention weights from the model.
    """
    h, w = image.shape
    
    # Create different attention patterns based on denoising step
    if step <= 2:
        # Early steps: focus on high-frequency regions (edges)
        from scipy import ndimage
        edges = ndimage.sobel(image)
        attention = np.abs(edges)
        # Add some global attention
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        center_mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) // 4)**2))
        attention = 0.7 * attention + 0.3 * center_mask
        
    elif step <= 4:
        # Middle steps: focus on anatomical structures (simulated)
        # Create attention around medium-intensity regions
        intensity_mask = np.exp(-((image - 0.5)**2) / (2 * 0.2**2))
        # Add some spatial structure
        y, x = np.ogrid[:h, :w]
        spatial_pattern = np.sin(2 * np.pi * x / w) * np.cos(2 * np.pi * y / h)
        attention = 0.6 * intensity_mask + 0.4 * (spatial_pattern + 1) / 2
        
    elif step <= 6:
        # Later middle steps: focus on fine details
        # Create attention around areas with moderate gradients
        from scipy import ndimage
        grad_x = ndimage.sobel(image, axis=1)
        grad_y = ndimage.sobel(image, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        attention = gradient_magnitude
        # Add some random patches for detail refinement
        random_attention = np.random.random((h, w)) * 0.3
        attention = 0.7 * attention + 0.3 * random_attention
        
    else:
        # Final steps: focus on global structure and consistency
        # Create smoother, more global attention patterns
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        global_pattern = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) // 3)**2))
        # Add some structured patterns
        wave_pattern = (np.sin(4 * np.pi * x / w) * np.sin(4 * np.pi * y / h) + 1) / 2
        attention = 0.6 * global_pattern + 0.4 * wave_pattern
    
    # Normalize attention map
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    # Apply some smoothing to make it more realistic
    from scipy import ndimage
    attention = ndimage.gaussian_filter(attention, sigma=1.0)
    
    return attention

def compare_attention_across_images(save_dir, num_images=3, attention_steps=[3, 5, 7]):
    """
    Compare attention patterns across different images at the same denoising steps.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load different images from the dataset
    dataset = LODOPAB(dataroot='', img_size=128, split='test')
    
    fig, axes = plt.subplots(num_images, len(attention_steps) + 1, figsize=((len(attention_steps) + 1) * 3, num_images * 3))
    
    for img_idx in range(num_images):
        image = dataset[img_idx]['LD'].numpy()
        
        # Ensure image is 2D
        if len(image.shape) == 3:
            image = image.squeeze()
        
        # Show original image
        image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        axes[img_idx, 0].imshow(image_normalized, cmap='gray', vmin=0, vmax=1)
        axes[img_idx, 0].set_title(f'Image {img_idx + 1}')
        axes[img_idx, 0].axis('off')
        
        # Show attention at different steps
        for step_idx, step in enumerate(attention_steps):
            # Create synthetic attention for this image and step
            attention_map = create_synthetic_attention_map(image_normalized, step, 10)
            
            # Show attention overlay
            axes[img_idx, step_idx + 1].imshow(image_normalized, cmap='gray', vmin=0, vmax=1)
            axes[img_idx, step_idx + 1].imshow(attention_map, cmap='hot', alpha=0.6, vmin=0, vmax=1)
            
            if img_idx == 0:
                axes[img_idx, step_idx + 1].set_title(f'Attention Step {step}')
            axes[img_idx, step_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_comparison_across_images.png"), dpi=150, bbox_inches='tight')
    print(f"Attention comparison across images saved in: {save_dir}")

def main():
    dataset = LODOPAB(dataroot='', img_size=128, split='test')
    
    image = dataset[1]['LD'].numpy()
    
    # Ensure image is 2D
    if len(image.shape) == 3:
        image = image.squeeze()
    
    # Visualize forward diffusion process
    visualize_forward_diffusion_process(
        image, 
        num_steps=10, 
        beta_start=0.0001, 
        beta_end=0.5, 
        schedule_type='linear', 
        save_dir='seminar_visualization_results'
    )
    
    # Visualize attention progression during denoising
    # visualize_denoising_attention_progression(
    #     image,
    #     num_steps=10,
    #     beta_start=0.0001,
    #     beta_end=0.15,
    #     schedule_type='linear',
    #     save_dir='seminar_visualization_results',
    #     attention_steps=[2, 4, 6, 8]
    # )
    
    # Compare attention across different images
    # compare_attention_across_images(
    #     save_dir='seminar_visualization_results',
    #     num_images=3,
    #     attention_steps=[3, 5, 7]
    # )
    
    # Pixel intensity histogram (uncomment if needed)
    # visualize_pixel_intensity_histogram(save_dir='seminar_visualization_results', bins=256, sample_size=1000)

if __name__ == '__main__':
    main()
