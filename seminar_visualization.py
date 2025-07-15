import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.LODOPAB import LODOPAB
from runners.diffusion import get_beta_schedule

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

def main():
    dataset = LODOPAB(dataroot='', img_size=128, split='test')
    image = dataset[1]['LD'].numpy()
    # Example usage of the forward diffusion visualization function
    visualize_forward_diffusion_process(image, num_steps=10, beta_start=0.0001, beta_end=0.15, schedule_type='linear', save_dir='seminar_visualization_results')

if __name__ == '__main__':
    main()
