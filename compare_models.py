import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml
import argparse
from tqdm import tqdm
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from lpips import LPIPS
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.filters import sobel

# Import models and utilities
from train_unet_denoiser import UNet, SimpleConfig, EMAHelper
from datasets.LODOPAB import LODOPAB
from runners.diffusion import Diffusion as DiffusionRunner
from models.diffusion import Model as DiffusionModel


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleConfig(config_dict)


def load_unet_model(model_path, config, device, use_ema=False):
    """Load UNet model from checkpoint"""
    model = UNet(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load EMA weights if available and requested
        if use_ema and 'ema_state_dict' in checkpoint:
            ema_helper = EMAHelper()
            ema_helper.register(model)
            ema_helper.load_state_dict(checkpoint['ema_state_dict'])
            ema_helper.ema(model)
            print("✓ Using UNet EMA weights")
        else:
            print("✓ Using UNet regular weights")
    else:
        # Old format - just state dict
        model.load_state_dict(checkpoint)
        print("✓ Loaded UNet from old format checkpoint")
    
    model.eval()
    return model


def load_diffusion_model(model_path, config, device):
    """Load diffusion model from checkpoint"""
    model = DiffusionModel(config)
    
    # Load checkpoint
    states = torch.load(model_path, map_location=device)
    
    # Move to device and wrap with DataParallel (as expected by diffusion model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    
    # Load model state dict
    model.load_state_dict(states[0], strict=True)
    
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(model)
        print("✓ Using Diffusion EMA weights")
    else:
        print("✓ Using Diffusion regular weights")
    
    model.eval()
    
    # Create diffusion runner for sampling
    runner = DiffusionRunner(args=None, config=config, device=device)
    runner.model = model
    
    return runner


def run_unet_inference(model, x_ld):
    """Run UNet inference on low-dose input"""
    with torch.no_grad():
        output = model(x_ld)
    return output


def run_diffusion_inference(runner, x_ld, num_timesteps=None):
    """Run diffusion model inference on low-dose input"""
    if num_timesteps is None:
        num_timesteps = runner.config.diffusion.num_diffusion_timesteps
    
    with torch.no_grad():
        # Initialize noise
        x_T = torch.randn_like(x_ld)
        
        # Set up sampling parameters
        runner.args = type('args', (), {})()  # Create empty args object
        runner.args.sample_type = "generalized"
        runner.args.scheduler_type = "uniform"
        runner.args.timesteps = num_timesteps
        runner.args.eta = 0.0
        runner.args.skip = runner.num_timesteps // num_timesteps
        
        # Use sg_sample_image which expects 2 channels: [x_img, xt]
        # where x_img is the low-dose input (condition)
        sample = runner.sg_sample_image(x_T, x_ld, runner.model, last=True)
    
    return sample


def normalize_image(img):
    """Normalize image from [-1, 1] to [0, 1] for visualization and metrics"""
    return np.clip((img + 1) / 2, 0, 1)


def calculate_metrics(gt_img, pred_img):
    """Calculate PSNR and SSIM metrics"""
    # Ensure images are in [0, 1] range
    gt_normalized = normalize_image(gt_img)
    pred_normalized = normalize_image(pred_img)
    
    psnr = compare_psnr(gt_normalized, pred_normalized, data_range=1)
    ssim = compare_ssim(gt_normalized, pred_normalized, data_range=1)
    
    return psnr, ssim


# Initialize LPIPS metric
lpips_metric = LPIPS(net='alex').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def calculate_additional_metrics(gt_img, pred_img, device):
    """Calculate additional metrics for CT image comparison"""
    # Ensure images are in [0, 1] range
    gt_normalized = normalize_image(gt_img)
    pred_normalized = normalize_image(pred_img)

    # Convert to tensors for LPIPS and move to the same device
    gt_tensor = torch.tensor(gt_normalized).unsqueeze(0).unsqueeze(0).to(device)
    pred_tensor = torch.tensor(pred_normalized).unsqueeze(0).unsqueeze(0).to(device)

    # LPIPS
    lpips_score = lpips_metric(gt_tensor, pred_tensor).item()

    # MAE
    mae = np.mean(np.abs(gt_normalized - pred_normalized))

    # RMSE
    rmse = np.sqrt(compare_mse(gt_normalized, pred_normalized))

    # GMSD
    gt_edges = sobel(gt_normalized)
    pred_edges = sobel(pred_normalized)
    gmsd = np.std(gt_edges - pred_edges)

    return lpips_score, mae, rmse, gmsd


def plot_comparison(input_img, unet_output, diffusion_output, gt_img, save_path=None, idx=0):
    """Create comparison plot showing all models' outputs"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Normalize all images for display
    input_norm = normalize_image(input_img)
    unet_norm = normalize_image(unet_output)
    diffusion_norm = normalize_image(diffusion_output)
    gt_norm = normalize_image(gt_img)

    # Calculate metrics
    unet_psnr, unet_ssim = calculate_metrics(gt_img, unet_output)
    diffusion_psnr, diffusion_ssim = calculate_metrics(gt_img, diffusion_output)
    unet_lpips, unet_mae, unet_rmse, unet_gmsd = calculate_additional_metrics(gt_img, unet_output, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    diffusion_lpips, diffusion_mae, diffusion_rmse, diffusion_gmsd = calculate_additional_metrics(gt_img, diffusion_output, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Plot images
    axes[0].imshow(input_norm, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Low-Dose Input')
    axes[0].axis('off')

    axes[1].imshow(unet_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'UNet Output\nPSNR: {unet_psnr:.2f}, SSIM: {unet_ssim:.3f}\nLPIPS: {unet_lpips:.3f}, MAE: {unet_mae:.3f}\nRMSE: {unet_rmse:.3f}, GMSD: {unet_gmsd:.3f}')
    axes[1].axis('off')

    axes[2].imshow(diffusion_norm, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Diffusion Output\nPSNR: {diffusion_psnr:.2f}, SSIM: {diffusion_ssim:.3f}\nLPIPS: {diffusion_lpips:.3f}, MAE: {diffusion_mae:.3f}\nRMSE: {diffusion_rmse:.3f}, GMSD: {diffusion_gmsd:.3f}')
    axes[2].axis('off')

    axes[3].imshow(gt_norm, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Ground Truth')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot: {save_path}")

    plt.show()

    return unet_psnr, unet_ssim, diffusion_psnr, diffusion_ssim


def create_summary_plot(results_df, save_dir):
    """Create summary plots comparing model performance"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSNR comparison
    axes[0, 0].boxplot([results_df['UNet_PSNR'], results_df['Diffusion_PSNR']], 
                      labels=['UNet', 'Diffusion'])
    axes[0, 0].set_title('PSNR Comparison')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # SSIM comparison
    axes[0, 1].boxplot([results_df['UNet_SSIM'], results_df['Diffusion_SSIM']], 
                      labels=['UNet', 'Diffusion'])
    axes[0, 1].set_title('SSIM Comparison')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PSNR scatter plot
    axes[1, 0].scatter(results_df['UNet_PSNR'], results_df['Diffusion_PSNR'], alpha=0.6)
    max_psnr = max(results_df['UNet_PSNR'].max(), results_df['Diffusion_PSNR'].max())
    min_psnr = min(results_df['UNet_PSNR'].min(), results_df['Diffusion_PSNR'].min())
    axes[1, 0].plot([min_psnr, max_psnr], [min_psnr, max_psnr], 'r--', alpha=0.7)
    axes[1, 0].set_xlabel('UNet PSNR (dB)')
    axes[1, 0].set_ylabel('Diffusion PSNR (dB)')
    axes[1, 0].set_title('PSNR: UNet vs Diffusion')
    axes[1, 0].grid(True, alpha=0.3)
    
    # SSIM scatter plot
    axes[1, 1].scatter(results_df['UNet_SSIM'], results_df['Diffusion_SSIM'], alpha=0.6)
    max_ssim = max(results_df['UNet_SSIM'].max(), results_df['Diffusion_SSIM'].max())
    min_ssim = min(results_df['UNet_SSIM'].min(), results_df['Diffusion_SSIM'].min())
    axes[1, 1].plot([min_ssim, max_ssim], [min_ssim, max_ssim], 'r--', alpha=0.7)
    axes[1, 1].set_xlabel('UNet SSIM')
    axes[1, 1].set_ylabel('Diffusion SSIM')
    axes[1, 1].set_title('SSIM: UNet vs Diffusion')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    summary_path = os.path.join(save_dir, 'model_comparison_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary plot: {summary_path}")
    plt.show()


def print_statistics(results_df):
    """Print detailed statistics comparing both models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON STATISTICS")
    print("="*60)
    
    # Calculate statistics
    unet_psnr_mean = results_df['UNet_PSNR'].mean()
    unet_psnr_std = results_df['UNet_PSNR'].std()
    diffusion_psnr_mean = results_df['Diffusion_PSNR'].mean()
    diffusion_psnr_std = results_df['Diffusion_PSNR'].std()
    
    unet_ssim_mean = results_df['UNet_SSIM'].mean()
    unet_ssim_std = results_df['UNet_SSIM'].std()
    diffusion_ssim_mean = results_df['Diffusion_SSIM'].mean()
    diffusion_ssim_std = results_df['Diffusion_SSIM'].std()
    
    print(f"\nPSNR Results:")
    print(f"  UNet:      {unet_psnr_mean:.4f} ± {unet_psnr_std:.4f} dB")
    print(f"  Diffusion: {diffusion_psnr_mean:.4f} ± {diffusion_psnr_std:.4f} dB")
    print(f"  Difference: {diffusion_psnr_mean - unet_psnr_mean:.4f} dB")
    
    print(f"\nSSIM Results:")
    print(f"  UNet:      {unet_ssim_mean:.4f} ± {unet_ssim_std:.4f}")
    print(f"  Diffusion: {diffusion_ssim_mean:.4f} ± {diffusion_ssim_std:.4f}")
    print(f"  Difference: {diffusion_ssim_mean - unet_ssim_mean:.4f}")
    
    # Statistical significance tests (if scipy is available)
    try:
        from scipy.stats import ttest_rel
        psnr_tstat, psnr_pval = ttest_rel(results_df['Diffusion_PSNR'], results_df['UNet_PSNR'])
        ssim_tstat, ssim_pval = ttest_rel(results_df['Diffusion_SSIM'], results_df['UNet_SSIM'])
        
        print(f"\nStatistical Significance (paired t-test):")
        print(f"  PSNR: t={psnr_tstat:.4f}, p={psnr_pval:.6f}")
        print(f"  SSIM: t={ssim_tstat:.4f}, p={ssim_pval:.6f}")
        
        alpha = 0.05
        if psnr_pval < alpha:
            print(f"  PSNR difference is statistically significant (p < {alpha})")
        else:
            print(f"  PSNR difference is not statistically significant (p >= {alpha})")
            
        if ssim_pval < alpha:
            print(f"  SSIM difference is statistically significant (p < {alpha})")
        else:
            print(f"  SSIM difference is not statistically significant (p >= {alpha})")
            
    except ImportError:
        print("\nNote: Install scipy for statistical significance testing")
    
    # Best/worst cases
    best_unet_idx = results_df['UNet_PSNR'].idxmax()
    best_diffusion_idx = results_df['Diffusion_PSNR'].idxmax()
    worst_unet_idx = results_df['UNet_PSNR'].idxmin()
    worst_diffusion_idx = results_df['Diffusion_PSNR'].idxmin()
    
    print(f"\nBest/Worst Cases:")
    print(f"  Best UNet PSNR:      {results_df.loc[best_unet_idx, 'UNet_PSNR']:.4f} dB (Image {best_unet_idx})")
    print(f"  Best Diffusion PSNR: {results_df.loc[best_diffusion_idx, 'Diffusion_PSNR']:.4f} dB (Image {best_diffusion_idx})")
    print(f"  Worst UNet PSNR:     {results_df.loc[worst_unet_idx, 'UNet_PSNR']:.4f} dB (Image {worst_unet_idx})")
    print(f"  Worst Diffusion PSNR:{results_df.loc[worst_diffusion_idx, 'Diffusion_PSNR']:.4f} dB (Image {worst_diffusion_idx})")


def main():
    parser = argparse.ArgumentParser(description='Compare UNet and Diffusion Models')
    parser.add_argument('--unet_path', type=str, required=True,
                       help='Path to UNet checkpoint')
    parser.add_argument('--diffusion_path', type=str, required=True,
                       help='Path to Diffusion checkpoint')
    parser.add_argument('--config_path', type=str, default='configs/lodopab_linear.yml',
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to compare')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for comparison')
    parser.add_argument('--save_dir', type=str, default='model_comparison_results',
                       help='Directory to save comparison results')
    parser.add_argument('--save_images', action='store_true',
                       help='Save individual comparison images')
    parser.add_argument('--use_unet_ema', action='store_true',
                       help='Use EMA weights for UNet')
    parser.add_argument('--diffusion_steps', type=int, default=None,
                       help='Number of diffusion steps (default: use config value)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    image_size = config.data.image_size
    dataset = LODOPAB(dataroot=config.data.sample_dataroot, img_size=image_size, split='calculate')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Loaded dataset with {len(dataset)} images")
    print(f"Image size: {image_size}x{image_size}")
    
    # Load models
    print("\nLoading models...")
    unet_model = load_unet_model(args.unet_path, config, device, use_ema=args.use_unet_ema)
    diffusion_runner = load_diffusion_model(args.diffusion_path, config, device)
    
    print(f"\nStarting comparison:")
    print(f"  UNet checkpoint: {args.unet_path}")
    print(f"  Diffusion checkpoint: {args.diffusion_path}")
    print(f"  Samples: {args.num_samples} (starting from index {args.start_idx})")
    print(f"  Diffusion steps: {args.diffusion_steps or 'default'}")
    
    # Store results
    results = []
    
    # Run comparison
    processed_count = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Comparing models")):
            # Skip images before start_idx
            if idx < args.start_idx:
                continue
            
            # Get batch data
            x_ld = batch['LD'].to(device)  # Low-dose input
            x_fd = batch['FD'].to(device)  # Full-dose ground truth
            
            # Run UNet inference
            unet_output = run_unet_inference(unet_model, x_ld)
            
            # Run Diffusion inference
            diffusion_output = run_diffusion_inference(diffusion_runner, x_ld, args.diffusion_steps)
            
            # Convert to numpy for analysis
            input_np = x_ld.squeeze().cpu().numpy()
            unet_np = unet_output.squeeze().cpu().numpy()
            diffusion_np = diffusion_output.squeeze().cpu().numpy()
            gt_np = x_fd.squeeze().cpu().numpy()
            
            # Calculate metrics
            unet_psnr, unet_ssim = calculate_metrics(gt_np, unet_np)
            diffusion_psnr, diffusion_ssim = calculate_metrics(gt_np, diffusion_np)

            # Calculate additional metrics
            unet_lpips, unet_mae, unet_rmse, unet_gmsd = calculate_additional_metrics(gt_np, unet_np, device)
            diffusion_lpips, diffusion_mae, diffusion_rmse, diffusion_gmsd = calculate_additional_metrics(gt_np, diffusion_np, device)

            # Store results
            result = {
                'Image_Index': idx,
                'UNet_PSNR': unet_psnr,
                'UNet_SSIM': unet_ssim,
                'Diffusion_PSNR': diffusion_psnr,
                'Diffusion_SSIM': diffusion_ssim,
                'UNet_LPIPS': unet_lpips,
                'UNet_MAE': unet_mae,
                'UNet_RMSE': unet_rmse,
                'UNet_GMSD': unet_gmsd,
                'Diffusion_LPIPS': diffusion_lpips,
                'Diffusion_MAE': diffusion_mae,
                'Diffusion_RMSE': diffusion_rmse,
                'Diffusion_GMSD': diffusion_gmsd,
                'PSNR_Diff': diffusion_psnr - unet_psnr,
                'SSIM_Diff': diffusion_ssim - unet_ssim
            }
            results.append(result)
            
            # Create comparison plot
            save_path = None
            if args.save_images:
                save_path = os.path.join(args.save_dir, f'comparison_{idx:04d}.png')
            
            plot_comparison(input_np, unet_np, diffusion_np, gt_np, save_path, idx)
            
            processed_count += 1
            if processed_count >= args.num_samples:
                break
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = os.path.join(args.save_dir, 'comparison_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to: {csv_path}")
    
    # Print statistics
    print_statistics(results_df)
    
    # Create summary plots
    create_summary_plot(results_df, args.save_dir)
    
    print(f"\nComparison complete! Results saved in: {args.save_dir}")


if __name__ == '__main__':
    main()
