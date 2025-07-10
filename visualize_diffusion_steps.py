import os
import torch
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from models.diffusion import Model
from models.ema import EMAHelper
from datasets.LODOPAB import LODOPAB
from functions.denoising import sg_generalized_steps, sg_ddpm_steps
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def visualize_diffusion_process(
    ckpt_path, 
    config_path, 
    save_dir='diffusion_visualization', 
    num_samples=4, 
    timesteps=100, 
    eta=0.0,
    sample_type='generalized',
    scheduler_type='uniform',
    save_intermediate=True,
    img_size=64,
    start_idx=0,
    shuffle=False
):
    """
    Visualize the diffusion denoising process using a trained model
    
    Args:
        ckpt_path: Path to the checkpoint file
        config_path: Path to the config file
        save_dir: Directory to save the visualization results
        num_samples: Number of samples to process
        timesteps: Number of timesteps for sampling
        eta: Parameter controlling noise level in the generalized sampling
        sample_type: 'generalized' or 'ddpm_noisy'
        scheduler_type: 'uniform' or 'non-uniform'
        save_intermediate: Whether to save intermediate steps as images
        img_size: Image size for the dataset
    """
    # create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # load config
    with open(os.path.join("configs", config_path), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    config.device = device
    
    # load model
    print(f"Loading model from {ckpt_path}")
    model = Model(config)
    states = torch.load(ckpt_path, map_location=device)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(states[0], strict=True)
    
    # apply EMA if available
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(model)
    
    model.eval()
    
    # dataset and dataloader
    dataset = LODOPAB(dataroot='', img_size=img_size, split='calculate')
    # only use a subset of the dataset starting from start_idx
    if start_idx > 0 and start_idx < len(dataset):
        indices = list(range(start_idx, min(start_idx + num_samples, len(dataset))))
        print(f"Using dataset samples from index {start_idx} to {start_idx + num_samples - 1}")
    else: # not in valid range
        indices = list(range(min(len(dataset), num_samples)))
        print(f"Using dataset samples from index 0 to {num_samples - 1}")
    
    if shuffle:
        import random
        random.shuffle(indices)
        print("Samples have been shuffled")
    
    
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    
    # compute betas and alphas for diffusion
    num_timesteps = 1000
    if config.diffusion.beta_schedule == "linear":
        betas = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, num_timesteps, device=device)
    elif config.diffusion.beta_schedule == "quad":
        betas = torch.linspace(config.diffusion.beta_start ** 0.5, config.diffusion.beta_end ** 0.5, num_timesteps, device=device) ** 2
    else:
        raise NotImplementedError(f"Unknown beta schedule: {config.diffusion.beta_schedule}")
    
    # Setup timestep sequence based on scheduler type
    if scheduler_type == 'uniform':
        skip = num_timesteps // timesteps
        seq = range(0, num_timesteps, skip)
        seq = list(seq)
        if seq[0] != 0:
            seq[0] = 0
    elif scheduler_type == 'non-uniform':
        # uses a non-uniform sequence that focuses more on the last 30% of the diffusion process
        if timesteps == 10:
            seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]
        else:
            num_1 = int(timesteps * 0.4)  # first 40% of steps for first 70% of diffusion process
            num_2 = timesteps - num_1  # last 60% of steps for final 30% of diffusion process
            stage_1 = np.linspace(0, 699, num_1+1)[:-1]  # 0-699 (70% of the 1000 steps)
            stage_2 = np.linspace(699, 999, num_2)       # 700-999 (30% of the 1000 steps)
            stage_1 = np.ceil(stage_1).astype(int)
            stage_2 = np.ceil(stage_2).astype(int)
            seq = np.concatenate((stage_1, stage_2))
            print(f"Using non-uniform scheduler with {len(seq)} total steps")
            print(f"  - {len(stage_1)} steps for the first 70% of diffusion (t=0 to t=699)")
            print(f"  - {len(stage_2)} steps for the last 30% of diffusion (t=700 to t=999)")
    else:
        raise ValueError("Scheduler type must be 'uniform' or 'non-uniform'")
    
    # reverse the sequence for visualization (from noisy to clean)
    vis_seq = list(reversed(seq))
    
    # process samples
    results = []
    psnr_list = []
    ssim_list = []
    
    print(f"Processing {min(num_samples, len(subset))} samples (out of {len(dataset)} available in the full dataset)")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # use the actual dataset index if available, otherwise use batch index
            original_idx = indices[batch_idx] if batch_idx < len(indices) else batch_idx
            
            x_ld = batch['LD'].to(device) # low dose input
            x_fd = batch['FD'].to(device) # normal dose ground truth
            
            # create random noise for initial image
            x = torch.randn(
                1,
                config.data.channels,
                img_size,
                img_size,
                device=device,
            )
            
            # sample from the model
            if sample_type == "generalized":
                xs, x0_preds = sg_generalized_steps(x, x_ld, seq, model, betas, eta=eta)
            elif sample_type == "ddpm_noisy":
                xs, x0_preds = sg_ddpm_steps(x, x_ld, seq, model, betas)
            else:
                raise ValueError("Sample type must be 'generalized' or 'ddpm_noisy'")
            
            # final output image
            x_output = xs[-1].to(device)
            
            # metrics (data is already in range [-1, 1])
            output_np = x_output.squeeze().cpu().numpy()
            gt_np = x_fd.squeeze().cpu().numpy()
            input_np = x_ld.squeeze().cpu().numpy()
            
            # scale to [0, 1] for visualization
            output_np = (output_np + 1) / 2
            gt_np = (gt_np + 1) / 2
            input_np = (input_np + 1) / 2
            
            # calculate PSNR and SSIM
            psnr = compare_psnr(gt_np, output_np, data_range=1.0)
            ssim = compare_ssim(gt_np, output_np, data_range=1.0)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            print(f"Sample {batch_idx} (dataset idx {original_idx}): PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
            
            # visualize diffusion process
            if save_intermediate:
                # select a subset of 10 steps to visualize
                num_vis_steps = 10
                if len(vis_seq) > num_vis_steps:
                    vis_indices = np.linspace(0, len(vis_seq)-1, num_vis_steps, dtype=int)
                    vis_indices = np.unique(vis_indices)
                    # if any indices are lost due to duplicates, add more at even intervals
                    if len(vis_indices) < num_vis_steps:
                        # find largest gaps and fill them
                        missing_count = num_vis_steps - len(vis_indices)
                        all_indices = set(range(len(vis_seq)))
                        unused_indices = list(all_indices - set(vis_indices))
                        # choose the indices that give the most even spacing
                        if len(unused_indices) >= missing_count:
                            # select evenly spaced indices from the unused ones
                            additional_indices = np.linspace(0, len(unused_indices)-1, missing_count, dtype=int)
                            additional_indices = [unused_indices[i] for i in additional_indices]
                            vis_indices = np.sort(np.concatenate((vis_indices, additional_indices)))
                    
                    # always include first and last step
                    if 0 not in vis_indices:
                        vis_indices = np.sort(np.concatenate(([0], vis_indices[1:])))
                    if len(vis_seq) - 1 not in vis_indices:
                        vis_indices = np.sort(np.concatenate((vis_indices[:-1], [len(vis_seq) - 1])))
                    
                    # ensure there are exactly num_vis_steps indices
                    if len(vis_indices) > num_vis_steps:
                        # keep first and last, evenly sample the rest
                        middle_indices = vis_indices[1:-1]
                        selected_middle = np.linspace(0, len(middle_indices)-1, num_vis_steps-2, dtype=int)
                        selected_middle = [middle_indices[i] for i in selected_middle]
                        vis_indices = np.sort(np.concatenate(([0], selected_middle, [len(vis_seq) - 1])))
                else:
                    vis_indices = np.arange(len(vis_seq))
                    
                # true noise steps (given 1000 steps)
                vis_steps = [vis_seq[i] for i in vis_indices]
                
                # prepare images
                step_images = []
                step_numbers = []
                # use vis_indices to get the correct steps in order
                for idx in vis_indices:
                    step = vis_seq[idx]
                    step_idx = idx  # this is the index in xs
                    x_step = xs[step_idx].to(device)
                    # scale to [0, 1]
                    x_step_np = (x_step.squeeze().cpu().numpy() + 1) / 2
                    step_images.append(x_step_np)
                    step_numbers.append(step)
                
                # plot diffusion steps
                plt.figure(figsize=(20, 4))
                for i, img in enumerate(step_images):
                    plt.subplot(1, len(step_images), i + 1)
                    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
                    plt.axis('off')
                    
                    step_number = i + 1
                    progress = int((step_number / len(step_images)) * 100)
                    # show the actual step number out of total steps
                    actual_step = vis_indices[i] + 1
                    total_steps = len(vis_seq)
                    plt.title(f"Step {actual_step}/{total_steps}\n({progress}% denoised)")
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"diffusion_steps_sample{original_idx}.png"))
                plt.close()
            
            # plot final results
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(input_np, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title("Low Dose Input")
            
            plt.subplot(1, 3, 2)
            plt.imshow(output_np, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title(f"Diffusion Output\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
            
            plt.subplot(1, 3, 3)
            plt.imshow(gt_np, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title("Ground Truth")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"comparison_sample{original_idx}.png"))
            plt.close()
            
            # save images
            save_image(torch.tensor(output_np).unsqueeze(0), os.path.join(save_dir, f"sample{original_idx}_output.png"))
            save_image(torch.tensor(input_np).unsqueeze(0), os.path.join(save_dir, f"sample{original_idx}_input.png"))
            save_image(torch.tensor(gt_np).unsqueeze(0), os.path.join(save_dir, f"sample{original_idx}_gt.png"))
    
    # print average metrics
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")
    
    # save metrics to file
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write("\nPer-sample metrics:\n")
        for i, (psnr, ssim) in enumerate(zip(psnr_list, ssim_list)):
            f.write(f"Sample {i}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize diffusion process for a trained model')
    parser.add_argument('--ckpt_path', type=str, default='experiments/logs/lodopab_fast_ddpm/ckpt_10000.pth', 
                        help='Path to the checkpoint file')
    parser.add_argument('--config', type=str, default='lodopab_linear.yml', 
                        help='Path to the config file')
    parser.add_argument('--save_dir', type=str, default='diffusion_visualization', 
                        help='Directory to save the visualization results')
    parser.add_argument('--num_samples', type=int, default=4, 
                        help='Number of samples to process')
    parser.add_argument('--timesteps', type=int, default=100, 
                        help='Number of timesteps for sampling')
    parser.add_argument('--eta', type=float, default=0.0, 
                        help='Parameter controlling noise level in generalized sampling')
    parser.add_argument('--sample_type', type=str, default='generalized', 
                        choices=['generalized', 'ddpm_noisy'], 
                        help='Type of sampling to use')
    parser.add_argument('--scheduler_type', type=str, default='uniform', 
                        choices=['uniform', 'non-uniform'], 
                        help='Type of scheduler to use for timesteps.')
    parser.add_argument('--save_intermediate', action='store_true', 
                        help='Whether to save intermediate diffusion steps as images')
    parser.add_argument('--img_size', type=int, default=64, 
                        help='Image size for the dataset')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index in the dataset')
    parser.add_argument('--shuffle', action='store_true',
                        help='Whether to shuffle the dataset samples')
    
    args = parser.parse_args()
    
    visualize_diffusion_process(
        ckpt_path=args.ckpt_path,
        config_path=args.config,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        timesteps=args.timesteps,
        eta=args.eta,
        sample_type=args.sample_type,
        scheduler_type=args.scheduler_type,
        save_intermediate=args.save_intermediate,
        img_size=args.img_size,
        start_idx=args.start_idx,
        shuffle=args.shuffle
    )
