import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from datasets.LODOPAB import LODOPAB
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml


class SimpleConfig:
    """Simple config class to hold configuration values"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)


class EMAHelper:
    """Exponential Moving Average helper for model parameters"""
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def nonlinearity(x):
    # swish activation function (same as diffusion model)
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    # GroupNorm same as diffusion model
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = 1  # Override to 1 for regular UNet (no noise channel)
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == self.resolution

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1))
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleConfig(config_dict)

def main():
    # Load config from lodopab_linear.yml
    config = load_config('configs/lodopab_linear.yml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use config values
    batch_size = config.training.batch_size  # 4
    epochs = config.training.n_epochs  # 25
    lr = config.optim.lr  # 0.00002
    image_size = config.data.image_size  # 128
    ch = config.model.ch  # 128
    ch_mult = config.model.ch_mult  # [1, 2, 2, 4]
    num_workers = config.data.num_workers  # 0
    use_ema = getattr(config.model, 'ema', False)  # True from config
    ema_rate = getattr(config.model, 'ema_rate', 0.999)  # 0.999 from config
    
    save_dir = 'unet_baseline_results'
    os.makedirs(save_dir, exist_ok=True)

    # Note: Using in_ch=1 instead of config in_channels=2 since regular UNet doesn't need noise channel
    dataset = LODOPAB(dataroot=config.data.train_dataroot, img_size=image_size, split='train')
    
    # For testing: use only a subset of the dataset
    # dataset = Subset(dataset, range(min(100, len(dataset))))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = UNet(config).to(device)
    
    # Initialize EMA if enabled in config
    ema_helper = None
    if use_ema:
        ema_helper = EMAHelper(mu=ema_rate)
        ema_helper.register(model)
    
    # Use config optimizer settings
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            betas=(config.optim.beta1, 0.999), 
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    criterion = nn.MSELoss()

    print(f"Training UNet with config settings:")
    print(f"  Image size: {image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Channels: {ch}")
    print(f"  Channel multipliers: {ch_mult}")
    print(f"  EMA enabled: {use_ema}")
    if use_ema:
        print(f"  EMA rate: {ema_rate}")

    batch_count = 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        for batch in pbar:
            x_ld = batch['LD'].to(device)
            x_fd = batch['FD'].to(device)
            optimizer.zero_grad()
            output = model(x_ld)
            loss = criterion(output, x_fd)
            loss.backward()
            optimizer.step()
            
            # Update EMA if enabled
            if ema_helper is not None:
                ema_helper.update(model)
            
            batch_count += 1
            epoch_loss += loss.item() * x_ld.size(0)
            pbar.set_postfix({'loss': loss.item(), 'batch': batch_count})
            
            # Save checkpoint every snapshot_freq batches
            if batch_count % config.training.snapshot_freq == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'batch': batch_count,
                    'loss': loss.item()
                }
                if ema_helper is not None:
                    checkpoint['ema_state_dict'] = ema_helper.state_dict()
                
                torch.save(checkpoint, os.path.join(save_dir, f'unet_batch{batch_count:06d}.pth'))
                # print(f"\nCheckpoint saved at batch {batch_count}")
        
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

def plot_images(imgs, titles=None, nrow=1, save_path=None):
    n = len(imgs)
    plt.figure(figsize=(4*n, 4))
    for i, img in enumerate(imgs):
        plt.subplot(nrow, n, i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# --- Inference function ---
def inference(model_path, config_path='configs/lodopab_linear.yml', save_dir='unet_baseline_results', num_samples=4, save_images=False, use_ema=False):
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
    
    # Load config
    config = load_config(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = config.data.image_size
    ch = config.model.ch
    ch_mult = config.model.ch_mult
    
    dataset = LODOPAB(dataroot=config.data.sample_dataroot, img_size=image_size, split='calculate')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model = UNet(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load EMA weights if available and requested
        if use_ema and 'ema_state_dict' in checkpoint:
            ema_helper = EMAHelper()
            ema_helper.register(model)
            ema_helper.load_state_dict(checkpoint['ema_state_dict'])
            ema_helper.ema(model)
            print("Using EMA weights for inference")
    else:
        # Old format - just state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Running inference with:")
    print(f"  Image size: {image_size}")
    print(f"  Model channels: {ch}")
    print(f"  Channel multipliers: {ch_mult}")
    
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x_ld = batch['LD'].to(device)
            x_fd = batch['FD'].to(device)
            output = model(x_ld)
            # Prepare images for plotting
            out_np = output.squeeze().cpu().numpy()
            gt_np = x_fd.squeeze().cpu().numpy()
            in_np = x_ld.squeeze().cpu().numpy()
            out_np = np.clip((out_np+1)/2, 0, 1)
            gt_np = np.clip((gt_np+1)/2, 0, 1)
            in_np = np.clip((in_np+1)/2, 0, 1)
            # Plot images
            plot_images([in_np, out_np, gt_np], titles=['Input', 'Output', 'Ground Truth'], nrow=1,
                        save_path=os.path.join(save_dir, f'infer_{idx:04d}_plot.png') if save_images else None)
            # Optionally save raw images
            if save_images:
                save_image(torch.tensor(out_np).unsqueeze(0), os.path.join(save_dir, f'infer_{idx:04d}_out.png'))
                save_image(torch.tensor(in_np).unsqueeze(0), os.path.join(save_dir, f'infer_{idx:04d}_in.png'))
                save_image(torch.tensor(gt_np).unsqueeze(0), os.path.join(save_dir, f'infer_{idx:04d}_gt.png'))
            # Compute PSNR/SSIM
            psnr = compare_psnr(gt_np, out_np, data_range=1)
            ssim = compare_ssim(gt_np, out_np, data_range=1)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            if idx+1 >= num_samples:
                break
    print(f'Average PSNR: {np.mean(psnr_list):.4f}, Average SSIM: {np.mean(ssim_list):.4f}')

if __name__ == '__main__':
    # run training
    main()
    # run inference
    # inference('unet_baseline_results/unet_epoch025.pth', num_samples=4, save_images=False)
