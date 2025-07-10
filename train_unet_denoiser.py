import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets.LODOPAB import LODOPAB
from tqdm import tqdm
import matplotlib.pyplot as plt


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ch=128, ch_mult=[1,2,2,2]):
        super().__init__()
        # encoder
        self.enc1 = UNetBlock(in_ch, ch*ch_mult[0])
        self.enc2 = UNetBlock(ch*ch_mult[0], ch*ch_mult[1])
        self.enc3 = UNetBlock(ch*ch_mult[1], ch*ch_mult[2])
        self.enc4 = UNetBlock(ch*ch_mult[2], ch*ch_mult[3])
        self.pool = nn.MaxPool2d(2)
        # decoder
        self.up3 = nn.ConvTranspose2d(ch*ch_mult[3], ch*ch_mult[2], 2, stride=2)
        self.dec3 = UNetBlock(ch*ch_mult[2]*2, ch*ch_mult[2])
        self.up2 = nn.ConvTranspose2d(ch*ch_mult[2], ch*ch_mult[1], 2, stride=2)
        self.dec2 = UNetBlock(ch*ch_mult[1]*2, ch*ch_mult[1])
        self.up1 = nn.ConvTranspose2d(ch*ch_mult[1], ch*ch_mult[0], 2, stride=2)
        self.dec1 = UNetBlock(ch*ch_mult[0]*2, ch*ch_mult[0])
        self.final = nn.Conv2d(ch*ch_mult[0], out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    epochs = 25
    lr = 2e-5
    save_dir = 'unet_baseline_results'
    os.makedirs(save_dir, exist_ok=True)

    dataset = LODOPAB(dataroot='', img_size=64, split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = UNet(in_ch=1, out_ch=1, ch=128, ch_mult=[1,2,2,4]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()

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
            epoch_loss += loss.item() * x_ld.size(0)
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")
        
        torch.save(model.state_dict(), os.path.join(save_dir, f'unet_epoch{epoch+1:03d}.pth'))

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
def inference(model_path, save_dir='unet_baseline_results', num_samples=4, save_images=False):
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LODOPAB(dataroot='', img_size=64, split='calculate')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = UNet(in_ch=1, out_ch=1, ch=128, ch_mult=[1,2,2,2]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
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

# Example usage:
# inference('unet_baseline_results/unet_epoch025.pth', save_images=True)

if __name__ == '__main__':
    # main()
    inference('unet_baseline_results/unet_epoch001.pth', num_samples=4, save_images=False)
