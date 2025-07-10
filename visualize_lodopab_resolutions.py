import matplotlib.pyplot as plt
from datasets.LODOPAB import LODOPAB
import torch

IMG_SIZES = [256, 128, 64, 32]
CACHE_DIR = r'C:\datasets\LoDoPaB-CT-cache'

# load all datasets for each resolution once
datasets = [LODOPAB(dataroot='', img_size=s, split='test', cache_dir=CACHE_DIR) for s in IMG_SIZES]
num_examples = min(len(ds) for ds in datasets)

plt.ion()

# prepare a 2-row plot: row 0 = ground truth, row 1 = low dose images
fig, axes = plt.subplots(2, len(IMG_SIZES), figsize=(16, 8))
fd_imgs = [None] * len(IMG_SIZES)
ld_imgs = [None] * len(IMG_SIZES)
fd_handles = []
ld_handles = []

def show_example(idx):
    for i, ds in enumerate(datasets):
        sample = ds[idx]
        fd_img = sample['FD']
        ld_img = sample['LD']
        fd_img = (fd_img + 1) / 2
        ld_img = (ld_img + 1) / 2
        fd_imgs[i] = fd_img.squeeze().cpu().numpy()
        ld_imgs[i] = ld_img.squeeze().cpu().numpy()
        # first row - ground truth
        if len(fd_handles) < len(IMG_SIZES):
            fd_handle = axes[0, i].imshow(fd_imgs[i], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'{IMG_SIZES[i]}x{IMG_SIZES[i]}')
            axes[0, i].axis('off')
            fd_handles.append(fd_handle)
        else:
            fd_handles[i].set_data(fd_imgs[i])
        # second row - input
        if len(ld_handles) < len(IMG_SIZES):
            ld_handle = axes[1, i].imshow(ld_imgs[i], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title("")
            axes[1, i].axis('off')
            ld_handles.append(ld_handle)
        else:
            ld_handles[i].set_data(ld_imgs[i])
    axes[0, 0].set_ylabel('Ground Truth', fontsize=14)
    axes[1, 0].set_ylabel('Input', fontsize=14)
    fig.suptitle(f'Example {idx}')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()
    fig.canvas.flush_events()

example_idx = 0
show_example(example_idx)

def on_key(event):
    global example_idx
    if event.key == 'enter' or event.key == 'return':
        example_idx += 1
        if example_idx < num_examples:
            show_example(example_idx)
        else:
            plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show(block=True)
