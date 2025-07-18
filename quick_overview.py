import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dival.datasets.lodopab_dataset import LoDoPaBDataset
from dival.datasets.fbp_dataset import FBPDataset
import time

lodopab = LoDoPaBDataset(impl='astra_cpu')
fbp_lodopab = FBPDataset(lodopab, lodopab.ray_trafo)

train_dataset = lodopab.create_torch_dataset(part='train')
train_dataset_fbp = fbp_lodopab.create_torch_dataset(part='train')

batch_size = 4
num_samples = len(train_dataset)

plt.ion()

fig, axes = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
plt.suptitle('LDCT (top) / NDCT (bottom) pairs')

def show_pairs(indices):
    x = []
    y = []
    x_fbp = []
    
    t0 = time.time()
    for idx in indices:
        x_t, y_t = train_dataset[idx]
        x.append(x_t)
        y.append(y_t)
    t1 = time.time()
    for idx in indices:
        x_f, _ = train_dataset_fbp[idx]
        x_fbp.append(x_f)
    t2 = time.time()
    
    print(f"Loaded ground truth images in {t1 - t0:.4f} seconds.")
    print(f"Loaded FBP images in {t2 - t1:.4f} seconds.")

    x = torch.stack(x)
    y = torch.stack(y)
    x_fbp = torch.stack(x_fbp)

    for i in range(batch_size):
        axes[0, i].imshow(x_fbp[i], cmap='gray')
        axes[0, i].set_title(f'LDCT #{indices[i]}')
        axes[0, i].axis('off')
        axes[1, i].imshow(y[i], cmap='gray')
        axes[1, i].set_title(f'NDCT #{indices[i]}')
        axes[1, i].axis('off')
    plt.draw()

def on_key(event):
    if event.key == 'enter':
        sample_indices = np.random.choice(num_samples, batch_size, replace=False)
        show_pairs(sample_indices)

fig.canvas.mpl_connect('key_press_event', on_key)
# Show initial batch
show_pairs(np.random.choice(num_samples, batch_size, replace=False))
plt.show(block=True)