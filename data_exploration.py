import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dival.datasets.lodopab_dataset import LoDoPaBDataset
from dival.datasets.fbp_dataset import FBPDataset

lodopab = LoDoPaBDataset(impl='astra_cuda')
fbp_lodopab = FBPDataset(lodopab, lodopab.ray_trafo)

# create dataset
train_dataset = lodopab.create_torch_dataset(part='train')
train_dataset_fbp = fbp_lodopab.create_torch_dataset(part='train')
print(f'\ntype: {type(train_dataset)}')
print(f'length: {len(train_dataset)}')
print(f'dataset type: {type(train_dataset).__mro__}\n')


# create dataloader
batch_size = 4
# TODO: Add multiple workers... follow note in create_torch_dataset function 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
fbp_loader = DataLoader(train_dataset_fbp, batch_size=batch_size, shuffle=False)

# get any batch from dataset
# x, y = next(iter(train_loader))
# x_fbp, y_fbp = next(iter(fbp_loader))

# get 4 of the same images from each dataset
sample_indices = np.random.choice(10000, 4, replace=False)
sample_indices = [0, 1, 2, 3]
x = []
y = []
x_fbp = []
y_fbp = []

for idx in sample_indices:
    x_t, y_t = train_dataset[idx]
    x.append(x_t)
    y.append(y_t)
    
    x_f, y_f = train_dataset_fbp[idx]
    x_fbp.append(x_f)
    y_fbp.append(y_f)

# Convert lists to tensors
x = torch.stack(x)
y = torch.stack(y)
x_fbp = torch.stack(x_fbp)
y_fbp = torch.stack(y_fbp)



print(f"Input shape: {x.shape}, Target shape: {y.shape}")



fig, ax = plt.subplots(1, 4, figsize=(15, 6))
plt.suptitle('input')
for i in range(4):
    ax[i].imshow(x[i], cmap='gray')


fig, ax = plt.subplots(2, 4, figsize=(15, 6))
plt.suptitle('output')
for i in range(4):
    ax[0, i].imshow(y[i], cmap='gray')
for i in range(4):
    ax[1, i].imshow(x_fbp[i], cmap='gray')


fig, ax = plt.subplots(1, 4, figsize=(15, 6))
plt.suptitle('difference')
for i in range(4):
    ax[i].imshow(abs(y[i] - x_fbp[i]), cmap='gray')

plt.show()