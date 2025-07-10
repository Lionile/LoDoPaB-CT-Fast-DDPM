import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from .sr_util import transform_augment
import os

class LODOPAB(Dataset):
    def __init__(self, dataroot, img_size, split='train', data_len=-1, cache_dir=r'C:\datasets\LoDoPaB-CT-cache'):
        self.img_size = img_size
        self.data_len = data_len
        self.split = split
        self.cache_dir = cache_dir

        # use the correct subfolder for the split and self.img_size * self.img_size images
        split_folder = {
            'train': f'train_{self.img_size}x{self.img_size}',
            'validation': f'validation_{self.img_size}x{self.img_size}',
            'test': f'test_{self.img_size}x{self.img_size}',
            'calculate': f'test_{self.img_size}x{self.img_size}',
        }[split]
        self.split_dir = os.path.join(self.cache_dir, split_folder)

        ldct_files = sorted([f for f in os.listdir(self.split_dir) if f.startswith('ldct_') and f.endswith('.npy')])
        ndct_files = sorted([f for f in os.listdir(self.split_dir) if f.startswith('ndct_') and f.endswith('.npy')])
        self.ldct_files = ldct_files
        self.ndct_files = ndct_files
        self.dataset_len = len(self.ldct_files)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        ldct_path = os.path.join(self.split_dir, self.ldct_files[index])
        ndct_path = os.path.join(self.split_dir, self.ndct_files[index])
        ldct_img = np.load(ldct_path)
        ndct_img = np.load(ndct_path)
        # convert to PIL Images
        ldct_pil = Image.fromarray((ldct_img * 255).astype(np.uint8))
        ndct_pil = Image.fromarray((ndct_img * 255).astype(np.uint8))
        [ldct_tensor, ndct_tensor] = transform_augment(
            [ldct_pil, ndct_pil], split=self.split, min_max=(-1, 1))
        case_name = f"case_{index:06d}"
        return {'FD': ndct_tensor, 'LD': ldct_tensor, 'case_name': case_name}