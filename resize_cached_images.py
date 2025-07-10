import os
import numpy as np
from PIL import Image

# ==== USER SETTINGS ====
IN_BASE = r"C:\datasets\LoDoPaB-CT-cache"  # Base directory with cached .npy files
OUT_BASE = r"C:\datasets\LoDoPaB-CT-cache" # Where to save resized .npy files
SPLITS = ["train", "validation", "test"]
FILE_TYPES = ["fbp", "gt"]
FILE_PREFIX = {"fbp": "ldct", "gt": "ndct"}
TARGET_RES = 256  # Desired output resolution
# =======================

def resize_and_save_individual(in_file, out_dir, prefix, target_res):
    arr = np.load(in_file)
    n_imgs = arr.shape[0]
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_imgs):
        img = arr[i]
        img_min, img_max = img.min(), img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img = pil_img.resize((target_res, target_res), Image.LANCZOS)
        img_resized = np.array(pil_img).astype(np.float32) / 255.0
        img_resized = img_resized * (img_max - img_min) + img_min
        fname = f"{prefix}_{i:05d}.npy"
        np.save(os.path.join(out_dir, fname), img_resized)
    print(f"Saved {n_imgs} images as {prefix}_INDEX.npy in {out_dir}")

if __name__ == "__main__":
    for split in SPLITS:
        out_split = os.path.join(OUT_BASE, f"{split}_{TARGET_RES}x{TARGET_RES}")
        os.makedirs(out_split, exist_ok=True)
        for ftype in FILE_TYPES:
            in_file = os.path.join(IN_BASE, f"{split}_{ftype}.npy")
            prefix = FILE_PREFIX[ftype]
            if os.path.exists(in_file):
                resize_and_save_individual(in_file, out_split, prefix, TARGET_RES)
            else:
                print(f"File not found: {in_file}")