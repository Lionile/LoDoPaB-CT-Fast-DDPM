# Fast-DDPM

Adapted code from: [Fast-DDPM](https://github.com/mirthAI/Fast-DDPM)

Which is th official PyTorch implementation of:

[Fast-DDPM: Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation](https://ieeexplore.ieee.org/abstract/document/10979336) (JBHI 2025)

The code is adapted to work with the LoDoPab-CT dataset for imate reconstruction.

## LODOPAB-CT commands

- **`--timesteps x`** &mdash; sets the number of diffusion steps for training and inference (must match for both)

* Run training:

```
python fast_ddpm_main.py --config lodopab_linear.yml --dataset LODOPAB --exp experiments --doc lodopab_fast_ddpm --scheduler_type uniform --timesteps 20
```

* Continue training:

```
python fast_ddpm_main.py --config lodopab_linear.yml --dataset LODOPAB --exp experiments --doc lodopab_fast_ddpm --scheduler_type uniform --timesteps 20 --resume_training
```

* Run inference:

```
python fast_ddpm_main.py --config lodopab_linear.yml --dataset LODOPAB --exp experiments --doc lodopab_fast_ddpm --scheduler_type uniform --timesteps 20 --sample --fid
```

* Compare models
```

```

* Generating unconditional samples
```
python generate_unconditional_samples.py --diffusion_path experiments/logs/lodopab_fast_ddpm/ckpt_63700.pth --config_path configs/lodopab_linear.yml --image_size 128 --save_path unconditional_images/generated_sample.png
```