import os
import yaml
import shutil
import re
import numpy as np
from pathlib import Path
import glob
import time
from collections import defaultdict
import random

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
from monai import transforms as mt

from data_loaders_l import NiftiSynthesisDataset, BalancedSamplingDataModule
from model_architectures import UNet, MonaiDDPM, ConditionalWGAN

# ── CONFIG ────────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision('medium')
# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

pl.seed_everything(config.get('seed', 42), workers=True)
# Extract parameters from config
batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
resize_dim = config.get('resize_dim', False) #set false for no resizing
# Prepare output directories
# Base directory for all experiments
base_dir = Path("experiments")
base_dir.mkdir(exist_ok=True)


# ── EXPERIMENT FOLDER ──────────────────────────────────────────────────────────
# Get existing experiment directories and find the highest prefix
existing = [
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and re.match(r'^\d{3}_', d)
]

if existing:
    # Extract numeric prefixes
    nums = [int(re.match(r'^(\d{3})_', name).group(1)) for name in existing]
    next_num = max(nums) + 1
else:
    next_num = 1

cfg_exp_name = config["experiment_name"]          
exp_dir_manual = base_dir / cfg_exp_name

if exp_dir_manual.exists():
    # if an existing experiment name has been given, continue training
    print(f"⚠️  experiment folder '{cfg_exp_name}' already exists – will resume if possible")
    experiment_name = cfg_exp_name
    experiment_path = exp_dir_manual
else:
    # create a new experiment and copy current set up
    existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name[:3].isdigit()]
    next_num = (max(int(d.name[:3]) for d in existing) + 1) if existing else 1
    experiment_name = f"{next_num:03d}_{config['model']}_augmentations{config['augmentations']}_{cfg_exp_name}"
    experiment_path = base_dir / experiment_name
    experiment_path.mkdir(exist_ok=True)
    
    with open(os.path.join(experiment_path, 'config.yaml'), 'w') as out_f:
        yaml.dump(config, out_f)
    shutil.copyfile('train.py', os.path.join(experiment_path, 'train.py'))
    shutil.copyfile('data_loaders_l.py', os.path.join(experiment_path, 'data_loaders_l.py'))
    shutil.copyfile('./model_architectures/monai_ddpm.py', os.path.join(experiment_path, 'monai_ddpm.py'))


# Prepare output directories
# experiment_name = f"{next_num:03}_{config['experiment_name']}_{resize_dim}x{resize_dim}"
# experiment_path = os.path.join(base_dir, experiment_name)
# os.makedirs(experiment_path, exist_ok=True)
# Save a copy of the config, training and data loading scripts for reproducibility
# with open(os.path.join(experiment_path, 'config.yaml'), 'w') as out_f:
#     yaml.dump(config, out_f)
# shutil.copyfile('train.py', os.path.join(experiment_path, 'train.py'))
# shutil.copyfile('data_loaders_l.py', os.path.join(experiment_path, 'data_loaders_l.py'))
# shutil.copyfile('./model_architectures/monai_ddpm.py', os.path.join(experiment_path, 'monai_ddpm.py'))



# ----------------------------------------------------------------------
#  1. DATA HANDLING
# ----------------------------------------------------------------------

# Define the data path directory
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

# UNCOMMENT FOR MNIST SANITY CHECK ====
# from torchvision import datasets

# transform = transforms.Compose([
#     transforms.Resize((32, 32)), 
#     transforms.ToTensor()
# ])

# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# UNCOMMENT FOR SANITY CHECK ====

transform_list =     [
        mt.LoadImaged(keys=["image"], image_only=True),
        mt.SqueezeDimd(keys=["image"], dim=-1), # (H,W,1) → (H,W)
        mt.EnsureChannelFirstd(keys=["image"]), # (1,H,W)


        # Crop relevant area
        # mt.CropForegroundd(keys=["image"], source_key="image"),
        # mt.Resized(keys=["image"], spatial_size=(resize_dim, resize_dim),
        #            mode="bilinear", align_corners=False),

        # local-contrast aug now hits only 20 % of the images
        # mt.RandAdjustContrastd(keys=["image"],
        #                        prob=0.20, gamma=(0.9, 1.1)),
        # mt.RandHistogramShiftd(keys=["image"],
        #                        prob=0.20, num_control_points=6),

        # normalize
        # mt.Lambdad(keys=["image"],
        #            func=lambda img: (img - img.mean()) / (img.std() + 1e-8)),
        # mt.ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0), # scale to [-1,1]. Diffusion Models do better if centered on a 0 mean
        # mt.ToTensord(keys=["image"]),

        # conditionality stuff, applied to labels
        # mt.RandLambdad(keys=["class"], prob=0.15, func=lambda x: -1 * torch.ones_like(x)),
        # mt.Lambdad(
        #     keys=["class"],
        #     func=lambda x: x.clone().detach().to(torch.float32).unsqueeze(0).unsqueeze(0)
        #     if isinstance(x, torch.Tensor)
        #     else torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # )
    ]

geometric_augmentations = [
    mt.RandAffined(
        keys=["image"],
        prob=0.9,
        rotate_range=[0, 0, np.pi / 12],   # ±15°
        shear_range=[0.1, 0.1],            # up to 10 %
        translate_range=[0.05, 0.05],      # ±5 %
        scale_range=[0.05, 0.05],          # ±5 %
        mode="bilinear",
        padding_mode="zeros",
        allow_missing_keys=False,
    ),
    mt.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
]

intensity_augmentations = [
    # ---------------- intensity -----------------
    mt.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.9, 1.1)),
    mt.RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0,
                          std=(0.0, 0.02)),                  # subtle noise
]

if config['augmentations']=='geometric':
    transform_list.extend(geometric_augmentations)
elif config['augmentations']=='all':
    transform_list.extend(geometric_augmentations)
    transform_list.extend(intensity_augmentations)
else:
    pass

# normalise AFTER all augs
transform_list.extend([
    mt.Lambdad(keys=["image"],
               func=lambda img: (img - img.mean()) / (img.std() + 1e-8)),
    mt.ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
    mt.ToTensord(keys=["image"]),
])

if config["model"] == "GAN":
    transform_list.extend([
        mt.Resized(keys=["image"], spatial_size=(64, 64),
                mode="bilinear", align_corners=False),
        mt.Lambdad(keys=["class"],
                func=lambda x: torch.as_tensor(x, dtype=torch.long).squeeze())
    ])

else:  # DDPM conditionality stuff, applied to labels
    transform_list.extend([
        mt.RandLambdad(keys=["class"], prob=0.15,
                       func=lambda x: -1 * torch.ones_like(x)),
        mt.Lambdad(
            keys=["class"],
            func=lambda x: x.clone().detach().to(torch.float32)
                .unsqueeze(0).unsqueeze(0)
            if isinstance(x, torch.Tensor)
            else torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
    ])


train_transforms = mt.Compose(transform_list)


if config['dynamic_balanced_sampling']: # ASSURE EQUAL CLASS NUMBER PER EPOCH. This way the synthetic model is not biased toward a benign prior
    train_loader = BalancedSamplingDataModule(
        full_data_path=full_data_path,
        batch_size=batch_size,
        transform=train_transforms,
        num_workers=8,
        ratio=config.get("benign_to_malignant_ratio", 1.0),
    )
    reload_dataloaders_every_n_epochs = 1
else:
    dataset = NiftiSynthesisDataset(full_data_path, transform=train_transforms, samples_per_class=config['samples_per_class'])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    reload_dataloaders_every_n_epochs = 0

if isinstance(train_loader, pl.LightningDataModule):
    # Initialise pools (once) then pull a loader
    train_loader.setup("fit")
    sample_loader = train_loader.train_dataloader()
else:
    sample_loader = train_loader

img_batch, _ = next(iter(sample_loader))
print(img_batch.shape, img_batch.min().item(), img_batch.max().item())



# ----------------------------------------------------------------------
# 2. MODEL: resume checkpoint or fresh start
# ----------------------------------------------------------------------

def find_latest_ckpt(exp_dir):
    """Return newest .ckpt file inside <exp_dir>/checkpoints (or None)."""
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    # sort by modification time, newest first
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]

resume_ckpt = find_latest_ckpt(experiment_path)
if resume_ckpt:
    print(f"🔄  resuming from checkpoint: {resume_ckpt}")
    mdl_cls = MonaiDDPM if config["model"] == "DDPM" else ConditionalWGAN
    model   = mdl_cls.load_from_checkpoint(resume_ckpt)
    
else:
    print("🆕  no checkpoint found – starting from scratch")
    print(f"Using model {config['model']}..")
    if config['model'] == 'DDPM':
        model = MonaiDDPM(lr=learning_rate, T=1000)
    elif config['model'] == 'GAN':
        model = ConditionalWGAN(n_classes=2, z_dim=128, lr=1e-4,
                        n_critic=5, grad_penalty_weight=10.0)

# if config['conditional']:
#     model = MonaiDDPM(lr=learning_rate, T=1000)
# else:
#     model = MonaiDDPM_unconditional(lr=learning_rate, T=1000)

print(f"Model initialized & EMBED loaded. Initiating experiment {experiment_name}")

# ----------------------------------------------------------------------
# 3.  LIGHTNING TRAINER
# ----------------------------------------------------------------------

# Set up callbacks
tb_logger = pl_loggers.TensorBoardLogger('logs/', name=experiment_name)

metric_to_watch = "wasserstein" if config["model"] == "GAN" else "train_loss"

checkpoint_callback = ModelCheckpoint(
    dirpath=experiment_path / "checkpoints",
    filename="{epoch:02d}-{step}",
    monitor=metric_to_watch,
    mode="min",
    save_top_k=1,
)

patience = 5 if config ['model'] == 'DDPM' else 10
early_stopping = EarlyStopping(
    monitor=metric_to_watch,
    mode="min",
    patience=patience,
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Set up Trainer
clip_val = 1.0 if config["model"] == "DDPM" else 0.0
grad_accum = 4 if config["model"] == "DDPM" else 1
use_amp = False if config["model"] == "GAN" else True

trainer = pl.Trainer(
    fast_dev_run=config['fast_dev_run'],
    reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs, # for balancing per-epoch
    max_epochs=num_epochs,
    accelerator="auto",
    precision = 16 if use_amp else 32,   # full FP32 for the GAN
    logger=tb_logger,
    callbacks=[checkpoint_callback, early_stopping, lr_monitor],
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    gradient_clip_val=clip_val,
    accumulate_grad_batches=grad_accum,
    **({"resume_from_checkpoint": resume_ckpt} if resume_ckpt else {})
    )


trainer.fit(model, train_dataloaders=train_loader)

print('Training complete!')

