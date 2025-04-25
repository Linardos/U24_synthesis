import os
import yaml
import shutil
import re
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
from monai import transforms as mt

from data_loaders_l import NiftiSynthesisDataset
from model_architectures import DDPM, UNet, MonaiDDPM 

torch.set_float32_matmul_precision('medium')
# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

pl.seed_everything(config.get('seed', 42), workers=True)
# Extract parameters from config
batch_size = config['batch_size']
learning_rate = config['learning_rate']
label_dim = config.get('label_dim', 4)
resize_dim = config.get('resize_dim', False) #set false for no resizing

# ------------------------------------------------------------------------
# 0a.  House-keeping: define & organize experiment folder
# ------------------------------------------------------------------------
# Base directory for all experiments
base_dir = 'experiments'
os.makedirs(base_dir, exist_ok=True)

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

# Prepare output directories
experiment_name = f"{next_num:03}_{config['experiment_name']}"
experiment_path = os.path.join(base_dir, experiment_name)
os.makedirs(experiment_path, exist_ok=True)
# Save a copy of the config, training and data loading scripts for reproducibility
with open(os.path.join(experiment_path, 'config.yaml'), 'w') as out_f:
    yaml.dump(config, out_f)
shutil.copyfile('train_progressive.py', os.path.join(experiment_path, 'train_progressive.py'))
shutil.copyfile('data_loaders_l.py', os.path.join(experiment_path, 'data_loaders_l.py'))
shutil.copyfile('./model_architectures/monai_ddpm.py', os.path.join(experiment_path, 'monai_ddpm.py'))

# Define the root directory
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

# ------------------------------------------------------------------------
# 0b.  House-keeping: where we save checkpoints for every stage
# ------------------------------------------------------------------------
STAGES = [64, 128, 256, 512]           # target “output” resolutions
EPOCHS = [40, 20, 10, 10]             # how long each stage trains
CKPT_DIR = os.path.join(experiment_path, "checkpoints")

# ------------------------------------------------------------------------
# 1.  Tiny helper that returns a DataLoader for a given resolution
# ------------------------------------------------------------------------
def make_loader(spatial, batch):
    tfm = mt.Compose(
        [
            mt.LoadImaged(keys=["image"], image_only=True),
            mt.SqueezeDimd(keys=["image"], dim=-1),              # (H,W,1)→(H,W)
            mt.EnsureChannelFirstd(keys=["image"]),              # → (1,H,W)
            mt.Resized(keys=["image"], spatial_size=[spatial]*2,
                       mode="bilinear"),                         # <- res-switch
            mt.ScaleIntensityRanged(keys=["image"],
                                    a_min=0., a_max=255.,
                                    b_min=0., b_max=1., clip=True),
            mt.ToTensord(keys=["image"]),
            mt.RandLambdad(keys=["class"], prob=0.15, func=lambda x: -1 * torch.ones_like(x)),
            mt.Lambdad(
                keys=["class"],
                func=lambda x: x.clone().detach().to(torch.float32).unsqueeze(0).unsqueeze(0)
                if isinstance(x, torch.Tensor)
                else torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
        ]
    )
    ds = NiftiSynthesisDataset(full_data_path, transform=tfm)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True,
        num_workers=8, persistent_workers=True, pin_memory=True)

# ------------------------------------------------------------------------
# 2.  Create the model *once* (in_channels/out_channels don’t change)
# ------------------------------------------------------------------------
model = MonaiDDPM(lr=learning_rate, T=1000)

# ------------------------------------------------------------------------
# 3.  Loop over stages – fit, swap DataLoader, resume
# ------------------------------------------------------------------------

print(f"Initiating Experiment {experiment_name}...")
resume_ckpt = None          # path of last stage’s best model
for stage, (res, num_epochs) in enumerate(zip(STAGES, EPOCHS), 1):

    print(f"\n=== Stage {stage}: training at {res}×{res} for {num_epochs} epochs ===")

    # • Lightning logger will create sub-folders automatically
    tb_logger = pl_loggers.TensorBoardLogger(
        'logs/', name=f"{experiment_name}_{res}px"
    )

    # • new Trainer each time – cheapest way to change max_epochs / callbacks
    trainer = pl.Trainer(
        # max_epochs=num_epochs,            # Comment out for dev test
        max_epochs=3,                     # Uncomment for dev test
        limit_train_batches=0.02,         # Uncomment for dev test
        limit_val_batches=0.02,           # Uncomment for dev test
        accelerator="auto",
        precision=16,
        logger=tb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(experiment_path, "checkpoints"), # one dir for all stages
                filename=f"best_{res}px",
                save_top_k=1, monitor="train_loss", mode="min"),
            LearningRateMonitor(logging_interval='epoch')],
        accumulate_grad_batches=2,             # keeps “effective” batch big
        benchmark=True, deterministic=False,
    )

    # • (re)load last checkpoint before jumping to the next res
    if resume_ckpt is not None:
        model = MonaiDDPM.load_from_checkpoint(resume_ckpt)

    # • stage-specific DataLoader
    loader = make_loader(res, batch_size)
    img_batch, _ = next(iter(loader))
    print("Sample batch shapes:")
    print(img_batch.shape, img_batch.min().item(), img_batch.max().item())

    # • train
    trainer.fit(model, train_dataloaders=loader)

    # • remember best ckpt path for next loop iteration
    resume_ckpt = trainer.checkpoint_callback.best_model_path


print(f"✅ Experiment {experiment_name} completed.")