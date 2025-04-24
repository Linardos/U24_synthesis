import os
import yaml
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
from torchvision.utils import save_image
# from torchvision import transforms
from monai import transforms as mt

from data_loaders_l import SynthesisDataModule, NiftiSynthesisDataset
from model_architectures import DDPM, UNet, MonaiDDPM 

torch.set_float32_matmul_precision('medium')
# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

pl.seed_everything(config.get('seed', 42), workers=True)
# Extract parameters from config
batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
label_dim = config.get('label_dim', 4)
experiment_name = config['experiment_name']
model_type = config['model_type']
resize_dim = config.get('resize_dim', False) #set false for no resizing
# Prepare output directories
experiment_path = os.path.join('experiments', experiment_name)
os.makedirs(experiment_path, exist_ok=True)
# Save a copy of the config for reproducibility
with open(os.path.join(experiment_path, 'config.yaml'), 'w') as out_f:
    yaml.dump(config, out_f)

# Define the root directory
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


train_transforms = mt.Compose(
    [
        mt.LoadImaged(keys=["image"], image_only=True),
        mt.SqueezeDimd(keys=["image"], dim=-1), # (H,W,1) → (H,W)
        mt.EnsureChannelFirstd(keys=["image"]), # (1,H,W)
        # mt.Resized(keys=["image"], spatial_size=[64, 64], mode="bilinear"),
        mt.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        mt.ToTensord(keys=["image"]),

    ]
)

dataset = NiftiSynthesisDataset(full_data_path, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
img_batch, _ = next(iter(train_loader))
print(img_batch.shape, img_batch.min().item(), img_batch.max().item())


model = MonaiDDPM(lr=learning_rate, T=1000)
print("Model initialized & EMBED loaded. Sanity check (batch min/max/shape):")

# Set up callbacks
tb_logger = pl_loggers.TensorBoardLogger('logs/', name=experiment_name)

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(experiment_path, "checkpoints"),
    filename=f"{model_type}-{{epoch:02d}}-{{step}}",
    auto_insert_metric_name=True,
    save_top_k=1,
    monitor="train_loss",
    mode="min",
)
early_stopping = EarlyStopping(
    monitor="train_loss",
    patience=15,
    mode="min",
    check_on_train_epoch_end=True,
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Set up Trainer
trainer = pl.Trainer(
    fast_dev_run=False, #set to true for tests
    max_epochs=num_epochs,
    accelerator="auto",
    precision=32,
    logger=tb_logger,
    callbacks=[checkpoint_callback, early_stopping, lr_monitor],
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    gradient_clip_val=1.0,
    accumulate_grad_batches=4
    )


trainer.fit(model, train_dataloaders=train_loader)

print('Training complete!')

