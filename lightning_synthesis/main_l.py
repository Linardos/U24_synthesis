import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
from torchvision.utils import save_image
from torchvision import transforms

from data_loaders_l import SynthesisDataModule
from models_l import CVAE

torch.set_float32_matmul_precision('medium')
# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract parameters from config
batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
latent_dim = config.get('latent_dim', 100)
label_dim = config.get('label_dim', 4)
experiment_name = config['experiment_name']
model_dir = config.get('model_dir', 'saved_models')

# Prepare output directories
experiment_path = os.path.join('experiments', experiment_name)
os.makedirs(experiment_path, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Set up data module
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # Rescale to [0,1]. Necessary to do explicitly with nibabel as ToTensor will omit it.
    transforms.Normalize((0.5,), (0.5,)),  # Then normalize if needed
    transforms.Lambda(lambda x: x.to(torch.float32))
])
data_module = SynthesisDataModule(batch_size=batch_size, transform=transform)

# Initialize model
model = CVAE(latent_dim=latent_dim, label_dim=label_dim, learning_rate=learning_rate)

# Set up logger
tb_logger = pl_loggers.TensorBoardLogger('logs/', name=experiment_name)

# Set up callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=model_dir,
    filename='cvae-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    monitor='train_loss', # we don't use a validation since we only care to reconstruct the entire in-distribution data
    mode='min'
)
early_stopping_callback = EarlyStopping(monitor='train_loss', patience=10, mode='min')
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Set up Trainer
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    logger=tb_logger,
    callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
    enable_progress_bar=True,
)

# Start training
trainer.fit(model, data_module)

print('Training complete!')
