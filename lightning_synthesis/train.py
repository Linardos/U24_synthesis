import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
from torchvision.utils import save_image
from torchvision import transforms

from data_loaders_l import SynthesisDataModule, NiftiSynthesisDataset
# from model_architectures import CVAE, DDPM
from model_architectures import DDPM, UNet


torch.set_float32_matmul_precision('medium')
# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

pl.seed_everything(config.get('seed', 42), workers=True)
# Extract parameters from config
batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
latent_dim = config.get('latent_dim', 100)
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

# UNCOMMENT FOR MNIST SANITY CHECK ====
# from torchvision import datasets

# transform = transforms.Compose([
#     transforms.Resize((32, 32)), 
#     transforms.ToTensor()
# ])

# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# UNCOMMENT FOR SANITY CHECK ====

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resize_dim) if resize_dim else transforms.Lambda(lambda x: x),
    # transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # Rescale to [0,1]
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.to(torch.float32))
])



# Define the root directory
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)
dataset = NiftiSynthesisDataset(full_data_path, transform=transform, samples_per_class=None)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("EMBED loaded")
# data_module = SynthesisDataModule(full_path=full_data_path, batch_size=batch_size, transform=transform)
# train_loader = data_module.train_dataloader()
# Initialize model
noise_predictor = UNet()
model = DDPM(1000, noise_predictor)

# Set up logger
tb_logger = pl_loggers.TensorBoardLogger('logs/', name=experiment_name)

# Set up callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(experiment_path, "checkpoints"),
    filename=f"{model_type}" + "-{epoch:02d}-{train_loss_epoch:.4f}",
    save_top_k=1,
    monitor="train_loss_epoch",
    mode="min",
)
early_stopping = EarlyStopping(
    monitor="train_loss_epoch",
    patience=15,
    mode="min",
    check_on_train_epoch_end=True,
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Set up Trainer
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator="auto",
    precision=32,
    logger=tb_logger,
    callbacks=[checkpoint_callback, early_stopping, lr_monitor],
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    gradient_clip_val=1.0,
)

# Start training
trainer.fit(model, train_dataloaders=train_loader)
# trainer.fit(model, data_module) # train_dataloaders=train_loader)

print('Training complete!')
