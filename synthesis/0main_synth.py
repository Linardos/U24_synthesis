import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from monai_models import MONAICVAE
from models_synth import get_synthesis_model  # Import synthesis models
from data_loaders_synth import get_synthesis_dataloader  # Import synthesis data loader
from torchvision import transforms


# Load configuration
with open('config_synth.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract parameters from config
batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
latent_dim = config['latent_dim']
label_dim = config['label_dim']
experiment_name = config['experiment_name']
model_dir = config.get('model_dir', 'saved_models')

# Define experiment folder
experiment_path = os.path.join('experiments', experiment_name)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

# Define model saving path
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Get models
print("Initializing models...")
if config['model_type'] == 'cgan':
    generator, discriminator = get_synthesis_model('cgan')
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    print("CGAN models loaded.")
elif config['model_type'] == 'cvae':
    vae = get_synthesis_model('cvae')
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    print("CVAE model loaded.")
else:
    raise ValueError("Invalid model_type in config.yaml")



# Load data
def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

transform = transforms.Compose([
    transforms.Lambda(lambda x: min_max_normalization(x)),  # Min-max normalization
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1,1] (Tanh output)
])
dataloader = get_synthesis_dataloader(batch_size=batch_size, transform=transform)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move models to device
if config['model_type'] == 'cgan':
    generator.to(device)
    discriminator.to(device)
elif config['model_type'] == 'cvae':
    vae.to(device)

print("Starting training...")


# Training loop
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    epoch_loss = 0
    for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.shape[0]
        if torch.isnan(imgs).any() or torch.isinf(imgs).any():
            print("Warning: NaN or Inf detected in input images!")


        if config['model_type'] == 'cgan':
            # Create noise vector
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, label_dim, (batch_size,), device=device)
            
            # Generate images
            gen_imgs = generator(z, gen_labels)

            # Train Discriminator
            real_validity = discriminator(imgs, labels)
            fake_validity = discriminator(gen_imgs.detach(), gen_labels)
            d_loss = criterion(real_validity, torch.ones_like(real_validity)) + \
                    criterion(fake_validity, torch.zeros_like(fake_validity))
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            fake_validity = discriminator(gen_imgs, gen_labels)
            g_loss = criterion(fake_validity, torch.ones_like(fake_validity))
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            
            epoch_loss += g_loss.item() + d_loss.item()
            
            # if i % 10 == 0:
            #     print(f"Batch {i}: Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")
        
        elif config['model_type'] == 'cvae':
            print(f"imgs min: {imgs.min()}, max: {imgs.max()}")
            if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                print("NaN detected in input images!")
                exit()

            gen_imgs, mu, logvar = vae(imgs, labels)

            logvar = torch.clamp(logvar, min=-4, max=4)  # Prevent extreme values
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            gen_imgs = torch.clamp(gen_imgs, -1, 1)  # Keep values inside valid range
            scaled_imgs = imgs * 2 - 1

            print(f"mu min: {mu.min()}, max: {mu.max()}")
            print(f"logvar min: {logvar.min()}, max: {logvar.max()}")

            recon_loss = criterion(gen_imgs, scaled_imgs)
            loss = recon_loss + 0.1 * kl_loss  # Scale KL loss down

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Batch {i}: Reconstruction Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")

    # Save generated samples
    if epoch % 10 == 0:
        save_image(gen_imgs[:25], os.path.join(experiment_path, f'epoch_{epoch}.png'), nrow=5, normalize=True)
        print(f"Saved generated images at epoch {epoch}")
    
    # Save model every 5 epochs
    if epoch % 5 == 0:
        if config['model_type'] == 'cgan':
            torch.save(generator.state_dict(), os.path.join(model_dir, f'generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f'discriminator_epoch_{epoch}.pth'))
        elif config['model_type'] == 'cvae':
            torch.save(vae.state_dict(), os.path.join(model_dir, f'vae_epoch_{epoch}.pth'))
        print(f"Saved model at epoch {epoch}")

    print(f"Epoch {epoch+1}/{num_epochs} completed. Avg Loss: {epoch_loss/len(dataloader):.4f}")

print("Training complete.")
