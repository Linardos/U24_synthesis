import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from conditional_models import CGAN, CVAE  # Assuming the CGAN class is in model.py
from data_loaders_synth import get_synthesis_dataloader
from torchvision import transforms

# Load configuration
with open('config_synth.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract parameters from config
batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
latent_dim = config.get('latent_dim', 100)
label_dim = config.get('label_dim', 4)
model_type = config.get('model_type', 'cvae')  # 'cvae' or 'cgan'
experiment_name = config['experiment_name']
model_dir = config.get('model_dir', 'saved_models')

# Prepare output directories
experiment_path = os.path.join('experiments', experiment_name)
os.makedirs(experiment_path, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}. Training model {model_type}")

# Initialize models and optimizers
if model_type == 'cvae':
    model = CVAE(latent_dim=latent_dim, label_dim=label_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
elif model_type == 'cgan':
    model = CGAN(latent_dim=latent_dim, label_dim=label_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
else:
    raise ValueError(f"Invalid model_type '{model_type}' in config.yaml. Choose 'cvae' or 'cgan'.")

# Data loading and transforms
def min_max_normalization(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

transform = transforms.Compose([
    transforms.Lambda(min_max_normalization),
])

dataloader = get_synthesis_dataloader(batch_size=batch_size, transform=transform)

# Training loop
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    epoch_loss = 0
    for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        imgs, labels = imgs.to(device), labels.to(device)

        if model_type == 'cvae':
            # Forward pass for CVAE
            gen_imgs, mu, logvar, z = model(imgs, labels)
            
            logvar = torch.clamp(logvar, min=-4, max=4)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            recon_loss = criterion(gen_imgs, imgs)
            loss = recon_loss + 0.1 * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        elif model_type == 'cgan':
            # Create noise vector
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, label_dim, (batch_size,), device=device)

            # Generate images
            gen_imgs = model.generate(z, gen_labels)

            # Train Discriminator
            real_validity = model.discriminate(imgs, labels)
            fake_validity = model.discriminate(gen_imgs.detach(), gen_labels)

            real_loss = criterion(real_validity, torch.ones_like(real_validity))
            fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            fake_validity = model.discriminate(gen_imgs, gen_labels)
            g_loss = criterion(fake_validity, torch.ones_like(fake_validity))

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            epoch_loss += g_loss.item() + d_loss.item()

    if epoch % 10 == 0:
        combined_imgs = torch.cat((imgs[:10], gen_imgs[:10]), dim=0)
        save_image(combined_imgs, 
                   os.path.join(experiment_path, f'epoch_{epoch}.png'), 
                   nrow=10,  
                   normalize=True)

    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, f'{model_type}_epoch_{epoch}.pth'))

    print(f"Epoch {epoch+1}/{num_epochs} completed. Avg Loss: {epoch_loss/len(dataloader):.4f}")

print("Training complete.")
