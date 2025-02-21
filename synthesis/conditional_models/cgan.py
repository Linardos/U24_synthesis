import os
import torch
import torch.nn as nn
import yaml

# Load configuration
with open('config_synth.yaml', 'r') as f:
    config = yaml.safe_load(f)

latent_dim = config.get('latent_dim', 100)
label_dim = config.get('label_dim', 4)

# Weight initialization function
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class CGAN(nn.Module):
    def __init__(self, latent_dim=latent_dim, label_dim=label_dim, img_channels=1):
        super(CGAN, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.img_channels = img_channels

        # Generator
        self.generator = nn.Sequential(
            nn.Embedding(label_dim, label_dim),
            nn.Linear(latent_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * 256 * 256),
            nn.Tanh()
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Embedding(label_dim, label_dim),
            nn.Linear(img_channels * 256 * 256 + label_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Apply weight initialization
        self.apply(weights_init)

    def generate(self, z, labels):
        label_embedding = self.generator[0](labels)
        x = torch.cat((z, label_embedding), dim=1)
        x = self.generator[1:](x)
        return x.view(-1, self.img_channels, 256, 256)

    def discriminate(self, img, labels):
        img_flat = img.view(img.shape[0], -1)
        label_embedding = self.discriminator[0](labels)
        x = torch.cat((img_flat, label_embedding), dim=1)
        return self.discriminator[1:](x)
