import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Load the config to get label dimension
with open('config_synth.yaml', 'r') as f:
    config = yaml.safe_load(f)
label_dim = config.get('label_dim', 4)
latent_dim = config.get('latent_dim', 100)

# Weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=latent_dim, label_dim=label_dim, img_channels=1):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 16384),
            nn.ReLU(),
            nn.Linear(16384, 1 * 256 * 256),
            nn.Tanh()
        )
        self.img_channels = img_channels

    def forward(self, z, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat((z, label_embedding), dim=1)
        x = self.model(x)
        return x.view(-1, self.img_channels, 256, 256)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, label_dim=label_dim, img_channels=1):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(1 * 256 * 256 + label_dim, 16384),
            nn.ReLU(),
            nn.Linear(16384, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.shape[0], -1)
        label_embedding = self.label_embedding(labels)
        x = torch.cat((img_flat, label_embedding), dim=1)
        return self.model(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Residual connection
        return F.relu(out)

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=256, label_dim=4, img_channels=1):
        super(ConditionalVAE, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, 256)

        # Encoder: Takes 2 channels (image + label)
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1),  # Output: [B, 32, 128, 128]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: [B, 64, 64, 64]
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: [B, 128, 32, 32]
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: [B, 256, 16, 16]
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(256 * 16 * 16, latent_dim)
        self.logvar_layer = nn.Linear(256 * 16 * 16, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + 256, 256 * 16 * 16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: [B, 128, 32, 32]
            nn.ReLU(),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: [B, 64, 64, 64]
            nn.ReLU(),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: [B, 32, 128, 128]
            nn.ReLU(),
            ResidualBlock(32),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),  # Output: [B, 1, 256, 256]
            nn.Tanh()  # Output is scaled to [-1,1]
        )

    def reparameterize(self, mu, logvar):
        # std = torch.sqrt(torch.clamp(torch.exp(logvar), min=1e-8))
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-2)  # Prevent vanishing variance
        print(f"std min: {std.min()}, max: {std.max()}")
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)  # Shape: [B, 256, 1, 1]
        label_embedding = label_embedding.expand(-1, -1, 256, 256)  # Now shape: [B, 256, 256, 256]

        # Ensure label embedding is a separate channel (not extra dimensions)
        label_embedding = label_embedding.mean(dim=1, keepdim=True)  # Reduce to [B, 1, 256, 256]

        # Concatenate image and label as separate channels
        x = torch.cat((x, label_embedding), dim=1)  # Now input is [B, 2, 256, 256]

        # Encode
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)  # Flatten before latent layers
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decode
        z = torch.cat((z, self.label_embedding(labels)), dim=1)  # Concatenate latent vector with label embedding
        x = self.decoder_input(z).view(-1, 256, 16, 16)
        gen_imgs = self.decoder(x)

        return gen_imgs, mu, logvar


def get_synthesis_model(model_name):
    if model_name == 'cgan':
        return ConditionalGenerator(), ConditionalDiscriminator()
    elif model_name == 'cvae':
        return ConditionalVAE()
    else:
        raise ValueError(f"Model {model_name} is not supported.")

if __name__ == "__main__":
    generator, discriminator = get_synthesis_model('cgan')
    vae = get_synthesis_model('cvae')
    print("Conditional GAN Generator:")
    print(generator)
    print("\nConditional GAN Discriminator:")
    print(discriminator)
    print("\nConditional VAE:")
    print(vae)
