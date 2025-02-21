import torch
import torch.nn as nn
import yaml

with open('config_synth.yaml', 'r') as f:
    config = yaml.safe_load(f)
label_dim = config.get('label_dim', 4)
latent_dim = config.get('latent_dim', 100)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Residual connection
        return torch.relu(out)

class CVAE(nn.Module):
    def __init__(self, latent_dim=latent_dim, label_dim=label_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(label_dim, latent_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(256 * 16 * 16, latent_dim)
        self.logvar_layer = nn.Linear(256 * 16 * 16, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim * 2, 256 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output scaled to [-1,1]
        )
        
        # Apply weight initialization to the entire model
        self.apply(weights_init)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-6)  # Prevent vanishing std
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        label_embedding = self.label_embedding(labels)
        label_channel = label_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 256, 256)
        label_channel = label_channel[:, :1, :, :]
        
        x = torch.cat((x, label_channel), dim=1)
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        z = self.reparameterize(mu, logvar)
        
        z = torch.cat((z, label_embedding), dim=1)
        z = self.decoder_input(z).view(-1, 256, 16, 16)
        gen_imgs = self.decoder(z)
        
        return gen_imgs, mu, logvar, z
