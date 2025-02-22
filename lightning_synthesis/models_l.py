import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml

with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)
label_dim = config.get('label_dim', 4)
latent_dim = config.get('latent_dim', 100)
learning_rate = config['learning_rate']

# Weight Initialization
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm([in_channels, 256, 256])
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([in_channels, 256, 256])

    def forward(self, x):
        identity = x
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity  # Residual connection
        return torch.relu(out)

class Encoder(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(label_dim, latent_dim)
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

    def forward(self, x, labels):
        label_embedding = self.label_embedding(labels)
        label_channel = label_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 256, 256)
        label_channel = label_channel[:, :1, :, :]

        x = torch.cat((x, label_channel), dim=1)
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)

        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(label_dim, latent_dim)
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
            nn.Sigmoid()  # Output scaled to [0,1]
        )

    def forward(self, z, labels):
        label_embedding = self.label_embedding(labels)
        z = torch.cat((z, label_embedding), dim=1)
        z = self.decoder_input(z).view(-1, 256, 16, 16)
        return self.decoder(z)

class CVAE(pl.LightningModule):
    def __init__(self, latent_dim=latent_dim, label_dim=label_dim, learning_rate=learning_rate):
        super().__init__()
        self.encoder = Encoder(latent_dim, label_dim)
        self.decoder = Decoder(latent_dim, label_dim)
        self.learning_rate = learning_rate
        self.apply(weights_init)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-6)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        gen_imgs = self.decoder(z, labels)
        return gen_imgs, mu, logvar, z

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        gen_imgs, mu, logvar, z = self.forward(imgs, labels)

        logvar = torch.clamp(logvar, min=-4, max=4)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        gen_imgs = torch.clamp(gen_imgs, 0, 1)
        recon_loss = self.criterion(gen_imgs, imgs)
        loss = recon_loss + 0.1 * kl_loss
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
