import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
from torchmetrics.functional import structural_similarity_index_measure as ssim

with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)
label_dim = config.get('label_dim', 4)
latent_dim = config.get('latent_dim', 100)
learning_rate = config['learning_rate']


def masked_l1_loss(output, target, mask_threshold=0.0, weight=10.0):
    """
    Computes an L1 loss that gives more weight to non-zero regions.
    - `mask_threshold`: Pixels greater than this value are considered important.
    - `weight`: Importance factor for non-zero pixels.
    """
    mask = (target > mask_threshold).float() * weight + (target <= mask_threshold).float()
    return (mask * F.l1_loss(output, target, reduction='none')).mean()


# Weight Initialization using Xavier (Glorot) Initialization
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels)

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
            # nn.Sigmoid()  # Output scaled to [0,1]
        )

    def forward(self, z, labels):
        label_embedding = self.label_embedding(labels)
        z = torch.cat((z, label_embedding), dim=1)
        z = self.decoder_input(z).view(-1, 256, 16, 16)
        return self.decoder(z)

class CVAE(pl.LightningModule):
    def __init__(self, latent_dim=latent_dim, label_dim=label_dim, learning_rate=learning_rate, verbose=False):
        super().__init__()
        self.encoder = Encoder(latent_dim, label_dim)
        self.decoder = Decoder(latent_dim, label_dim)
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.apply(weights_init)
        # L1 loss for pixel-level differences
        # self.l1_loss_fn = nn.L1Loss()
        # Weights for combining L1 and SSIM losses
        self.l1_weight = 1
        self.ssim_weight = 0

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
        gen_imgs = torch.clamp(gen_imgs, 0, 1)
        logvar_clamped = torch.clamp(logvar, min=-4, max=4)
        kl_loss = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
    
        l1_loss = masked_l1_loss(gen_imgs, imgs)
        ssim_val = ssim(gen_imgs, imgs, data_range=1.0)
        ssim_loss = 1 - ssim_val
    
        recon_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        loss = l1_loss + 0.1 * kl_loss
    
        if self.verbose:
            debug_str = f"Batch {batch_idx}:\n"
            debug_str += f"  mu: mean={mu.mean().item():.6f}, std={mu.std().item():.6f}\n"
            debug_str += f"  logvar: mean={logvar.mean().item():.6f}, std={logvar.std().item():.6f}\n"
            debug_str += f"  KL loss: {kl_loss.item():.6f}\n"
            debug_str += f"  L1 loss: {l1_loss.item():.6f}\n"
            debug_str += f"  SSIM value: {ssim_val.item():.6f}\n"
            debug_str += f"  SSIM loss: {ssim_loss.item():.6f}\n"
            debug_str += f"  Reconstruction loss: {recon_loss.item():.6f}\n"
            debug_str += f"  Total loss: {loss.item():.6f}\n"
            print(debug_str)
    
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
