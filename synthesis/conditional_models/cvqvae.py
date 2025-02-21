# Custom CVQVAE with CNN-Based Encoder, Residual Blocks, and Vector Quantization
import torch
import torch.nn as nn
import yaml

with open('config_synth.yaml', 'r') as f:
    config = yaml.safe_load(f)
label_dim = config.get('label_dim', 4)
latent_dim = config.get('latent_dim', 100)
codebook_dim = 512  # Number of discrete codes

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

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=codebook_dim, latent_dim=latent_dim):
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
    
    def forward(self, z):
        print(f"Initial z shape: {z.shape}")
        print(f"Codebook weight shape: {self.codebook.weight.shape}")

        if z.dim() == 2:  # Flattened input
            batch_size, num_features = z.shape
            num_channels = self.codebook.weight.shape[1]
            z_flattened = z.view(batch_size, -1, num_channels)
        else:
            batch_size, num_channels, height, width = z.shape
            z_flattened = z.view(batch_size, num_channels, -1).permute(0, 2, 1)  

        print(f"z_flattened shape: {z_flattened.shape}")
        print(f"codebook weight shape: {self.codebook.weight.T.shape}")

        codebook = self.codebook.weight
        distances = (
            (z_flattened ** 2).sum(dim=-1, keepdim=True)  
            - 2 * (z_flattened @ codebook.T)  
            + (codebook ** 2).sum(dim=-1)  
        )

        print(f"distances shape: {distances.shape}")
        encoding_indices = torch.argmin(distances, dim=-1)
        print(f"encoding_indices shape: {encoding_indices.shape}")
        z_q = self.codebook(encoding_indices).permute(0, 2, 1).reshape(z.shape)
        print(f"z_q shape: {z_q.shape}")

        return z_q, encoding_indices

class CVQVAE(nn.Module):
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
        
        self.vector_quantizer = VectorQuantizer(num_codes=codebook_dim, latent_dim=latent_dim)
        
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
            nn.Tanh()  # Output scaled to [-1,1]
        )

    def forward(self, x, labels):
        label_embedding = self.label_embedding(labels)
        label_channel = label_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 256, 256)
        label_channel = label_channel[:, :1, :, :]
        
        x = torch.cat((x, label_channel), dim=1)
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        
        z_q, encoding_indices = self.vector_quantizer(x)
        
        z = torch.cat((z_q, label_embedding), dim=1)
        z = self.decoder_input(z).view(-1, 256, 16, 16)
        gen_imgs = self.decoder(z)
        
        return gen_imgs, z_q, encoding_indices
