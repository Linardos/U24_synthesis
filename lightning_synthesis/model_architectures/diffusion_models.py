import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl

# from unet import UNet
class DDPM(pl.LightningModule):

    def __init__(self, T: int, noise_predictor: nn.Module, lr: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.T = T #timesteps
        self.noise_predictor = noise_predictor 
        self.lr = lr

        beta = torch.linspace(1e-4, 0.02, T)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        # Register buffers so they're automatically moved to the correct device
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

    def forward(self, x, t):
        return self.noise_predictor(x, t)

    def training_step(self, batch, batch_idx):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """

        x0, _ = batch # ignore labels for now
        batch_size = x0.size(0)

        device = x0.device
        t = torch.randint(1, self.T + 1, (batch_size,), device=device, dtype=torch.long)

        eps = torch.randn_like(x0)

        # Take one gradient descent step
        alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
        eps_predicted = self.noise_predictor(x_t, t - 1)
        loss = nn.functional.mse_loss(eps, eps_predicted)
        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32),
                 use_tqdm=True):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models - Reverse Diffusion
        """

        device = self.beta.device
        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]), device=device)
        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            t = torch.ones(n_samples, dtype=torch.long, device=device) * t

            beta_t = self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(
                1 - alpha_bar_t)) * self.noise_predictor(x, t - 1))
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
        return x
