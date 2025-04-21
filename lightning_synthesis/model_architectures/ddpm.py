import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
from piq import ssim

with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

learning_rate = config['learning_rate']
timesteps = config.get('timesteps', 1000)
label_dim = config.get('label_dim', 4)
guidance_scale = config.get('guidance_scale', 5.0)
attention_on = config.get('attention_on', True)
loss_type = config.get('loss_type', 'l1')
dynamic_thresholding = config.get('dynamic_thresholding', False) #inspired by Deepmind's Imagen

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.exp(-np.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1))
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.silu(self.norm2(self.conv2(x)))
        return x + skip

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock(in_channels, out_channels),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = ResBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Center crop skip connection to match upsampled x
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        skip = skip[:, :, diffY // 2 : diffY // 2 + x.size(2), diffX // 2 : diffX // 2 + x.size(3)]
        
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

print(f"[INFO] Attention is {'ON' if attention_on else 'OFF'}")
class MaybeAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = Residual(SpatialLinearAttention(dim)) if attention_on else nn.Identity()

    def forward(self, x):
        return self.attn(x)

# class DenoiseModel(nn.Module):
#     def __init__(self, label_dim, time_dim=256):
#         super().__init__()
#         self.label_embedding = nn.Embedding(label_dim, time_dim)
#         self.unconditional_embedding = nn.Parameter(torch.randn(1, time_dim))
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(time_dim),
#             nn.Linear(time_dim, time_dim),
#             nn.ReLU()
#         )

#         self.input_proj = nn.Conv2d(2, 64, kernel_size=3, padding=1)

#         self.down1_block = Down(64, 128)
#         self.down1_attn = MaybeAttentionBlock(128)
#         self.down2_block = Down(128, 256)
#         self.down2_attn = MaybeAttentionBlock(256)
#         self.down3_block = Down(256, 512)
#         self.down3_attn = MaybeAttentionBlock(512)
#         self.down4_block = Down(512, 1024)
#         self.down4_attn = MaybeAttentionBlock(1024)

#         self.mid = nn.Sequential(
#             ResBlock(1024, 1024),
#             MaybeAttentionBlock(1024),
#             ResBlock(1024, 1024)
#         )

#         self.up1_block = Up(1024 + 512, 512)
#         self.up1_attn = MaybeAttentionBlock(512)
#         self.up2_block = Up(512 + 256, 256)
#         self.up2_attn = MaybeAttentionBlock(256)
#         self.up3_block = Up(256 + 128, 128)
#         self.up3_attn = MaybeAttentionBlock(128)

#         self.up4 = Up(128 + 64, 64)

#         self.output_proj = nn.Conv2d(64, 1, kernel_size=3, padding=1)

#     def forward(self, x, t, labels=None):
#         time_emb = self.time_mlp(t)

#         if labels is not None and isinstance(labels, torch.Tensor) and (labels != -1).any():
#             valid_labels = labels.clone()
#             unconditional_mask = (labels == -1).unsqueeze(1)
#             label_emb = self.label_embedding(torch.clamp(valid_labels, min=0))
#             label_emb = torch.where(unconditional_mask, self.unconditional_embedding, label_emb)
#         else:
#             label_emb = self.unconditional_embedding.expand(x.size(0), -1)

#         emb = time_emb + label_emb
#         emb = emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
#         emb = emb.mean(dim=1, keepdim=True)
#         x = torch.cat([x, emb], dim=1)

#         x0 = self.input_proj(x)
#         x1 = self.down1_attn(self.down1_block(x0))
#         x2 = self.down2_attn(self.down2_block(x1))
#         x3 = self.down3_attn(self.down3_block(x2))
#         x4 = self.down4_attn(self.down4_block(x3))

#         x_mid = self.mid(x4)

#         x = self.up1_attn(self.up1_block(x_mid, x3))
#         x = self.up2_attn(self.up2_block(x, x2))
#         x = self.up3_attn(self.up3_block(x, x1))
#         x = self.up4(x, x0)

#         if dynamic_thresholding:
#             s = torch.quantile(x.view(x.shape[0], -1).abs(), 0.9, dim=1).view(-1, 1, 1, 1)
#             x = torch.clamp(x, -s, s) / s

#         return self.output_proj(x)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule for beta values (noise variance), as used in improved DDPM. Cosine schedules tend to result in better quality than linear ones."""
    steps = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = torch.cos((steps / timesteps + s) / (1 + s) * (torch.pi / 2)) ** 2
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)

# Diffusion Model with Classifier-Free Guidance
# class DDPM(pl.LightningModule):
#     """
#     Implements the DDPM (Denoising Diffusion Probabilistic Model) training framework.
#     Includes support for classifier-free guidance (unconditional + conditional loss blending).
#     """
#     def __init__(self, label_dim, learning_rate=learning_rate, timesteps=1000, guidance_scale=guidance_scale, verbose=False):
#         super().__init__()
#         self.model = DenoiseModel(label_dim)
#         self.learning_rate = learning_rate
#         self.timesteps = timesteps
#         self.verbose = verbose
#         self.guidance_scale = guidance_scale

#         self.register_buffer('betas', cosine_beta_schedule(timesteps))
#         self.register_buffer('alphas', 1.0 - self.betas)
#         self.register_buffer('alpha_hat', torch.cumprod(self.alphas, dim=0))

#         # Ensure alpha_hat never hits exactly 0 (avoids NaN division)
#         self.alpha_hat = torch.clamp(self.alpha_hat, min=1e-8)

#     def forward(self, x, t, labels=None):
#         if labels is not None and isinstance(labels, torch.Tensor) and (labels != -1).any():
#             noise_pred_cond = self.model(x, t, labels)  # Conditional
#             noise_pred_uncond = self.model(x, t, -1)    # Unconditional

#             noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
#             noise_pred = torch.clamp(noise_pred, -5, 5)
#         else:
#             noise_pred = self.model(x, t, -1)

#         return noise_pred

#     def training_step(self, batch, batch_idx):
#         imgs, labels = batch

#         batch_size = imgs.shape[0]
#         t = torch.randint(0, self.timesteps, (batch_size,), device=imgs.device).long()

#         noise = torch.randn_like(imgs, device=imgs.device)
#         noisy_imgs = self.q_sample(imgs, t, noise)

#         # Randomly drop labels during training (e.g., 10% of the time) for unconditional training
#         drop_labels = torch.rand(batch_size, device=imgs.device) < 0.1
#         conditional_labels = labels.clone().detach()
#         conditional_labels[drop_labels] = -1

#         predicted_noise = self.model(noisy_imgs, t, conditional_labels)
#         if loss_type == 'l1':
#             loss = F.l1_loss(predicted_noise, noise)
#         elif loss_type == 'mse':
#             loss = F.mse_loss(predicted_noise, noise)
#         else:
#             raise ValueError(f"Unknown loss_type: {loss_type}")

#         if self.verbose:
#             unconditional_ratio = (conditional_labels == -1).float().mean().item()
#             print(f"Batch {batch_idx}: Loss = {loss.item():.6f}, Unconditional Ratio = {unconditional_ratio:.2f}")


#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

#         # **Log reconstructions at fixed timesteps for consistent visualization**

#         if batch_idx % 2000 == 0:
#             fixed_timesteps = [100, 300, 500, 700, 900]
#             with torch.no_grad():
#                 grid_size = min(4, batch_size)  # Smaller grid to fit all t
#                 imgs_for_logging = imgs[:grid_size]
#                 labels_for_logging = labels[:grid_size]

#                 self.logger.experiment.add_images("Original Images", (imgs_for_logging + 1) / 2, global_step=self.current_epoch)

#                 for t_val in fixed_timesteps:
#                     t_fixed = torch.full((grid_size,), t_val, device=imgs.device, dtype=torch.long)
#                     noise = torch.randn_like(imgs_for_logging)
                    
#                     alpha_hat_t = self.alpha_hat[t_val].to(imgs.device)
#                     sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
#                     sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat_t)

#                     noisy_imgs = sqrt_alpha_hat * imgs_for_logging + sqrt_one_minus_alpha_hat * noise

#                     # Predict with classifier-free guidance
#                     noise_pred_cond = self.model(noisy_imgs, t_fixed, labels_for_logging)
#                     noise_pred_uncond = self.model(noisy_imgs, t_fixed, -1 * torch.ones_like(labels_for_logging))
#                     predicted_noise = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

#                     reconstructed = (noisy_imgs - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
#                     reconstructed = torch.clamp(reconstructed, -1, 1)
#                     # Compute SSIM on reconstructed images
#                     ssim_val = ssim(
#                         (imgs_for_logging + 1) / 2,
#                         torch.clamp((reconstructed[:grid_size] + 1) / 2, 0, 1),
#                         data_range=1.0,
#                         reduction='mean'
#                     )

#                     self.log('train_ssim', ssim_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

#                     # Normalize for TensorBoard
#                     self.logger.experiment.add_images(f"Denoised_t{t_val}", (reconstructed[:grid_size] + 1) / 2, global_step=self.current_epoch)

#         return loss

#     def q_sample(self, x_start, t, noise):
#         """ Diffuse the data (add noise) using the forward process q(x_t | x_0) """
#         sqrt_alpha_hat = self.alpha_hat[t].to(t.device)[:, None, None, None]
#         sqrt_one_minus_alpha_hat = (1.0 - self.alpha_hat[t].to(t.device))[:, None, None, None]
#         return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

#         # Option 1: Reduce on plateau
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=5, verbose=True
#         )

#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': {
#                 'scheduler': scheduler,
#                 'monitor': 'train_loss',  # must match your log key
#                 'frequency': 1,
#                 'interval': 'epoch'
#             }
#         }


## ============ SIMPLE VERSION for sanity ====================
class DenoiseModel(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        self.input_proj = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Downsampling path
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        # Bottleneck
        self.mid = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256)
        )

        # Upsampling path
        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)

        # Output projection
        self.output_proj = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, t, labels=None):  # labels ignored for unconditional
        emb = self.time_mlp(t)
        emb = emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        emb = emb.mean(dim=1, keepdim=True)  # down to 1 channel

        x = x + emb  # Add time embedding instead of concatenating

        x0 = self.input_proj(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        x_mid = self.mid(x2)

        x = self.up1(x_mid, x1)
        x = self.up2(x, x0)

        return self.output_proj(x)

class DDPM(pl.LightningModule):
    def __init__(self, label_dim, learning_rate=1e-4, timesteps=1000, loss_type='l1'):
        super().__init__()
        self.model = DenoiseModel()  # DenoiseModel (U-Net) instance
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        self.loss_type = loss_type

        self.register_buffer('betas', cosine_beta_schedule(timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alpha_hat', torch.cumprod(self.alphas, dim=0))
        self.alpha_hat = torch.clamp(self.alpha_hat, min=1e-8)  # avoid div by zero

    def forward(self, x, t):
        return self.model(x, t, None)  # unconditional

    def q_sample(self, x_start, t, noise):
        sqrt_alpha_hat = self.alpha_hat[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = (1.0 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise

    def training_step(self, batch, batch_idx):
        imgs, _ = batch  # labels ignored
        batch_size = imgs.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=imgs.device).long()
        noise = torch.randn_like(imgs)
        noisy_imgs = self.q_sample(imgs, t, noise)

        predicted_noise = self.model(noisy_imgs, t, None)

        if self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # Intermediate visualization every N batches
        if batch_idx % 2000 == 0:
            fixed_timesteps = [100, 300, 500, 700, 900, 999]
            grid_size = min(4, batch_size)

            with torch.no_grad():
                imgs_for_logging = imgs[:grid_size]

                # Log the original images (rescaled to [0,1])
                self.logger.experiment.add_images(
                    "Original Images", 
                    (imgs_for_logging + 1) / 2, 
                    global_step=self.global_step
                )

                for t_val in fixed_timesteps:
                    t_fixed = torch.full((grid_size,), t_val, device=imgs.device, dtype=torch.long)
                    noise = torch.randn_like(imgs_for_logging)

                    alpha_hat_t = self.alpha_hat[t_val].to(imgs.device)
                    sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
                    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat_t)

                    noisy_imgs = sqrt_alpha_hat * imgs_for_logging + sqrt_one_minus_alpha_hat * noise
                    predicted_noise = self.model(noisy_imgs, t_fixed, None)

                    x_recon = (noisy_imgs - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
                    x_recon = torch.clamp(x_recon, -1, 1)

                    self.logger.experiment.add_images(
                        f"Denoised_t{t_val}", 
                        (x_recon + 1) / 2, 
                        global_step=self.global_step
                    )

        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'frequency': 1,
                'interval': 'epoch'
            }
        }
