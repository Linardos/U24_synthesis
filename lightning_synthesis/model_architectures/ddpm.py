import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml
import torch.nn.functional as F
import numpy as np

with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

learning_rate = config['learning_rate']
timesteps = config.get('timesteps', 1000)  # Number of diffusion steps
label_dim = config.get('label_dim', 4)
guidance_scale = config.get('guidance_scale', 5.0)  # Strength of classifier-free guidance

class SinusoidalPositionEmbeddings(nn.Module):
    """Creates a sinusoidal embedding of diffusion timestep `t`, so the model knows where in the noise schedule it is."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.exp(-np.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1))
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class DenoiseModel(nn.Module):
    """
    A lightweight U-Net-like model for denoising, with classifier-free guidance.
    Takes in a noisy image, timestep `t`, and optional labels.
    Concatenates image + timestep + label embedding and predicts the noise.
    """
    def __init__(self, label_dim, time_dim=256):
        super().__init__()
        self.label_dim = label_dim
        self.label_embedding = nn.Embedding(label_dim, time_dim)
        self.unconditional_embedding = nn.Parameter(torch.randn(1, time_dim))
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

        self.activation = nn.SiLU()

    def forward(self, x, t, labels=None):
        # Time embeddings
        time_emb = self.time_mlp(t)

        if labels is not None and isinstance(labels, torch.Tensor) and (labels != -1).any():
            valid_labels = labels.clone()
            unconditional_mask = (labels == -1).unsqueeze(1)
            
            # Ensure valid indices for label embedding
            label_emb = self.label_embedding(torch.clamp(valid_labels, min=0))
            
            # Use torch.where to handle conditional/unconditional embeddings
            label_emb = torch.where(unconditional_mask, self.unconditional_embedding, label_emb)
        else:
            # Pure unconditional embedding
            label_emb = self.unconditional_embedding.expand(x.size(0), -1)

        # Combine time and label embeddings
        emb = time_emb[:, :, None, None] + label_emb[:, :, None, None]

        # print("Embedding Shape Before Expansion:", emb.shape)  # Should be [batch_size, 1, 1, 1]
        
        # print("Embedding Shape Before Expansion:", emb.shape)  # Should be [batch_size, 256, 1, 1]
        # Ensure emb has the correct shape (batch_size, 1, 256, 256) before concatenation
        # emb = emb.expand(-1, 1, x.shape[2], x.shape[3])  # Change -1 to 1 for channels
        emb = emb.permute(0, 2, 3, 1).expand(-1, x.shape[2], x.shape[3], -1).permute(0, 3, 1, 2)

        # Instead of expanding to 256 channels, reduce it to 1 channel
        emb = emb.mean(dim=1, keepdim=True)  # Reduce 256 channels → 1 channel

        # print("Embedding Shape After Expansion:", emb.shape)  # Should be [batch_size, 256, 256, 256]
        
        # Concatenate the embedding with the image
        # print("Input Shape before concat:", x.shape)  # Should be [batch_size, 1, 256, 256]
        x = torch.cat((x, emb), dim=1)  # Should now have exactly 2 channels
        # print("Final Input Shape to Conv1:", x.shape)  # Should be [batch_size, 2, 256, 256]

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return self.conv4(x)

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule for beta values (noise variance), as used in improved DDPM. Cosine schedules tend to result in better quality than linear ones."""
    steps = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = torch.cos((steps / timesteps + s) / (1 + s) * (torch.pi / 2)) ** 2
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)

# Diffusion Model with Classifier-Free Guidance
class DDPM(pl.LightningModule):
    """
    Implements the DDPM (Denoising Diffusion Probabilistic Model) training framework.
    Includes support for classifier-free guidance (unconditional + conditional loss blending).
    """
    def __init__(self, label_dim, learning_rate=learning_rate, timesteps=1000, guidance_scale=guidance_scale, verbose=False):
        super().__init__()
        self.model = DenoiseModel(label_dim)
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        self.verbose = verbose
        self.guidance_scale = guidance_scale

        # Replace this in DDPM initialization
        self.register_buffer('betas', cosine_beta_schedule(timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alpha_hat', torch.cumprod(self.alphas, dim=0))

        # Ensure alpha_hat never hits exactly 0 (avoids NaN division)
        self.alpha_hat = torch.clamp(self.alpha_hat, min=1e-8)

    def forward(self, x, t, labels=None):
        if labels is not None and isinstance(labels, torch.Tensor) and (labels != -1).any():
            noise_pred_cond = self.model(x, t, labels)  # Conditional
            noise_pred_uncond = self.model(x, t, -1)    # Unconditional

            # Fix: Use correct guidance formula
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Fix: Clamp extreme values to prevent numerical instability
            noise_pred = torch.clamp(noise_pred, -5, 5)
        else:
            noise_pred = self.model(x, t, -1)

        return noise_pred

    def training_step(self, batch, batch_idx):
        imgs, labels = batch

        batch_size = imgs.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=imgs.device).long()

        noise = torch.randn_like(imgs, device=imgs.device)
        noisy_imgs = self.q_sample(imgs, t, noise)

        # Randomly drop labels during training (e.g., 10% of the time) for unconditional training
        drop_labels = torch.rand(batch_size, device=imgs.device) < 0.1
        conditional_labels = labels.clone().detach()
        conditional_labels[drop_labels] = -1

        predicted_noise = self.forward(noisy_imgs, t, conditional_labels)
        loss = F.mse_loss(predicted_noise, noise)

        if self.verbose:
            unconditional_ratio = (conditional_labels == -1).float().mean().item()
            print(f"Batch {batch_idx}: Loss = {loss.item():.6f}, Unconditional Ratio = {unconditional_ratio:.2f}")


        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # **Log reconstructions at fixed timesteps for consistent visualization**

        if batch_idx % 500 == 0:
            fixed_timesteps = [100, 300, 500, 700, 900]
            with torch.no_grad():
                grid_size = min(4, batch_size)  # Smaller grid to fit all t
                imgs_for_logging = imgs[:grid_size]
                labels_for_logging = labels[:grid_size]

                self.logger.experiment.add_images("Original Images", (imgs_for_logging + 1) / 2, global_step=self.current_epoch)

                for t_val in fixed_timesteps:
                    t_fixed = torch.full((grid_size,), t_val, device=imgs.device, dtype=torch.long)
                    noise = torch.randn_like(imgs_for_logging)
                    
                    alpha_hat_t = self.alpha_hat[t_val].to(imgs.device)
                    sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
                    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat_t)

                    noisy_imgs = sqrt_alpha_hat * imgs_for_logging + sqrt_one_minus_alpha_hat * noise

                    # Predict with classifier-free guidance
                    noise_pred_cond = self.model(noisy_imgs, t_fixed, labels_for_logging)
                    noise_pred_uncond = self.model(noisy_imgs, t_fixed, -1 * torch.ones_like(labels_for_logging))
                    predicted_noise = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    reconstructed = (noisy_imgs - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
                    reconstructed = torch.clamp(reconstructed, -1, 1)

                    # Normalize for TensorBoard
                    self.logger.experiment.add_images(
                        f"Denoised_t{t_val}", (reconstructed + 1) / 2, global_step=self.current_epoch
            )


        # **Log Noisy and Denoised Images to TensorBoard**
        # if batch_idx % 500 == 0:
        #     # Reconstruct x₀ from predicted noise
        #     alpha_hat_t = self.alpha_hat[t].to(t.device)[:, None, None, None]
        #     sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
        #     sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat_t)

        #     reconstructed = (noisy_imgs - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat

        #     # log them
        #     grid_size = min(8, batch_size)
        #     imgs_to_show = (imgs[:grid_size] + 1) / 2
        #     noisy_to_show = (noisy_imgs[:grid_size] + 1) / 2
        #     denoised_to_show = reconstructed[:grid_size]  # Now truly denoised!

        #     self.logger.experiment.add_images("Original Images", imgs_to_show, global_step=self.current_epoch)
        #     self.logger.experiment.add_images("Noisy Images", noisy_to_show, global_step=self.current_epoch)
        #     self.logger.experiment.add_images("Denoised Images", denoised_to_show, global_step=self.current_epoch)


        return loss

    def q_sample(self, x_start, t, noise):
        """ Diffuse the data (add noise) using the forward process q(x_t | x_0) """
        sqrt_alpha_hat = self.alpha_hat[t].to(t.device)[:, None, None, None]
        sqrt_one_minus_alpha_hat = (1.0 - self.alpha_hat[t].to(t.device))[:, None, None, None]
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
# model = DDPM(label_dim=4, learning_rate=1e-4, timesteps=1000, guidance_scale=5.0, verbose=True)

# Check model device and parameters
# print("Model Parameters:", sum(p.numel() for p in model.parameters()))
# print("Model Device:", next(model.parameters()).device)