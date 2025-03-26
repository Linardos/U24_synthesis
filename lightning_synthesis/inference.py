import os
import yaml
import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchvision.utils import save_image

from data_loaders_l import SynthesisDataModule
from model_architectures import DDPM

# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract parameters from config
batch_size = config.get('batch_size', 1)
label_dim = config.get('label_dim', 4)
experiment_name = config.get('experiment_name', 'default_experiment')
model_dir = config.get('model_dir', 'saved_models')

# Specify checkpoint
checkpoint_path = os.path.join(model_dir, 'ddpm-epoch=12-train_loss=0.0965.ckpt')

# Set up transform (same as training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.to(torch.float32))
])

# Set up data module for inference
data_module = SynthesisDataModule(batch_size=batch_size, transform=transform)
data_module.setup()

# Load trained model
model = DDPM.load_from_checkpoint(
    checkpoint_path,
    label_dim=label_dim,
    learning_rate=0  # Not used at inference
)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Create output dir
inference_dir = os.path.join('inference_results', experiment_name)
os.makedirs(inference_dir, exist_ok=True)

# Parameters
num_images_to_generate = 10
num_timesteps = model.timesteps


def sample_ddpm(model, num_samples, num_timesteps, label=None, guidance_scale=None):
    device = next(model.parameters()).device
    img_size = (num_samples, 1, 256, 256)
    noisy_img = torch.randn(img_size, device=device)

    if label is not None:
        label_tensor = torch.full((num_samples,), label, dtype=torch.long, device=device)
    else:
        label_tensor = None

    for t in reversed(range(num_timesteps)):
        timestep_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)

        # Optional classifier-free guidance block (only if trained with it)
        if guidance_scale is not None and label_tensor is not None:
            noise_pred_cond = model(noisy_img, timestep_tensor, label_tensor)
            noise_pred_uncond = model(noisy_img, timestep_tensor, None)
            predicted_noise = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            predicted_noise = model(noisy_img, timestep_tensor, label_tensor)

        alpha_t = model.alphas[t].to(device)
        beta_t = model.betas[t].to(device)
        alpha_hat_t = model.alpha_hat[t].to(device)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat_t)

        # Predict x0
        x0_pred = (noisy_img - sqrt_one_minus_alpha_hat * predicted_noise) / torch.sqrt(alpha_hat_t)

        if t > 0:
            noise = torch.randn_like(noisy_img, device=device)
            noisy_img = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(1 - alpha_t) * noise
        else:
            noisy_img = x0_pred  # Final step, no noise added

    return torch.clamp(noisy_img, -1, 1)


# Generate and save
generated_images = sample_ddpm(model, num_images_to_generate, num_timesteps)

# Rescale from [-1, 1] to [0, 1]
def denormalize(img):
    return (img + 1) / 2

generated_images = denormalize(generated_images)

print("Generated Images Tensor Shape:", generated_images.shape)
print("Min:", generated_images.min().item(), "Max:", generated_images.max().item())

for i, img in enumerate(generated_images):
    save_path = os.path.join(inference_dir, f"generated_image_{i+1}.png")
    save_image(img, save_path)
    print(f"Saved: {save_path}")

print("Generation complete! Images saved in:", inference_dir)
