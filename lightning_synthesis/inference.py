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
batch_size = config.get('batch_size', 1)  # Use small batch size for inference
label_dim = config.get('label_dim', 4)
experiment_name = config.get('experiment_name', 'default_experiment')
model_dir = config.get('model_dir', 'saved_models')

# Specify the checkpoint path to load the trained model
checkpoint_path = os.path.join(model_dir, 'ddpm-epoch=12-train_loss=0.0965.ckpt')

# Set up the same transform as used during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)), 
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.to(torch.float32))
])

# Set up data module for inference
data_module = SynthesisDataModule(batch_size=batch_size, transform=transform)
data_module.setup()

# Load the trained model from checkpoint
model = DDPM.load_from_checkpoint(
    checkpoint_path,
    label_dim=label_dim,
    learning_rate=0  # learning_rate is not used during inference
)
model.eval()  # Set model to evaluation mode

# Move model to the correct device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Create a directory to save inference results
inference_dir = os.path.join('inference_results', experiment_name)
os.makedirs(inference_dir, exist_ok=True)

# Determine inference parameters
num_images_to_generate = 10  # Number of images to generate
num_timesteps = model.timesteps  # Number of denoising steps


def sample_ddpm(model, num_samples, num_timesteps, label=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size = (num_samples, 1, 256, 256)  # Assuming grayscale 256x256 images
    noisy_img = torch.randn(img_size, device=device)

    if label is not None:
        label_tensor = torch.full((num_samples,), label, dtype=torch.long, device=device)
    else:
        label_tensor = None

    for t in reversed(range(num_timesteps)):
        timestep_tensor = torch.tensor([t] * num_samples, dtype=torch.long, device=device)
        predicted_noise = model(noisy_img, timestep_tensor, label_tensor)

        alpha_t = model.alpha_hat[t].to(device)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        # Fix: Add correct noise scaling
        if t > 0:
            sigma_t = torch.sqrt((1 - alpha_t) / (1 - model.alpha_hat[t-1]))
            noise = torch.randn_like(noisy_img, device=device) * sigma_t
            noisy_img = (1 / sqrt_alpha_t) * (noisy_img - sqrt_one_minus_alpha_t * predicted_noise) + noise
        else:
            noisy_img = (1 / sqrt_alpha_t) * (noisy_img - sqrt_one_minus_alpha_t * predicted_noise)

    return torch.clamp(noisy_img, -1, 1)  # Ensure final values are in range



# Generate images
generated_images = sample_ddpm(model, num_images_to_generate, num_timesteps)
# Fix: Apply the exact same de-normalization at inference time before saving
def denormalize(img):
    return (img + 1) / 2  # Maps from [-1,1] to [0,1]


# Convert generated images from [-1,1] to [0,1] for saving
generated_images = (generated_images + 1) / 2
print("Generated Images Tensor Shape:", generated_images.shape)
print("Min:", generated_images.min().item(), "Max:", generated_images.max().item())
# Ensure all saved images are correctly scaled
for i, img in enumerate(generated_images):
    save_image(denormalize(img), f"inference_results/generated_image_{i+1}.png")
# Save generated images
# for i in range(num_images_to_generate):
#     img_path = os.path.join(inference_dir, f"generated_image_{i+1}.png")
#     save_image(generated_images[i], img_path)
#     print(f"Saved: {img_path}")

print("Generation complete! Images saved in:", inference_dir)
