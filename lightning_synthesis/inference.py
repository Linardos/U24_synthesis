import os
import yaml
import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchvision.utils import save_image

from data_loaders_l import SynthesisDataModule
from models_l import CVAE

# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract parameters from config
batch_size = config.get('batch_size', 1)  # use a small batch size for inference
latent_dim = config.get('latent_dim', 100)
label_dim = config.get('label_dim', 4)
experiment_name = config.get('experiment_name', 'default_experiment')
model_dir = config.get('model_dir', 'saved_models')

# Specify the checkpoint path to load the trained model from.
# Update the filename below to match your saved checkpoint.
checkpoint_path = os.path.join(model_dir, 'cvae-best.ckpt')

# Set up the same transform as used during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # Rescale to [0,1]
    transforms.Normalize((0.5,), (0.5,)),  # Normalize if desired (this will shift [0,1] to approx. [-1,1])
    transforms.Lambda(lambda x: x.to(torch.float32))
])

# Set up data module for inference
data_module = SynthesisDataModule(batch_size=batch_size, transform=transform)
data_module.setup()

# Load the trained model from checkpoint
model = CVAE.load_from_checkpoint(
    checkpoint_path,
    latent_dim=latent_dim,
    label_dim=label_dim,
    learning_rate=0  # learning_rate is not used during inference
)
model.eval()  # Set model to evaluation mode

# Create a directory to save inference results
inference_dir = os.path.join('inference_results', experiment_name)
os.makedirs(inference_dir, exist_ok=True)

# Choose which dataloader to use for inference. 
# If you don't have a separate test set, you can use the train or validation dataloader.
dataloader = data_module.test_dataloader()

# Run inference and save results
with torch.no_grad():
    for i, (imgs, labels) in enumerate(dataloader):
        # Generate reconstruction using the trained model
        reconstructed, mu, logvar, z = model(imgs, labels)
        
        # Save the original input image and the reconstructed image for comparison
        save_image(imgs, os.path.join(inference_dir, f"input_{i}.png"))
        save_image(reconstructed, os.path.join(inference_dir, f"reconstructed_{i}.png"))
        
        print(f"Saved inference results for batch {i}")

print("Inference complete! Check the directory:", inference_dir)
