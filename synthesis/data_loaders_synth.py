import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

# Load configuration
with open('config_synth.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the root directory (WSL format)
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

class NiftiSynthesisDataset(Dataset):
    def __init__(self, full_data_path, transform=None):
        """
        Dataset for label-conditioned synthesis tasks.
        
        Args:
            full_data_path (str): Path to the dataset.
            transform (callable, optional): Transformations to apply.
        """
        self.full_data_path = full_data_path
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        class_labels = {
            'benign': 0,
            'malignant': 1,
            'probably_benign': 2,
            'suspicious': 3
        }

        for class_name, label in class_labels.items():
            class_dir = os.path.join(self.full_data_path, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist. Skipping...")
                continue

            for subdir in os.listdir(class_dir):
                subdir_path = os.path.join(class_dir, subdir)
                nii_files = [f for f in os.listdir(subdir_path) if f.endswith('.nii.gz')]
                if nii_files:
                    file_path = os.path.join(subdir_path, nii_files[0])
                    samples.append((file_path, label))
                else:
                    print(f"No .nii.gz file found in {subdir_path}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        nifti_img = nib.load(file_path)
        img_array = nifti_img.get_fdata()
        img_tensor = torch.tensor(img_array, dtype=torch.float32)  # Convert to tensor
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension: [1, H, W, D] if 3D

        # If it's a 3D image and has an extra singleton dimension, remove it
        if img_tensor.shape[-1] == 1:
            img_tensor = img_tensor.squeeze(-1)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.tensor(label, dtype=torch.long)  # Label for conditioning

# Function to create a DataLoader
def get_synthesis_dataloader(batch_size=8, shuffle=True, transform=None):
    dataset = NiftiSynthesisDataset(full_data_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
