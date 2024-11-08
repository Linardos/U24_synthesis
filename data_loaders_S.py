import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the root directory (WSL format)
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

class NiftiDataset(Dataset):
    def __init__(self, full_data_path, transform=None):
        self.full_data_path = full_data_path
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # Define the labels for each class
        for label in [['benign', 0], ['malignant', 1]]:
            class_dir = os.path.join(self.full_data_path, label[0])
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist. Skipping...")
                continue

            # Iterate over each subdirectory within the class directory
            for subdir in os.listdir(class_dir):
                subdir_path = os.path.join(class_dir, subdir)

                # Find any .nii.gz file within the subdir
                nii_files = [f for f in os.listdir(subdir_path) if f.endswith('.nii.gz')]
                if nii_files:
                    file_path = os.path.join(subdir_path, nii_files[0])  # Take the first .nii.gz file found
                    samples.append((file_path, label[1]))
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
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension: [1, H, W, D]

        # Check if the tensor has an extra singleton dimension at the end and remove it
        if img_tensor.shape[-1] == 1:
            img_tensor = img_tensor.squeeze(-1)  # Remove singleton dimension at the end

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label

# Example usage
transform = transforms.Compose([
    # Add any additional transformations here
])

dataset = NiftiDataset(full_data_path=full_data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate through the dataset (for demonstration purposes)
for images, labels in dataloader:
    print("Images shape:", images.shape)
    print("Labels:", labels)
    break  # Remove this line to go through the whole dataset
