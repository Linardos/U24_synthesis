import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from torchvision import transforms
import pytorch_lightning as pl

# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the root directory
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

class NiftiSynthesisDataset(Dataset):
    def __init__(self, full_data_path, transform=None):
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
        # Convert the image array to float32 and apply transformations
        if self.transform:
            img_tensor = self.transform(img_array)
        else:
            img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # Fallback if no transform is provided

        return img_tensor, torch.tensor(label, dtype=torch.long)

# PyTorch Lightning DataModule
class SynthesisDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        self.dataset = NiftiSynthesisDataset(full_data_path, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

# Example of transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Example of usage
if __name__ == "__main__":
    data_module = SynthesisDataModule(batch_size=8, transform=transform)
    data_module.setup()

    for batch in data_module.train_dataloader():
        imgs, labels = batch
        print(imgs.shape, labels)
        break
