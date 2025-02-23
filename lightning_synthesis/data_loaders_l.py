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
    def __init__(self, full_data_path, transform=None, samples_per_class=None):
        """
        Args:
            full_data_path (str): Path to the root directory of your data.
            transform (callable, optional): Transform to be applied to each image.
            samples_per_class (int, optional): Fixed number of samples to use from each label.
                                               If provided, each class must have at least this many samples.
        """
        self.full_data_path = full_data_path
        self.transform = transform
        self.samples_per_class = samples_per_class
        self.samples = self._load_samples()

    def _load_samples(self):
        samples_by_label = {
            'benign': [],
            'malignant': [],
            'probably_benign': [],
            'suspicious': []
        }
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
                    samples_by_label[class_name].append((file_path, label))
                else:
                    print(f"No .nii.gz file found in {subdir_path}")

        # If the user requested a fixed number per class, enforce that
        if self.samples_per_class is not None:
            fixed_samples = []
            for class_name, sample_list in samples_by_label.items():
                if len(sample_list) < self.samples_per_class:
                    raise ValueError(f"Not enough samples for class '{class_name}'. "
                                     f"Required {self.samples_per_class}, but found {len(sample_list)}.")
                # For determinism, we sort and take the first N.
                # (Alternatively, you can randomize with a fixed seed.)
                sample_list = sorted(sample_list)[:self.samples_per_class]
                fixed_samples.extend(sample_list)
            return fixed_samples
        else:
            # Otherwise, return all samples from all classes.
            all_samples = []
            for sample_list in samples_by_label.values():
                all_samples.extend(sample_list)
            return all_samples

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
    def __init__(self, batch_size=8, transform=None, samples_per_class=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.samples_per_class = samples_per_class

    def setup(self, stage=None):
        self.dataset = NiftiSynthesisDataset(full_data_path, transform=self.transform, 
                                              samples_per_class=self.samples_per_class)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
