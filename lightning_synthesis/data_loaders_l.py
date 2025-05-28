import os
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import pytorch_lightning as pl

# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the root directory
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

class NiftiSynthesisDataset(Dataset):
    def __init__(self, full_data_path, num_classes, transform=None, samples_per_class=None):
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
        self.num_classes = num_classes
        self.samples = self._load_samples()

    def _load_samples(self):

        samples_by_label = {
            'benign': [],
            'malignant': [],
            'suspicious': []
        }
        class_labels = {
            'benign': 0,
            'malignant': 1,
            'suspicious': 2
        }

        if self.num_classes == 4:
            class_labels['probably_benign'] = 3
            samples_by_label['probably_benign'] = []

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
        file_path, label = self.samples[idx] # path string!

        sample = {"image": file_path, "class": label}
        if self.transform:
            sample = self.transform(sample)

        # after ToTensord, sample["image"] is a (1,64,64) tensor
        return sample["image"], sample["class"] #.clone().detach().to(torch.long) #torch.tensor(sample["class"], dtype=torch.long)
