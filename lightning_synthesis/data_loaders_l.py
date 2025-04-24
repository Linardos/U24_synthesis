import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from torchvision import transforms, datasets
import pytorch_lightning as pl

# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the root directory
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

data_mode = config.get('data_mode', 'embed')  # 'embed', 'mnist', 'cifar'
print(f"Loading data: {data_mode}")

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
        file_path, label = self.samples[idx] # path string!

        sample = {"image": file_path, "class": label}
        if self.transform:
            sample = self.transform(sample)

        # after ToTensord, sample["image"] is a (1,64,64) tensor
        return sample["image"], torch.tensor(sample["class"], dtype=torch.long)

    # def __getitem__(self, idx):
    #     file_path, label = self.samples[idx]
    #     nifti_img = nib.load(file_path)
    #     img_array = nifti_img.get_fdata()
    #     # Convert the image array to float32 and apply transformations
    #     if self.transform:
    #         img_tensor = self.transform(img_array)
    #     else:
    #         img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # Fallback if no transform is provided

    #     return {"image":img_tensor, "class":torch.tensor(label, dtype=torch.long)}


# PyTorch Lightning DataModule
# class SynthesisDataModule(pl.LightningDataModule):
#     def __init__(self, batch_size=8, transform=None, samples_per_class=None):
#         super().__init__()
#         self.batch_size = batch_size
#         self.transform = transform
#         self.samples_per_class = samples_per_class

#     # def setup(self, stage=None):
#     #     self.dataset = NiftiSynthesisDataset(full_data_path, transform=self.transform, 
#     #                                           samples_per_class=self.samples_per_class)
#     def setup(self, stage=None):
#         self.dataset = NiftiSynthesisDataset(full_data_path, transform=self.transform, 
#                                                 samples_per_class=self.samples_per_class)

#     def train_dataloader(self):
#         return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

#     def test_dataloader(self):
#         return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
class SynthesisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        full_path,
        batch_size=8,
        transform=None,
        samples_per_class=None,
        num_workers=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transform = transform
        self.samples_per_class = samples_per_class

    def prepare_data(self) -> None:
        full_path = self.hparams.full_path
        if not os.path.isdir(full_path):
            raise FileNotFoundError(full_path)

    def setup(self, stage) -> None:
        # Build splits once, or whenever stage=="fit" (Lightning calls twice)
        if stage in ("fit", None):
            full_path = self.hparams.full_path
            full_ds = NiftiSynthesisDataset(
                full_path,
                transform=self.transform,
                samples_per_class=self.samples_per_class,
            )
            train_sz = int(0.9 * len(full_ds))
            val_sz   = len(full_ds) - train_sz
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                full_ds, [train_sz, val_sz],
                generator=torch.Generator().manual_seed(42),
            )

    # dataloaders ----------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

