import os, random
import yaml
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from monai import transforms as mt
# Load configuration
with open('config_l.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the root directory
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

# ── NEW: LightningDataModule that re-samples benign each epoch ───────────────
class BasicSynthesisDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        sample = {"image": file_path, "class": label}
        if self.transform:
            sample = self.transform(sample)
        return sample["image"], sample["class"]

class BalancedSamplingDataModule(pl.LightningDataModule):
    """
    • Loads *all* malignant (.nii.gz) paths once.
    • Keeps a pool of *all* benign paths.
    • On every call to `train_dataloader` it draws a fresh benign subset the
      same size as malignant, merges, shuffles, and returns a DataLoader.

    Set `reload_dataloaders_every_n_epochs=1` in the Trainer so Lightning
    will call `train_dataloader()` at the start of every epoch.
    """

    def __init__(
        self,
        full_data_path: str,
        batch_size: int,
        transform=None,
        num_workers: int = 8,
    ):
        super().__init__()
        self.full_data_path = Path(full_data_path)
        self.batch_size     = batch_size
        self.transform      = transform
        self.num_workers    = num_workers

        self.benign_pool       = []  # list[(path, 0)]
        self.malignant_samples = []  # list[(path, 1)]

    # ── called once on every rank before training ────────────────────────────
    def setup(self, stage):
        all_samples_by_label = defaultdict(list)
        for class_name in ["benign", "malignant"]:
            class_dir = self.full_data_path / class_name
            if not class_dir.exists():
                continue
            label = 0 if class_name == "benign" else 1
            for subdir in class_dir.iterdir():
                nii_files = [f for f in subdir.iterdir() if f.suffix == ".gz"]
                if nii_files:
                    all_samples_by_label[class_name].append((str(nii_files[0]), label))

        self.malignant_samples = all_samples_by_label["malignant"]
        self.benign_pool       = all_samples_by_label["benign"]

        if len(self.malignant_samples) == 0:
            raise RuntimeError("No malignant samples found!")
        if len(self.benign_pool) < len(self.malignant_samples):
            raise RuntimeError("Benign pool smaller than malignant pool!")

    # ── helper that re-samples benign & builds a dataset ─────────────────────
    def _make_dataset(self):
        benign_subset = random.sample(
            self.benign_pool, k=len(self.malignant_samples)
        )
        balanced_samples = benign_subset + self.malignant_samples
        random.shuffle(balanced_samples)
        return BasicSynthesisDataset(balanced_samples, transform=self.transform)

    # ── Lightning hooks ──────────────────────────────────────────────────────
    def train_dataloader(self):
        dataset = self._make_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,               # already shuffled, but this mixes batches
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,   # new DataLoader each epoch → keep False
        )

# Original one used:
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

        # label order per setting binary, 3-class, 4-class
        label_sets = {
            2: ["benign", "malignant"],
            3: ["benign", "malignant", "suspicious"],
            4: ["benign", "malignant", "suspicious", "probably_benign"],
        }

        labels = label_sets.get(config["num_classes"])
        if labels is None:
            raise ValueError("num_classes must be 2, 3, or 4")

        samples_by_label = {lbl: []      for lbl in labels}
        class_labels     = {lbl: idx     for idx, lbl in enumerate(labels)}
        # ------


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
        if self.samples_per_class:
            fixed_samples = []
            for class_name, sample_list in samples_by_label.items():
                if len(sample_list) < self.samples_per_class:
                    print(f"Not enough samples for class '{class_name}'. "
                                     f"Requested {self.samples_per_class}, but found {len(sample_list)}, which is the sample number to be used")
                    sample_list = sorted(sample_list)[:len(sample_list)]
                    fixed_samples.extend(sample_list)
                else:
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
