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

# ── DataModule with adjustable benign:malignant ratio ───────────────────────
class BalancedSamplingDataModule(pl.LightningDataModule):
    """
    Each epoch returns  (ratio × N_minority)  samples of the majority class
    plus all N_minority samples of the minority class.

    Works no matter which label has more files.

    Parameters
    ----------
    full_data_path : str
        root/ └─ benign/…  malignant/…
    ratio : float
        1.0 → perfectly balanced (1 : 1)  
        2.0 → majority twice as frequent as minority (1 : 2)
    """

    def __init__(
        self,
        full_data_path: str,
        batch_size: int,
        transform=None,
        num_workers: int = 8,
        ratio: float = 1.0,
    ):
        super().__init__()
        self.full_data_path = Path(full_data_path)
        self.batch_size     = batch_size
        self.transform      = transform
        self.num_workers    = num_workers
        self.ratio          = ratio

        # will be filled in setup()
        self.buckets = {}            # {"benign": [(path,label), …], "malignant":[…]}
        self.minority_key  = None
        self.majority_key  = None

    # ---------------------------------------------------------
    #  SETUP
    # ---------------------------------------------------------
    def setup(self, stage=None):
        # 1. scan folders into buckets
        for cls, label in [("benign", 0), ("malignant", 1)]:
            cls_dir = self.full_data_path / cls
            if not cls_dir.exists():
                continue
            paths = [
                (str(f), label)
                for subj in cls_dir.iterdir()
                for f in subj.glob("*.nii.gz")
            ]
            self.buckets[cls] = paths

        if len(self.buckets) < 2:
            raise RuntimeError("Need both benign and malignant folders present.")

        # 2. decide minority / majority
        if len(self.buckets["benign"]) <= len(self.buckets["malignant"]):
            self.minority_key  = "benign"
            self.majority_key  = "malignant"
        else:
            self.minority_key  = "malignant"
            self.majority_key  = "benign"

        minority_n  = len(self.buckets[self.minority_key])
        majority_n  = len(self.buckets[self.majority_key])
        needed      = int(self.ratio * minority_n)

        if majority_n < needed:
            raise RuntimeError(
                f"ratio={self.ratio} needs {needed} {self.majority_key} images "
                f"but only {majority_n} available."
            )

        print(
            f"BalancedSampling - minority={self.minority_key}({minority_n})  "
            f"majority={self.majority_key}({majority_n})  "
            f"→ each epoch uses {minority_n + needed} samples "
            f"({self.ratio} : 1 balance)."
        )

    # ---------------------------------------------------------
    #  make a fresh dataset each epoch
    # ---------------------------------------------------------
    def _make_dataset(self):
        minority   = self.buckets[self.minority_key]
        majority   = self.buckets[self.majority_key]
        need_major = int(self.ratio * len(minority))
        majority_subset = random.sample(majority, k=need_major)

        samples = majority_subset + minority
        random.shuffle(samples)
        return BasicSynthesisDataset(samples, transform=self.transform)

    def train_dataloader(self):
        ds = self._make_dataset()
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

# class BalancedSamplingDataModule(pl.LightningDataModule):
#     """
#     Draws <ratio>× as many benign as malignant each epoch.

#     • Loads *all* malignant and benign paths once in `setup`.
#     • At every call to `train_dataloader`, samples a fresh benign subset
#       of size  ratio * len(malignant)  (no replacement), merges, shuffles,
#       and returns a DataLoader.
#     • Set Trainer(reload_dataloaders_every_n_epochs=1).
#     """

#     def __init__(
#         self,
#         full_data_path: str,
#         batch_size: int,
#         transform=None,
#         num_workers: int = 8,
#         ratio: float = 1.0,                # 1.0 → 1 : 1, 2.0 → 2 : 1 …
#     ):
#         super().__init__()
#         self.full_data_path = Path(full_data_path)
#         self.batch_size     = batch_size
#         self.transform      = transform
#         self.num_workers    = num_workers
#         self.ratio          = ratio

#         self.benign_pool       = []
#         self.malignant_samples = []

#     # called once per rank
#     def setup(self, stage=None):
#         buckets = defaultdict(list)
#         for cls in ["benign", "malignant"]:
#             cls_dir = self.full_data_path / cls
#             if not cls_dir.exists(): continue
#             label = 0 if cls == "benign" else 1
#             for sub in cls_dir.iterdir():
#                 nii_files = [f for f in sub.iterdir() if f.name.endswith(".nii.gz")]
#                 if nii_files:
#                     buckets[cls].append((str(nii_files[0]), label))

#         self.malignant_samples = buckets["malignant"]
#         self.benign_pool       = buckets["benign"]

#         if not self.malignant_samples:
#             raise RuntimeError("No malignant samples found!")
#         need = int(self.ratio * len(self.malignant_samples))
#         if len(self.benign_pool) < need:
#             raise RuntimeError(
#                 f"Ratio {self.ratio} requires {need} benign images, "
#                 f"but pool has only {len(self.benign_pool)}."
#             )

#     # helper per epoch
#     def _make_dataset(self):
#         need = int(self.ratio * len(self.malignant_samples))
#         benign_subset = random.sample(self.benign_pool, k=need)
#         samples = benign_subset + self.malignant_samples
#         random.shuffle(samples)
#         return BasicSynthesisDataset(samples, transform=self.transform)

#     def train_dataloader(self):
#         ds = self._make_dataset()
#         return DataLoader(
#             ds,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             persistent_workers=False,
#         )

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
