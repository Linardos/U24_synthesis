import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the root directory (WSL format)
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

NUM_CLASSES = config.get('num_classes', 4)  

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



def stratified_real_synth_mix(real_ds, synth_ds, real_fraction, seed=0):
    """
    Returns a ConcatDataset where, for *each* class label, the proportion of
    real vs. synthetic samples approximates `real_fraction`.

    Args
    ----
    real_ds, synth_ds : datasets that yield (image, label)
    real_fraction     : float in [0,1] – desired share of real per class
    seed              : int – controls the random draw

    Notes
    -----
    • If one folder does not have enough samples for a class,
      the function falls back to whatever is available and logs a warning.
    • Sampling is WITHOUT replacement and deterministic under `seed`.
    """
    rng = np.random.RandomState(seed)

    # 1) collect indices by class for each dataset
    real_by_class  = {}
    synth_by_class = {}
    for idx in range(len(real_ds)):
        lbl = real_ds[idx][1]
        real_by_class.setdefault(lbl, []).append(idx)
    for idx in range(len(synth_ds)):
        lbl = synth_ds[idx][1]
        synth_by_class.setdefault(lbl, []).append(idx)

    sampled_real, sampled_synth = [], []

    # 2) iterate over the union of classes
    for lbl in set(real_by_class) | set(synth_by_class):
        real_pool  = real_by_class.get(lbl, [])
        synth_pool = synth_by_class.get(lbl, [])

        total_target   = len(real_pool) + len(synth_pool)          # keep class-size unchanged
        real_target    = int(real_fraction * total_target)
        synth_target   = total_target - real_target

        # clamp to what’s available
        real_take  = min(real_target,  len(real_pool))
        synth_take = min(synth_target, len(synth_pool))

        if real_take < real_target or synth_take < synth_target:
            print(f"[stratified mix]  ⚠  class {lbl}: insufficient samples "
                  f"(request R={real_target},S={synth_target} | "
                  f"have R={len(real_pool)},S={len(synth_pool)})")

        sampled_real.extend(rng.choice(real_pool,  size=real_take,  replace=False))
        sampled_synth.extend(rng.choice(synth_pool, size=synth_take, replace=False))

    # 3) wrap them into Subset + ConcatDataset
    return ConcatDataset([
        Subset(real_ds,  sampled_real),
        Subset(synth_ds, sampled_synth)
    ])
