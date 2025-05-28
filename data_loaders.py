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

class NiftiDataset(Dataset):
    def __init__(self, full_data_path, transform=None):
        self.full_data_path = full_data_path
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # Define the labels for each class
        class_labels = {
            'benign': 0,
            'malignant': 1,
            'suspicious': 2,
            # 'probably_benign': 3,
        }

        if NUM_CLASSES == 4:
            class_labels['probably_benign'] = 3

        for class_name, label in class_labels.items():
            class_dir = os.path.join(self.full_data_path, class_name)
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
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension: [1, H, W, D]

        # Check if the tensor has an extra singleton dimension at the end and remove it
        if img_tensor.shape[-1] == 1:
            img_tensor = img_tensor.squeeze(-1)  # Remove singleton dimension at the end

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


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
