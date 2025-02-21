import os
import random
import shutil
from tqdm import tqdm

# Define source and target directories
source_base_dir = "/mnt/d/Datasets/EMBED/EMBED_clean/train/original"
target_base_dir = "/mnt/d/Datasets/EMBED/EMBED_clean/train_downsampled/original"

# Categories and the number of samples to copy
categories = {
    "benign": 16000,
    "probably_benign": 8000,
    "suspicious": 6000,
    "malignant": None  # Copy all
}

for category, target_count in categories.items():
    source_dir = os.path.join(source_base_dir, category)
    target_dir = os.path.join(target_base_dir, category)

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Get list of all subdirectories (assuming each case is in its own folder)
    folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    current_count = len(folders)

    print(f"\nCategory: {category} | Available: {current_count} | Target: {target_count if target_count else 'All'}")

    # Determine how many folders to copy
    if target_count is None or current_count <= target_count:
        folders_to_copy = folders  # Copy all if target_count is None or not enough samples
    else:
        folders_to_copy = random.sample(folders, target_count)

    print(f"Copying {len(folders_to_copy)} folders to '{target_dir}'...")

    # Copy selected folders with tqdm progress bar
    for folder in tqdm(folders_to_copy, desc=f"Copying {category}", unit="folder"):
        src_path = os.path.join(source_dir, folder)
        dest_path = os.path.join(target_dir, folder)
        shutil.copytree(src_path, dest_path)

    print(f"Finished copying {len(folders_to_copy)} cases to '{target_dir}'.\n")

print("All requested categories have been processed successfully. ✅")
