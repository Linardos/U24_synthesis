import os
import random
import shutil
from tqdm import tqdm

# Because we filtered out implants and other artifacts, another round of matching is warranted.

def matching_3221(source_base_dir, target_base_dir, malignant_num):
    categories = {
        "benign":          3 * malignant_num,  # 3:2:2:1 ratio
        "probably_benign": 2 * malignant_num,
        "suspicious":      2 * malignant_num,
        "malignant":       None  # Copy all
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



# Define source and target directories
EMBED_downsampled = "EMBED_clean_256x256"

source_base_dir = f"/mnt/d/Datasets/EMBED/{EMBED_downsampled}/train/original"
target_base_dir = f"/mnt/d/Datasets/EMBED/{EMBED_downsampled}/train_3221/original"
malignant_num = 1148 # for train

matching_3221(source_base_dir, target_base_dir, malignant_num)

source_base_dir = f"/mnt/d/Datasets/EMBED/{EMBED_downsampled}/test"
target_base_dir = f"/mnt/d/Datasets/EMBED/{EMBED_downsampled}/test_3221"
malignant_num = 324

print("All requested categories have been processed successfully. ✅")

