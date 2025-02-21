import os
import random
import shutil

# Define paths
benign_dir = "/mnt/d/Datasets/EMBED/EMBED_clean/train/original/benign"
target_count = 50000  # Desired number of benign samples

# Get list of all subdirectories (assuming each case is in its own folder)
benign_folders = [f for f in os.listdir(benign_dir) if os.path.isdir(os.path.join(benign_dir, f))]

# Check current count
current_count = len(benign_folders)
print(f"Current benign cases: {current_count}")

if current_count > target_count:
    # Randomly select folders to keep
    folders_to_keep = random.sample(benign_folders, target_count)

    # Find folders to delete
    folders_to_delete = set(benign_folders) - set(folders_to_keep)

    print(f"Deleting {len(folders_to_delete)} folders...")

    # Delete extra folders
    for folder in folders_to_delete:
        folder_path = os.path.join(benign_dir, folder)
        shutil.rmtree(folder_path)
        print(f"Removed: {folder_path}")

    print(f"Benign cases reduced to {target_count}")
else:
    print(f"No folders removed. Current count ({current_count}) is already below the target.")
