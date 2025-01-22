import os
import shutil
from tqdm import tqdm

# Define paths
source_dir = "/mnt/d/Datasets/EMBED/EMBED_clean/train/original"
dest_dir = "/mnt/d/Datasets/EMBED/sanity_EMBED"

# Categories to process
categories = ["benign", "malignant", "probably_benign", "suspicious"]

# Number of directories to copy from each category
num_folders = 100

# Ensure destination directories exist
for category in categories:
    os.makedirs(os.path.join(dest_dir, category), exist_ok=True)

# Function to copy the first `num_folders` directories from source to destination
def copy_first_folders(source_folder, dest_folder, num_folders):
    # Get all entries in the source folder and filter to only directories
    folders = sorted([f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))])

    # Debug: Print the number of directories found in the source directory
    print(f"Found {len(folders)} directories in '{source_folder}'.")

    # Select the first `num_folders` directories (or fewer if not enough exist)
    selected_folders = folders[:num_folders]

    # Copy each folder to the destination directory with a progress bar
    for folder in tqdm(selected_folders, desc=f"Copying directories from {source_folder}"):
        shutil.copytree(os.path.join(source_folder, folder), os.path.join(dest_folder, folder))

    print(f"Copied {len(selected_folders)} directories from '{source_folder}' to '{dest_folder}'")

# Copy directories for each category
for category in categories:
    source_path = os.path.join(source_dir, category)
    dest_path = os.path.join(dest_dir, category)
    copy_first_folders(source_path, dest_path, num_folders)

print("Subset copying complete.")
