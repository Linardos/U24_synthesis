import os
import shutil
from tqdm import tqdm

# Define paths
source_dir = "/home/locolinux2/datasets/BCS-DBT-Szymon/original_256x256"
dest_dir = "/home/locolinux2/datasets/BCS-DBT-Szymon/original_sanity_subset"

# Ensure destination directories exist
os.makedirs(os.path.join(dest_dir, "abnormal"), exist_ok=True)
os.makedirs(os.path.join(dest_dir, "healthy"), exist_ok=True)

# Function to copy the first 50 folders in each category
def copy_first_folders(source_folder, dest_folder, num_folders=50):
    # Get all entries in the source folder and filter to only directories
    folders = sorted([f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))])
    
    # Debug: Print the number of folders found in the source directory
    print(f"Found {len(folders)} folders in '{source_folder}'.")

    # Select the first specified number of folders (or fewer if there aren't enough)
    selected_folders = folders[:num_folders]
    
    # Copy each folder to the destination directory with a progress bar
    for folder in tqdm(selected_folders, desc=f"Copying folders from {source_folder}"):
        shutil.copytree(os.path.join(source_folder, folder), os.path.join(dest_folder, folder))

    print(f"Copied {len(selected_folders)} folders from '{source_folder}' to '{dest_folder}'")

# Copy folders from 'abnormal' and 'healthy' categories
copy_first_folders(os.path.join(source_dir, "abnormal"), os.path.join(dest_dir, "abnormal"))
copy_first_folders(os.path.join(source_dir, "healthy"), os.path.join(dest_dir, "healthy"))

print("Folder copying complete.")
