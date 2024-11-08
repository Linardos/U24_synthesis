import os
import shutil
import nibabel as nib
import numpy as np
from skimage.transform import resize

# Paths for each dataset
ddsm_train_path = "/mnt/c/Datasets/DDSM/DDSM_clean/train"
cmmd_train_path = "/mnt/c/Datasets/CMMD/CMMD_clean/train"
bcs_dbt_path = "/mnt/c/Datasets/BCS-DBT-Szymon/original_256x256"

# Destination path
destination_path = "/mnt/c/Datasets/ThreeDataDM_train"

# Target dimensions
target_dims = (256, 256)

def resize_and_save_nii(source_path, dest_path):
    img = nib.load(source_path)
    data = img.get_fdata()

    # Check if resizing is needed
    if data.shape[:2] != target_dims:
        resized_data = resize(data, (*target_dims, data.shape[2]), mode='reflect', anti_aliasing=True)
    else:
        resized_data = data

    # Save to destination path
    new_img = nib.Nifti1Image(resized_data, img.affine, img.header)
    nib.save(new_img, dest_path)

def copy_and_resize_data(source_root, dest_root):
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file == "slice.nii.gz":
                # Define source and destination paths
                source_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_root)
                dest_dir = os.path.join(dest_root, relative_path)
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = os.path.join(dest_dir, file)

                # Resize and save
                resize_and_save_nii(source_file, dest_file)
                print(f"Copied and resized {source_file} to {dest_file}")

# Copy and resize for each dataset
print("Processing DDSM...")
copy_and_resize_data(ddsm_train_path, destination_path)
print("Processing CMMD...")
copy_and_resize_data(cmmd_train_path, destination_path)
print("Processing BCS-DBT...")
copy_and_resize_data(bcs_dbt_path, destination_path)

print("All datasets have been processed and copied to the destination folder.")
