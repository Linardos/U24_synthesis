import os
import shutil
import random
import nibabel as nib
import numpy as np
from skimage.transform import resize

# Paths for each dataset
ddsm_train_path = "/mnt/c/Datasets/DDSM/DDSM_clean/train"
cmmd_train_path = "/mnt/c/Datasets/CMMD/CMMD_clean/train"
bcs_dbt_path = "/mnt/c/Datasets/BCS-DBT-Szymon/original_256x256"

# Destination path for sanity dataset
sanity_path = "/mnt/c/Datasets/SanityThree_train"

# Target dimensions for resizing
target_dims = (256, 256)

def resize_and_save_nii(source_path, dest_path):
    """
    Resizes and saves NIfTI images to the target dimensions.
    """
    img = nib.load(source_path)
    data = img.get_fdata()

    # Resize if dimensions don't match the target
    if data.shape[:2] != target_dims:
        resized_data = resize(data, (*target_dims, data.shape[2]), mode='reflect', anti_aliasing=True)
    else:
        resized_data = data

    # Save to destination path
    new_img = nib.Nifti1Image(resized_data, img.affine, img.header)
    nib.save(new_img, dest_path)

def create_sanity_data(source_root, dest_root, sanity_sample_size=50):
    """
    Creates a sanity dataset with a fixed number of samples per class (benign/malignant).
    """
    for class_label in ["benign", "malignant"]:
        class_source = os.path.join(source_root, class_label)
        if not os.path.exists(class_source):
            continue

        # Get all subdirectories for this class
        subdirs = [os.path.join(class_source, d) for d in os.listdir(class_source) if os.path.isdir(os.path.join(class_source, d))]
        
        # Randomly select samples
        selected_samples = random.sample(subdirs, min(sanity_sample_size, len(subdirs)))

        for subdir in selected_samples:
            # Define destination directory
            relative_path = os.path.relpath(subdir, source_root)
            dest_dir = os.path.join(dest_root, relative_path)
            os.makedirs(dest_dir, exist_ok=True)

            # Copy and resize "slice.nii.gz"
            for file in os.listdir(subdir):
                if file == "slice.nii.gz":
                    source_file = os.path.join(subdir, file)
                    dest_file = os.path.join(dest_dir, file)
                    resize_and_save_nii(source_file, dest_file)
                    print(f"Copied and resized {source_file} to {dest_file}")

# Process each dataset
print("Creating sanity dataset from DDSM...")
create_sanity_data(ddsm_train_path, sanity_path, sanity_sample_size=50)

print("Creating sanity dataset from CMMD...")
create_sanity_data(cmmd_train_path, sanity_path, sanity_sample_size=50)

print("Creating sanity dataset from BCS-DBT...")
create_sanity_data(bcs_dbt_path, sanity_path, sanity_sample_size=50)

print("Sanity dataset has been created successfully!")
