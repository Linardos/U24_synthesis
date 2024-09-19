import os
import nibabel as nib
import yaml

# Load configuration
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

root_dir = config['root_dir']

# Define paths to a VQ-VAE and original slice
vqvae_data_dir = 'vqvae'
original_data_dir = 'original_256x256'

# Combine root directory with data directories
vqvae_full_data_path = os.path.join(root_dir, vqvae_data_dir)
original_full_data_path = os.path.join(root_dir, original_data_dir)

# Function to find a .nii.gz file in a directory
def find_nifti_file(data_path):
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.nii.gz'):
                return os.path.join(root, file)
    return None

# Check if there's a slice in the respective directories
vqvae_slice_path = find_nifti_file(vqvae_full_data_path)
original_slice_path = find_nifti_file(original_full_data_path)

# Function to check NIfTI dimensions
def check_nifti_dimensions(slice_path):
    # Load the NIfTI file
    nifti_img = nib.load(slice_path)
    img_array = nifti_img.get_fdata()

    # Print the dimensions
    print(f"Slice Path: {slice_path}")
    print(f"Dimensions: {img_array.shape}")
    print(f"Number of Dimensions: {img_array.ndim}")
    print('-' * 40)

# Check and display results for both VQ-VAE and Original dataset
if vqvae_slice_path:
    check_nifti_dimensions(vqvae_slice_path)
else:
    print(f"No .nii.gz file found in {vqvae_full_data_path}")

if original_slice_path:
    check_nifti_dimensions(original_slice_path)
else:
    print(f"No .nii.gz file found in {original_full_data_path}")
