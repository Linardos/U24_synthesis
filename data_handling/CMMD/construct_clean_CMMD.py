import os
import pandas as pd
import nibabel as nib
import pydicom
import numpy as np
from pydicom import dcmread
from nibabel import Nifti1Image

# Define the DICOM-to-NIfTI conversion function
def load_dicom_and_save_as_nifti(dicom_file_path):
    dicom_data = dcmread(dicom_file_path)
    pixel_array = dicom_data.pixel_array  # Extract pixel array
    volume = np.expand_dims(pixel_array, axis=-1)  # Expand dims for single-slice
    
    # Create a NIfTI image with a simple affine matrix
    affine = np.eye(4)
    nifti_image = Nifti1Image(volume, affine)
    return nifti_image

# Define paths
source_folder = '/mnt/c/Datasets/CMMD/'
destination_folder = os.path.join(source_folder, 'CMMD_clean')
os.makedirs(destination_folder, exist_ok=True)

# Load the combined metadata file
metadata_df = pd.read_csv(os.path.join(source_folder, 'combined_metadata.csv'))

# Iterate over each row in the metadata DataFrame
for _, row in metadata_df.iterrows():
    # Normalize and join path
    raw_file_location = row['File Location'].replace('\\', '/')
    dicom_folder_path = os.path.join(source_folder, raw_file_location.lstrip('./'))
    
    subject_id = row['Subject ID']
    classification = row['classification'].lower()  # e.g., 'benign'
    
    # Get and sort the list of DICOM files in the folder
    dicom_files = sorted([f for f in os.listdir(dicom_folder_path) if f.endswith('.dcm')])

    # Create the destination directory based on classification and subject ID
    for slice_index, dicom_file in enumerate(dicom_files, start=1):
        new_folder_name = f"{subject_id}-{slice_index}"
        new_folder_path = os.path.join(destination_folder, classification, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # Define new file path for the .nii.gz file
        new_file_path = os.path.join(new_folder_path, 'slice.nii.gz')
        
        # Convert the DICOM to NIfTI and save
        dicom_file_path = os.path.join(dicom_folder_path, dicom_file)
        try:
            nifti_image = load_dicom_and_save_as_nifti(dicom_file_path)
            nib.save(nifti_image, new_file_path)
        except Exception as e:
            print(f"Failed to convert or move file for {subject_id} - Slice {slice_index}: {e}")
        
        print(f"Processed {subject_id} - Slice {slice_index} to {new_file_path}")
