import os
import math
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

# Paths
source_folder = '/mnt/c/Datasets/inbreast/ALL-IMGS'
metadata_file = '/mnt/c/Datasets/inbreast/INbreast.xls'
destination_folder = '/mnt/c/Datasets/inbreast/inbreast_clean'

# Create destination directories
os.makedirs(os.path.join(destination_folder, 'benign'), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'malignant'), exist_ok=True)

# Load the metadata file
metadata_df = pd.read_excel(metadata_file)

# Define Bi-Rads classification rules
def classify_birads(birads):
    """
    Classify BI-RADS categories into binary labels:
    - BI-RADS 2, 3, and 4a as benign (0).
    - BI-RADS 4c and 5 as malignant (1).
    - Discard BI-RADS 1 and 4b (return None) as theey are not fitting for this binary classification
    """

    # Handle NaN explicitly
    if birads is None or (isinstance(birads, float) and math.isnan(birads)):
        return None

    if isinstance(birads, str):
        birads = birads.lower().strip()
        if birads == "4a":
            return "benign"
        elif birads == "4b":
            print(f"Skipping file with BI-RADS {birads}.")
            return None  # Discard
        elif birads == "4c":
            return "malignant"
    
    try:
        birads = int(birads)
    except ValueError:
        raise ValueError(f"Unexpected BI-RADS value: {birads}")

    if 2 <= birads <= 4:  # Treat plain 4 as benign
        return "benign"
    elif birads >= 5:
        return "malignant"
    else:
        print(f"Skipping file with BI-RADS {birads}.")  # Log discard
        return None

# Process each row in the metadata
for _, row in metadata_df.iterrows():
    # Extract the base file name
    file_name_base = str(int(row['File Name']))  # Ensure it's a string
    birads = row['Bi-Rads']
    
    classification = classify_birads(birads)
    if classification is None:
        continue  # Skip files with irrelevant Bi-Rads values

    # Find the actual file in the source folder
    matching_files = [
        f for f in os.listdir(source_folder)
        if f.startswith(file_name_base) and f.endswith(".dcm")
    ]

    if not matching_files:
        print(f"File not found: {file_name_base}")
        continue

    # Use the first match (there should only be one)
    dicom_path = os.path.join(source_folder, matching_files[0])

    # Create classification folder
    subject_folder = os.path.join(destination_folder, classification, file_name_base)
    os.makedirs(subject_folder, exist_ok=True)

    # Convert and save as NIfTI
    try:
        nifti_image = load_dicom_and_save_as_nifti(dicom_path)
        nifti_path = os.path.join(subject_folder, 'slice.nii.gz')
        nib.save(nifti_image, nifti_path)
        print(f"Processed {matching_files[0]} -> {nifti_path}")
    except Exception as e:
        print(f"Failed to process {matching_files[0]}: {e}")

