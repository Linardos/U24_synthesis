import os
import math
import pandas as pd
import nibabel as nib
import numpy as np
from pydicom import dcmread
from nibabel import Nifti1Image

# ---------------------------------------------------------------------
#  DICOM -> NIfTI (single-slice)
# ---------------------------------------------------------------------
def load_dicom_and_save_as_nifti(dicom_file_path):
    dicom_data = dcmread(dicom_file_path)
    pixel_array = dicom_data.pixel_array  # Extract pixel array
    volume = np.expand_dims(pixel_array, axis=-1)  # (H, W, 1)

    affine = np.eye(4)
    nifti_image = Nifti1Image(volume, affine)
    return nifti_image

# ---------------------------------------------------------------------
#  BI-RADS classifier
#  Keep:
#    1,2 -> benign
#    5,6 -> malignant
#  Exclude:
#    3,4 and anything else
# ---------------------------------------------------------------------
def classify_birads(birads):
    if birads is None or (isinstance(birads, float) and math.isnan(birads)):
        return None

    # Normalize strings like "5", "5.0", " 2 ", etc.
    if isinstance(birads, str):
        birads = birads.strip().lower()

    try:
        birads_int = int(float(birads))
    except (ValueError, TypeError):
        print(f"Skipping unexpected BI-RADS value: {birads}")
        return None

    if birads_int in (1, 2):
        return "benign"
    elif birads_int in (5, 6):
        return "malignant"
    else:
        # Exclude 3 and 4
        return None

# ---------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------
source_folder = '/mnt/d/Datasets/inbreast/ALL-IMGS'
metadata_file = '/mnt/d/Datasets/inbreast/INbreast.xls'
destination_folder = '/mnt/d/Datasets/inbreast/inbreast_clean'

os.makedirs(os.path.join(destination_folder, 'benign'), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'malignant'), exist_ok=True)

# ---------------------------------------------------------------------
#  Load metadata
# ---------------------------------------------------------------------
metadata_df = pd.read_excel(metadata_file)

# Optional: show what BI-RADS values exist before processing
print("Unique BI-RADS values in metadata:")
print(sorted(metadata_df['Bi-Rads'].dropna().astype(str).unique()))

# ---------------------------------------------------------------------
#  Process each row
# ---------------------------------------------------------------------
for _, row in metadata_df.iterrows():
    file_name = row['File Name']

    if pd.isna(file_name):
        print("Skipping row with missing File Name")
        continue

    file_name_base = str(int(file_name))
    # file_name_base = str(int(row['File Name']))
    birads = row['Bi-Rads']

    classification = classify_birads(birads)
    if classification is None:
        continue

    matching_files = [
        f for f in os.listdir(source_folder)
        if f.startswith(file_name_base) and f.endswith(".dcm")
    ]

    if not matching_files:
        print(f"File not found: {file_name_base}")
        continue

    dicom_path = os.path.join(source_folder, matching_files[0])

    subject_folder = os.path.join(destination_folder, classification, file_name_base)
    os.makedirs(subject_folder, exist_ok=True)

    nifti_path = os.path.join(subject_folder, 'slice.nii.gz')

    # Skip existing files
    if os.path.exists(nifti_path):
        print(f"Skipping already processed: {file_name_base}")
        continue

    try:
        nifti_image = load_dicom_and_save_as_nifti(dicom_path)
        nib.save(nifti_image, nifti_path)
        print(f"Processed {matching_files[0]} -> {nifti_path}")
    except Exception as e:
        print(f"Failed to process {matching_files[0]}: {e}")