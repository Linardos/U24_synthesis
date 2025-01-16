import os
import pandas as pd
import nibabel as nib
import numpy as np
from pydicom import dcmread
from nibabel import Nifti1Image
from sklearn.model_selection import train_test_split

# Load CSV
csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_cleaned_metadata.csv"
data = pd.read_csv(csv_path)

# Map BI-RADS classifications to categories
birads_map = {
    "N": "benign",
    "B": "benign",
    "P": "probably_benign",
    "S": "suspicious",
    "M": "suspicious",
    "K": "malignant"
}
data["category"] = data["asses"].map(birads_map).fillna("unknown")

# Train/Test Split
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["category"])

# Directory structure
base_dir = "/mnt/d/Datasets/EMBED/images/"
output_dir = "/mnt/d/Datasets/EMBED/EMBED_clean"
categories = ["benign", "probably_benign", "suspicious", "malignant"]

for split in ["train", "test"]:
    for category in categories:
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)
    print(f"{split.capitalize()} directories created!")

# DICOM-to-NIfTI conversion function
def convert_dicom_to_nifti(dicom_path, output_path):
    try:
        dicom_data = dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array  # Extract pixel data
        volume = np.expand_dims(pixel_array, axis=-1)  # Add a dimension for NIfTI format

        # Create a simple affine matrix
        affine = np.eye(4)
        nifti_image = Nifti1Image(volume, affine)

        # Save as NIfTI
        nib.save(nifti_image, output_path)
        print(f"Converted and saved: {output_path}")
    except Exception as e:
        print(f"Error converting {dicom_path} to NIfTI: {e}")

# Helper function to process files
def process_files(df, split):
    for _, row in df.iterrows():
        src = os.path.join(base_dir, os.path.relpath(row["anon_dicom_path"], "/mnt/NAS2/mammo/anon_dicom/"))
        category = row["category"]

        if not os.path.isfile(src):
            print(f"File not found: {src}")
            continue

        # Create a folder for the DICOM file with its name
        dicom_name = os.path.splitext(os.path.basename(src))[0]  # Extract file name without extension
        dest_folder = os.path.join(output_dir, split, category, dicom_name)
        os.makedirs(dest_folder, exist_ok=True)

        # Convert and save the NIfTI file in the folder
        nifti_path = os.path.join(dest_folder, "slice.nii.gz")
        convert_dicom_to_nifti(src, nifti_path)

# Process train and test files
print("Processing train files...")
process_files(train, "train")
print("Processing test files...")
process_files(test, "test")

print("Dataset reorganization and conversion complete!")
