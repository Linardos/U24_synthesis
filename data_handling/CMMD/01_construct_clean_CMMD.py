import os
import pandas as pd
import nibabel as nib
import pydicom
import numpy as np
from pydicom import dcmread
from nibabel import Nifti1Image
from skimage.transform import resize   # ← new

# ---------------------------------------------------------------------
#  helper ── uniformly resize 2-D slice to 256✕256
# ---------------------------------------------------------------------
def resize_to_256(slice2d: np.ndarray) -> np.ndarray:
    """Resize a 2-D numpy slice to 256×256 with bilinear interpolation."""
    # skimage returns float64; cast back to original dtype after clamping
    resized = resize(
        slice2d,
        (256, 256),
        order=1,              # bilinear
        preserve_range=True,  # keep original intensity scale
        anti_aliasing=True
    )
    return resized.astype(slice2d.dtype)

# ---------------------------------------------------------------------
#  DICOM → NIfTI (single-slice) with resizing
# ---------------------------------------------------------------------
def load_dicom_and_save_as_nifti(dicom_file_path: str) -> Nifti1Image:
    dicom_data = dcmread(dicom_file_path)
    pixel_array = dicom_data.pixel_array            # shape (H, W)
    pixel_array = resize_to_256(pixel_array)        # shape → (256, 256)

    volume = np.expand_dims(pixel_array, axis=-1)   # (256, 256, 1)
    affine = np.eye(4)                              # simple identity affine
    return Nifti1Image(volume, affine)

# ---------------------------------------------------------------------
#  paths & metadata
# ---------------------------------------------------------------------
source_folder = '/mnt/d/Datasets/CMMD/'
destination_folder = os.path.join(source_folder, 'CMMD_binary_256x256')
os.makedirs(destination_folder, exist_ok=True)

metadata_df = pd.read_csv(os.path.join(source_folder, 'combined_metadata.csv'))

# ---------------------------------------------------------------------
#  iterate through metadata rows and convert
# ---------------------------------------------------------------------
for _, row in metadata_df.iterrows():
    raw_file_location = row['File Location'].replace('\\', '/')
    dicom_folder_path = os.path.join(source_folder, raw_file_location.lstrip('./'))

    subject_id    = row['Subject ID']
    classification = row['classification'].lower()

    dicom_files = sorted(f for f in os.listdir(dicom_folder_path) if f.endswith('.dcm'))

    for slice_index, dicom_file in enumerate(dicom_files, start=1):
        new_folder_name = f"{subject_id}-{slice_index}"
        new_folder_path = os.path.join(destination_folder, classification, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        new_file_path  = os.path.join(new_folder_path, 'slice.nii.gz')
        dicom_file_path = os.path.join(dicom_folder_path, dicom_file)

        try:
            nifti_image = load_dicom_and_save_as_nifti(dicom_file_path)
            nib.save(nifti_image, new_file_path)
        except Exception as e:
            print(f"Failed to convert {subject_id} slice {slice_index}: {e}")
            continue

        print(f"Processed {subject_id} - Slice {slice_index} → {new_file_path}")
