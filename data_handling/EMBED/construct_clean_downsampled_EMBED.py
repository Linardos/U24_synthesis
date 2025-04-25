#!/usr/bin/env python3
# ------------------------------------------------------------
#  Breast-cancer C-view  ➜  512×512 NIfTI dataset builder
#  – keeps ALL malignant cases, subsamples other classes
# ------------------------------------------------------------
import os, multiprocessing as mp
import pandas as pd
import numpy as np
import nibabel as nib
from pydicom import dcmread
from nibabel import Nifti1Image
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

# ------------------------------------------------------------
# 0.  Load metadata  ▸ map BI-RADS codes → four categories
# ------------------------------------------------------------
csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_cleaned_metadata.csv"
data = pd.read_csv(csv_path)

birads_map = {
    "N": "benign",
    "B": "benign",
    "P": "probably_benign",
    "S": "suspicious",
    "M": "suspicious",
    "K": "malignant",
}
data["category"] = data["asses"].map(birads_map).fillna("unknown")

# ------------------------------------------------------------
# 1.  Cap dataset size per category *before* train/test split
# ------------------------------------------------------------
desired_counts = {
    "benign":            16_000,
    "probably_benign":    8_000,
    "suspicious":         6_000,
    "malignant":          None,   # keep everything
}

def _cap(g):
    limit = desired_counts.get(g.name, None)
    if limit is None or len(g) <= limit:
        return g
    return g.sample(n=limit, random_state=42)

data = (
    data.groupby("category", group_keys=False)
        .apply(_cap)
)

# ------------------------------------------------------------
# 2.  Stratified 80/20 split on the *trimmed* dataframe
# ------------------------------------------------------------
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["category"],
)

# ------------------------------------------------------------
# 3.  Folder layout
# ------------------------------------------------------------
base_dir   = "/mnt/d/Datasets/EMBED/images/"
output_dir = "/mnt/d/Datasets/EMBED/EMBED_clean_512x512"
categories = list(desired_counts.keys())

for split in ["train", "test"]:
    for cat in categories:
        os.makedirs(os.path.join(output_dir, split, cat), exist_ok=True)
    print(f"{split.capitalize()} directories created!")

# ------------------------------------------------------------
# 4.  DICOM → 512×512 NIfTI conversion
# ------------------------------------------------------------
def convert_dicom_to_nifti(dicom_path, output_path, target_size=(512, 512)):
    try:
        ds = dcmread(dicom_path)
        img = ds.pixel_array

        scale = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
        resized = zoom(img, scale, order=3)             # bicubic

        volume = np.expand_dims(resized, -1)
        nib.save(Nifti1Image(volume, np.eye(4)), output_path)
        print(f"Converted: {output_path}")
    except Exception as e:
        print(f"Error {dicom_path} → {e}")

# ------------------------------------------------------------
# 5.  Worker helpers
# ------------------------------------------------------------
def _process_one(row, split):
    src = os.path.join(
        base_dir,
        os.path.relpath(row["anon_dicom_path"], "/mnt/NAS2/mammo/anon_dicom/")
    )
    cat = row["category"]

    if not os.path.isfile(src):
        print(f"Missing file: {src}")
        return

    dst_folder = os.path.join(output_dir, split, cat, os.path.splitext(os.path.basename(src))[0])
    os.makedirs(dst_folder, exist_ok=True)

    dst_nifti = os.path.join(dst_folder, "slice.nii.gz")
    if not os.path.exists(dst_nifti):
        convert_dicom_to_nifti(src, dst_nifti)
    else:
        print(f"Skipping (exists): {dst_nifti}")

def _process_split(df, split):
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(_process_one, [(row, split) for _, row in df.iterrows()])

# ------------------------------------------------------------
# 6.  Go!
# ------------------------------------------------------------
print("Processing train files …")
_process_split(train, "train")

print("Processing test files …")
_process_split(test, "test")

print("✅ Dataset reorganisation & conversion complete.")
