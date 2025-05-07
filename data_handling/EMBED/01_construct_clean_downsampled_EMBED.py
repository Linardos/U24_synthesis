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

RESIZE_DIM = 256
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

# map BI-RADS → coarse category labels
data["category"] = data["asses"].map(birads_map).fillna("unknown")

# 🔸 ignore records whose category is still unknown
data = data[data["category"] != "unknown"]

# 🔹 ignore magnification views
data = data[data["spot_mag"] != 1]

# ------------------------------------------------------------
# 1.  Cap dataset size per category – 3 : 2 : 2 : 1 matching
# ------------------------------------------------------------
# ➊  Count malignants after filtering
malignant_count = len(data[data["category"] == "malignant"])
if malignant_count == 0:
    raise ValueError("No malignant cases left after filtering!")

# ➋  Build the desired‑count dictionary on the fly
desired_counts = {
    "benign":            3 * malignant_count,
    "probably_benign":   2 * malignant_count,
    "suspicious":        2 * malignant_count,
    "malignant":         None,              # always keep all malignants
}

print("Target sample sizes per class:")
for k, v in desired_counts.items():
    print(f"  {k:<17} : {v if v is not None else 'ALL'}")

# ➌  Function to trim each group to its cap
def _cap(g):
    limit = desired_counts[g.name]          # every key exists by construction
    if limit is None or len(g) <= limit:    # keep all if limit is None
        return g
    return g.sample(n=limit, random_state=42)

# ➍  Apply the cap
data = data.groupby("category", group_keys=False).apply(_cap)

# ------------------------------------------------------------
# 2.  Stratified 80/20 split on the *trimmed* dataframe
# ------------------------------------------------------------
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["category"],
)

# Quick sanity check – prints counts to prove the 80 / 20 split
def _show_split(name, df):
    print(f"\n{name} counts:")
    print(df["category"].value_counts().to_string())
_show_split("TRAIN", train)
_show_split("TEST",  test)


# ------------------------------------------------------------
# 3.  Folder layout
# ------------------------------------------------------------
base_dir   = "/mnt/d/Datasets/EMBED/images/"
output_dir = f"/mnt/d/Datasets/EMBED/EMBED_clean_{RESIZE_DIM}x{RESIZE_DIM}"
categories = list(desired_counts.keys())

for split in ["train", "test"]:
    for cat in categories:
        os.makedirs(os.path.join(output_dir, split, cat), exist_ok=True)
    print(f"{split.capitalize()} directories created!")

# ------------------------------------------------------------
# 4.  DICOM → 256×256 NIfTI conversion
# ------------------------------------------------------------
def convert_dicom_to_nifti(dicom_path, output_path, target_size=(RESIZE_DIM, RESIZE_DIM)):
    try:
        ds = dcmread(dicom_path)
        img = ds.pixel_array

        scale = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
        resized = zoom(img, scale, order=3)             # bicubic

        volume = np.expand_dims(resized, -1)
        nib.save(Nifti1Image(volume, np.eye(4)), output_path)
        print(f"Converted: {output_path}")
    except Exception as e:
        print(f"❌ {dicom_path} → {e}")

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
