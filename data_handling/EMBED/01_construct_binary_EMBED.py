#!/usr/bin/env python3
# ------------------------------------------------------------
#  Breast-cancer C-view  ➜  256×256 NIfTI dataset builder
#  – keeps ALL malignant cases, subsamples other classes
#  – patient-level (empi_anon) stratified train/val/test split
# ------------------------------------------------------------
import os, multiprocessing as mp
import pandas as pd
import numpy as np
import nibabel as nib
from pydicom import dcmread
from nibabel import Nifti1Image
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from pydicom.tag import Tag

RESIZE_DIM = 256
RNG_SEED   = 42

# ------------------------------------------------------------
# 0.  Load metadata  ▸ map BI-RADS codes → coarse categories
# ------------------------------------------------------------
csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_cleaned_metadata.csv"
output_folder_name = "EMBED_binary_clean"
data = pd.read_csv(csv_path)

birads_map = {
    # "A": "further eval"
    "N": "benign",
    "B": "benign",
    # "P": "probably_benign",
    # "S": "malignant",
    "M": "malignant",
    "K": "malignant",
}
data["category"] = data["asses"].map(birads_map).fillna("unknown")

# 🔸 ignore records whose category is still unknown
data = data[data["category"] != "unknown"].copy()

# 🔹 ignore magnification views
data = data[data["spot_mag"] != 1].copy()

# ------------------------------------------------------------
# 0.b  Laterality filter (MALIGNANT ONLY)
#      - For malignant: keep if side == ImageLateralityFinal, or side == 'B' (both)
#      - For benign:    keep everything (no laterality constraint)
# ------------------------------------------------------------
def _norm_side(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().upper()
    if s in {"RIGHT", "RT"}: return "R"
    if s in {"LEFT", "LT"}:  return "L"
    if s in {"BOTH", "B"}:   return "B"
    if s in {"R", "L"}:      return s
    return np.nan  # treat anything else as unknown

required_cols = {"side", "ImageLateralityFinal"}
missing = required_cols - set(data.columns)
if missing:
    raise KeyError(f"Missing columns in CSV: {missing}")

data["side_norm"]    = data["side"].map(_norm_side)
data["img_lat_norm"] = data["ImageLateralityFinal"].map(_norm_side)

is_malig = data["category"] == "malignant"

# malignant rows must have valid laterality and match, unless side == 'B'
mask_malig_valid = (
    is_malig &
    data["img_lat_norm"].isin(["L", "R"]) &
    data["side_norm"].isin(["L", "R", "B"]) &
    ((data["side_norm"] == data["img_lat_norm"]) | (data["side_norm"] == "B"))
)

# benign rows pass through
mask_benign = ~is_malig

mask_keep = mask_benign | mask_malig_valid

dropped = (~mask_keep).sum()
if dropped:
    print(f"Dropping {dropped} rows (malignant laterality mismatch/unknown):")
    print(
        data.loc[~mask_keep, ["empi_anon","category","asses","side","ImageLateralityFinal","anon_dicom_path"]]
            .head(10).to_string(index=False)
    )

data = data.loc[mask_keep].drop(columns=["side_norm","img_lat_norm"]).reset_index(drop=True)


# ------------------------------------------------------------
# 1.  Cap dataset size per category – keep all malignant
# ------------------------------------------------------------
malignant_count = (data["category"] == "malignant").sum()
if malignant_count == 0:
    raise ValueError("No malignant cases left after filtering!")

desired_counts = {
    "benign":      3 * malignant_count,
    "malignant":   None,   # keep all
}
print("Target sample sizes per class:")
for k, v in desired_counts.items():
    print(f"  {k:<10} : {v if v is not None else 'ALL'}")

def _cap(g):
    limit = desired_counts[g.name]
    if limit is None or len(g) <= limit:
        return g
    return g.sample(n=limit, random_state=RNG_SEED)

data = data.groupby("category", group_keys=False).apply(_cap).reset_index(drop=True)

# ------------------------------------------------------------
# 2.  Patient-level label & patient-stratified 80/10/10 split
#     - patient label = malignant if ANY record malignant, else benign
#     - ensure all rows for a given empi_anon go to the same split
# ------------------------------------------------------------
# Build patient label table
patient_labels = (
    data.groupby("empi_anon")["category"]
        .apply(lambda s: "malignant" if (s == "malignant").any() else "benign")
        .rename("patient_label")
        .reset_index()
)

# First split: train_val vs test (80/20), stratified by patient label
train_val_pat, test_pat = train_test_split(
    patient_labels,
    test_size=0.20,
    random_state=RNG_SEED,
    stratify=patient_labels["patient_label"],
)

# Second split: train vs val from train_val (to yield ~70/10/20 overall)
val_fraction_of_trainval = 0.125  # 0.125 * 0.80 ≈ 0.10 overall
train_pat, val_pat = train_test_split(
    train_val_pat,
    test_size=val_fraction_of_trainval,
    random_state=RNG_SEED,
    stratify=train_val_pat["patient_label"],
)

# Turn patient sets into fast lookup
train_ids = set(train_pat["empi_anon"])
val_ids   = set(val_pat["empi_anon"])
test_ids  = set(test_pat["empi_anon"])

# Map rows to splits by patient membership
def _which_split(empi):
    if empi in train_ids: return "train"
    if empi in val_ids:   return "val"
    if empi in test_ids:  return "test"
    return "unknown"

data["split"] = data["empi_anon"].map(_which_split)
assert (data["split"] != "unknown").all(), "Some rows were not assigned a split!"

# Extract split frames
train = data[data["split"] == "train"].copy()
val   = data[data["split"] == "val"].copy()
test  = data[data["split"] == "test"].copy()

# Sanity: show image-level and patient-level counts
def _show_split(name, df):
    print(f"\n{name} (image-level) counts:")
    print(df["category"].value_counts().to_string())

    pats = df["empi_anon"].unique()
    pat_df = patient_labels[patient_labels["empi_anon"].isin(pats)]
    print(f"{name} (patient-level) counts:")
    print(pat_df["patient_label"].value_counts().to_string())

_show_split("TRAIN", train)
_show_split("VAL",   val)
_show_split("TEST",  test)

# ------------------------------------------------------------
# 3.  Folder layout
# ------------------------------------------------------------
base_dir   = "/mnt/d/Datasets/EMBED/images/"
output_dir = f"/mnt/d/Datasets/EMBED/{output_folder_name}"
categories = ["benign", "malignant"]

for split in ["train", "val", "test"]:
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
IMPLANT_TAG = Tag(0x0028, 0x1300)   # Breast Implant Present (CS)

def _process_one(row, split):
    src = os.path.join(
        base_dir,
        os.path.relpath(row["anon_dicom_path"], "/mnt/NAS2/mammo/anon_dicom/")
    )
    cat = row["category"]

    if not os.path.isfile(src):
        print(f"Missing file: {src}")
        return

    # ── quick header check ──────────────────────────────
    try:
        hdr = dcmread(src, stop_before_pixels=True)
        implant_flag = str(hdr.get(IMPLANT_TAG, "")).upper()
        if implant_flag == "YES":
            print(f"Skip implant   : {src}")
            return
    except Exception as e:
        print(f"Header read err : {src} → {e}")

    # ── normal processing ──────────────────────────────
    dst_folder = os.path.join(
        output_dir, split, cat, os.path.splitext(os.path.basename(src))[0]
    )
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

print("Processing val files …")
_process_split(val, "val")

print("Processing test files …")
_process_split(test, "test")

print("✅ Dataset reorganisation & conversion complete.")
