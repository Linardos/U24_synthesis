#!/usr/bin/env python3
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────────────────────
clinical_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_clinical.csv"
metadata_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_metadata_reduced.csv"
output_csv_path   = "/mnt/d/Datasets/EMBED/tables/EMBED_cleaned_metadata.csv"
# output_csv_path   = "/mnt/d/Datasets/EMBED/tables/EMBED_cleaned_Project_2.csv"

# ─────────────────────────────────────────────────────────────
# 1.  Load CSVs
# ─────────────────────────────────────────────────────────────
clinical_df  = pd.read_csv(clinical_csv_path, low_memory=False)
metadata_df  = pd.read_csv(metadata_csv_path, low_memory=False)

# ─────────────────────────────────────────────────────────────
# 2.  Keep only the columns we need
# ─────────────────────────────────────────────────────────────
clinical_cols = [
    "empi_anon",
    "acc_anon",
    "asses",        # BI-RADS
    "implanfind",   # implant findings flag
    "tissueden",
    "path_severity",
    "side",         # laterality of finding described in current row (R,L,B)
    "bside"         # laterality of biopsied finding (R,L)
]
metadata_cols = [
    "empi_anon",
    "acc_anon",
    "anon_dicom_path",
    "FinalImageType",
    "ImageLateralityFinal",
    "ViewPosition",
    "spot_mag",     # paddle / spot-compression flag
]

clinical_df = clinical_df[clinical_cols]
metadata_df = metadata_df[metadata_cols]

# ─────────────────────────────────────────────────────────────
# 3. Image-side filter  ➜  drop implant-displaced (“ID”) views
# ─────────────────────────────────────────────────────────────
metadata_df = metadata_df[
    ~metadata_df["ViewPosition"]
        .fillna("")              # handle NaNs
        .str.contains("ID", case=False)
]

# ─────────────────────────────────────────────────────────────
# 4.  Merge on patient & accession IDs  (many-to-many)
# ─────────────────────────────────────────────────────────────
cleaned_df = pd.merge(
    clinical_df,
    metadata_df,
    on=["empi_anon", "acc_anon"],
    how="inner",
)

# ─────────────────────────────────────────────────────────────
# 4b. Deduplicate per DICOM:
#     - worst BI-RADS (asses)
#     - then highest path_severity
# ─────────────────────────────────────────────────────────────

# Map BI-RADS categories to an ordinal severity
asses_order = {"N": 0, "B": 1, "A": 2, "P": 3, "S": 4, "M":5, "K":6}
cleaned_df["asses_ord"] = cleaned_df["asses"].map(asses_order)

# If path_severity is NaN, treat as lowest severity for sorting
if "path_severity" in cleaned_df.columns:
    cleaned_df["path_severity_sort"] = cleaned_df["path_severity"].fillna(-np.inf)
else:
    cleaned_df["path_severity_sort"] = -np.inf

# Sort so that "worst" rows come first within each anon_dicom_path
cleaned_df = (
    cleaned_df
    .sort_values(
        ["anon_dicom_path", "asses_ord", "path_severity_sort"],
        ascending=[True, False, False]
    )
    .drop_duplicates(subset=["anon_dicom_path"], keep="first")
    .drop(columns=["asses_ord", "path_severity_sort"])
)

# Optional sanity check
assert cleaned_df["anon_dicom_path"].is_unique

# ─────────────────────────────────────────────────────────────
# 5.  Save
# ─────────────────────────────────────────────────────────────
cleaned_df.to_csv(output_csv_path, index=False)
print(f"Cleaned data saved to {output_csv_path}")
print(f"Final rows (unique DICOMs): {len(cleaned_df)}")

df = pd.read_csv(output_csv_path)
print("Duplicates found:")
print(df["anon_dicom_path"].duplicated().sum())  # should be 0
