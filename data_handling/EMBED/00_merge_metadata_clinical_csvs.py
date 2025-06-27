#!/usr/bin/env python3
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────────────────────
clinical_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_clinical.csv"
metadata_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_metadata.csv"
output_csv_path   = "/mnt/d/Datasets/EMBED/tables/EMBED_cleaned_metadata.csv"

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
# 4.  Merge on patient & accession IDs
# ─────────────────────────────────────────────────────────────
cleaned_df = pd.merge(
    clinical_df,
    metadata_df,
    on=["empi_anon", "acc_anon"],
    how="inner",
)

# ─────────────────────────────────────────────────────────────
# 5.  Save
# ─────────────────────────────────────────────────────────────
cleaned_df.to_csv(output_csv_path, index=False)
print(f"Cleaned data saved to {output_csv_path}")