import pandas as pd

# Paths to CSV files
clinical_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_clinical_reduced.csv"
metadata_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_metadata.csv"
output_csv_path   = "/mnt/d/Datasets/EMBED/tables/EMBED_cleaned_metadata.csv"

# ─────────────────────────────────────────────────────────────
# 1.  Load CSVs
# ─────────────────────────────────────────────────────────────
clinical_df  = pd.read_csv(clinical_csv_path)
metadata_df  = pd.read_csv(metadata_csv_path)

# ─────────────────────────────────────────────────────────────
# 2.  Keep only the columns we want
# ─────────────────────────────────────────────────────────────
clinical_columns = ["empi_anon", "acc_anon", "asses"]              # BI‑RADS codes
metadata_columns = [
    "empi_anon",
    "acc_anon",
    "anon_dicom_path",
    "FinalImageType",
    "ImageLateralityFinal",
    "ViewPosition",
    "spot_mag",           # ← NEW: magnification flag
]

clinical_df = clinical_df[clinical_columns]
metadata_df = metadata_df[metadata_columns]

# ─────────────────────────────────────────────────────────────
# 3.  Merge on patient & accession IDs
# ─────────────────────────────────────────────────────────────
cleaned_df = pd.merge(
    clinical_df,
    metadata_df,
    on=["empi_anon", "acc_anon"],
    how="inner",
)

# ─────────────────────────────────────────────────────────────
# 4.  Save
# ─────────────────────────────────────────────────────────────
cleaned_df.to_csv(output_csv_path, index=False)
print(f"Cleaned data saved to {output_csv_path}")
