#!/usr/bin/env python3
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────────────────────
clinical_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_clinical.csv"
metadata_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_metadata.csv"
output_csv_path   = "/mnt/d/Datasets/EMBED/tables/EMBED_mastersheet.csv"

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
    "asses",        # BI-RADS (letters A/N/B/P/S/M/K)
    "implanfind",   # implant findings flag
    "tissueden"
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

clinical_df = clinical_df[clinical_cols].copy()
metadata_df = metadata_df[metadata_cols].copy()

# ─────────────────────────────────────────────────────────────
# 3. Image-side filter
#    a) drop implant-displaced (“ID”) views
#    b) keep ONLY ViewPosition in {"MLO", "CC"}
# ─────────────────────────────────────────────────────────────
vp = metadata_df["ViewPosition"].astype(str).str.strip()
metadata_df = metadata_df.loc[
    ~vp.str.contains("ID", case=False, na=False) &
    vp.str.upper().isin({"MLO", "CC"})
].copy()

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
# 5.  Column renames
#     - empi_anon -> MRN
#     - acc_anon  -> ACC_ID
# ─────────────────────────────────────────────────────────────
cleaned_df = cleaned_df.rename(columns={
    "empi_anon": "MRN",
    "acc_anon": "ACC_ID",
    "tissueden": "BIRADS DENSITY"
})

# ─────────────────────────────────────────────────────────────
# 6.  Map BI-RADS letters -> numbers in `asses`
#     A→0, N→1, B→2, P→3, S→4, M→5, K→6
# ─────────────────────────────────────────────────────────────
birads_map = {
    "A": 0,  # Additional evaluation
    "N": 1,  # Negative
    "B": 2,  # Benign
    "P": 3,  # Probably benign
    "S": 4,  # Suspicious
    "M": 5,  # Highly suggestive of malignancy
    "K": 6,  # Known biopsy proven
}
# normalize, then map
cleaned_df["BIRADS TUMOR ASSESSMENT"] = (
    cleaned_df["asses"]
    .astype(str).str.strip().str.upper()
    .map(birads_map)  # unmapped become NaN
)

# ─────────────────────────────────────────────────────────────
# 7.  Rename FinalImageType value "2D" -> "conventional 2D mammogram"
# ─────────────────────────────────────────────────────────────
fit = cleaned_df["FinalImageType"].astype(str).str.strip()
cleaned_df["FinalImageType"] = fit.where(
    ~fit.str.upper().eq("2D"),
    other="conventional 2D mammogram"
)

# ─────────────────────────────────────────────────────────────
# 8.  Save
# ─────────────────────────────────────────────────────────────
cleaned_df.to_csv(output_csv_path, index=False)
print(f"Cleaned data saved to {output_csv_path}")
