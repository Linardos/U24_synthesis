#!/usr/bin/env python3
import pandas as pd
import numpy as np

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
    "tissueden",
    "side",         # laterality of finding described in current row (R,L,B)
    "ETHNICITY_DESC",
    "age_at_study",
    "study_date_anon",
    "desc",
    "loc_num",
    "path_severity",   # added so we can sort by worst pathology too
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
asses_order = {"N": 0, "B": 1, "A": 2, "P": 3, "S": 4, "M": 5, "K": 6}

cleaned_df["asses_norm"] = (
    cleaned_df["asses"]
    .astype(str).str.strip().str.upper()
)
cleaned_df["asses_ord"] = cleaned_df["asses_norm"].map(asses_order)

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
    .drop(columns=["asses_ord", "path_severity_sort", "asses_norm"])
)

# Optional sanity check
assert cleaned_df["anon_dicom_path"].is_unique

# ─────────────────────────────────────────────────────────────
# 5.  Column renames
#     - empi_anon -> MRN
#     - acc_anon  -> ACC_ID
#     - tissueden -> BIRADS DENSITY
#     - side -> LATERALITY_RELEVANT_TO_FINDING
#     - ETHNICITY_DESC -> RACE
#     - age_at_study -> AGE
#     - study_date -> STUDY_DATE
# ─────────────────────────────────────────────────────────────
cleaned_df = cleaned_df.rename(columns={
    "empi_anon": "MRN",
    "acc_anon": "ACC_ID",
    "tissueden": "BIRADS DENSITY",
    "side": "LATERALITY_RELEVANT_TO_FINDING",
    "ETHNICITY_DESC": "RACE",
    "age_at_study": "AGE",
    "study_date_anon": "STUDY_DATE",
    "desc": "STUDY_DESC",
    "loc_num": "INSTITUTION_LOCATION_CODE",
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
# 8.  Add constant columns
# ─────────────────────────────────────────────────────────────
cleaned_df["ORGAN"] = "BRST"
cleaned_df["SOURCE"] = "EMBED"

# ─────────────────────────────────────────────────────────────
# 9.  Save
# ─────────────────────────────────────────────────────────────
cleaned_df.to_csv(output_csv_path, index=False)
print(f"Cleaned data saved to {output_csv_path}")
print(f"Final rows (unique DICOMs): {len(cleaned_df)}")

# Quick duplicate sanity check
dup_count = cleaned_df["anon_dicom_path"].duplicated().sum()
print("Duplicates in anon_dicom_path:", dup_count)
