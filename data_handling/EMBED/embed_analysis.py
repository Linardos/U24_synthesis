import pandas as pd

# File paths
metadata_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_metadata_reduced.csv"
clinical_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_clinical_reduced.csv"
output_csv = "patients_with_diagnosis_changes.csv"

# Load datasets
metadata_df = pd.read_csv(metadata_path, low_memory=False)
clinical_df = pd.read_csv(clinical_path, low_memory=False)

# Convert study_date_anon to datetime for proper merging
metadata_df["study_date_anon"] = pd.to_datetime(metadata_df["study_date_anon"], errors="coerce")
clinical_df["study_date_anon"] = pd.to_datetime(clinical_df["study_date_anon"], errors="coerce")

# Merge based on patient ID and study date
merged_df = metadata_df.merge(clinical_df, on=["empi_anon", "study_date_anon"], how="left")

# Debugging: Check data integrity
print(f"Metadata rows: {len(metadata_df)}, Clinical rows: {len(clinical_df)}, Merged rows: {len(merged_df)}")
print(f"Non-null desc before merge: {clinical_df['desc'].notna().sum()}")
print(f"Non-null desc after merge: {merged_df['desc'].notna().sum()}")
print("Unique values in desc column:", clinical_df["desc"].unique())

# Identify patients with multiple timepoints and different diagnoses
grouped_desc = merged_df.groupby("empi_anon")["desc"].nunique()
patients_with_diagnosis_change = grouped_desc[grouped_desc > 1]

if patients_with_diagnosis_change.empty:
    print("No patients with changing diagnoses found.")
else:
    # Extract relevant details including BI-RADS (asses)
    diagnosis_changes_df = merged_df[merged_df["empi_anon"].isin(patients_with_diagnosis_change.index)]
    diagnosis_changes_df = diagnosis_changes_df[["empi_anon", "study_date_anon", "desc", "asses"]].drop_duplicates()
    
    # Rename 'asses' column to 'BI-RADS' for clarity
    diagnosis_changes_df.rename(columns={"asses": "BI-RADS"}, inplace=True)
    
    # Sort by patient and time
    diagnosis_changes_df = diagnosis_changes_df.sort_values(["empi_anon", "study_date_anon"])

    # Save results
    diagnosis_changes_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
