import pandas as pd

# Paths to CSV files
clinical_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_clinical_reduced.csv"
metadata_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_metadata_reduced.csv"
output_csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_cleaned_metadata.csv"

# Load the CSV files
clinical_df = pd.read_csv(clinical_csv_path)
metadata_df = pd.read_csv(metadata_csv_path)

# Select relevant columns
clinical_columns = ['empi_anon', 'acc_anon', 'asses']  # BI-RADS is in 'asses'
metadata_columns = ['empi_anon', 'acc_anon', 'anon_dicom_path', 'FinalImageType', 'ImageLateralityFinal', 'ViewPosition']

clinical_df = clinical_df[clinical_columns]
metadata_df = metadata_df[metadata_columns]

# Merge the dataframes on 'empi_anon' and 'acc_anon'
cleaned_df = pd.merge(clinical_df, metadata_df, on=['empi_anon', 'acc_anon'], how='inner')

# Save the cleaned data
cleaned_df.to_csv(output_csv_path, index=False)

print(f"Cleaned data saved to {output_csv_path}")
