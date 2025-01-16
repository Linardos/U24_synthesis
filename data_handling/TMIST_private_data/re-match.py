import pandas as pd
import hashlib

# Load original and curated de-identified data
# original_file = "/mnt/d/Datasets/TMIST/doris dbt search/doris_download_2024-12-05_135842.csv"
original_file = "/mnt/d/Datasets/TMIST/doris malignant search/doris_download_2024-12-05_140400.csv"
# curated_file = "/mnt/d/Datasets/TMIST/doris malignant search/curated_deidentified_malignant.csv"
curated_file = "/mnt/d/Datasets/TMIST/doris malignant search/deidentified_output_malignant.csv" # sanity check, this will be what we get from Celine

df_original = pd.read_csv(original_file)
df_curated = pd.read_csv(curated_file)

# Function to hash sensitive information (must match de-identification logic)
def hash_value(value):
    return hashlib.sha256(str(value).encode()).hexdigest() if pd.notnull(value) else value

# Hash the original identifiers to create a mapping
columns_to_hash = ['mrn', 'attending_npi', 'dictating_npi']  # Columns that were hashed
for col in columns_to_hash:
    if col in df_original.columns:
        df_original[f"{col}_hashed"] = df_original[col].apply(hash_value)

# Merge original and curated data based on hashed columns
merge_columns = [f"{col}_hashed" for col in columns_to_hash if col in df_original.columns]
df_matched = pd.merge(df_original, df_curated, left_on=merge_columns, right_on=columns_to_hash, how='inner')

# Drop hashed columns if no longer needed
df_matched.drop(columns=merge_columns, inplace=True)

# Save the matched data
output_matched_file = "/mnt/d/Datasets/TMIST/doris malignant search/matched_curated_original.csv"
df_matched.to_csv(output_matched_file, index=False)

print(f"Matched data saved to {output_matched_file}")
