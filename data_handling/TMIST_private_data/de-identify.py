import pandas as pd
import hashlib
import uuid
import re

# Load the CSV file
input_file = "/mnt/d/Datasets/TMIST/doris dbt search/doris_download_2024-12-05_135842.csv"
# input_file = "/mnt/d/Datasets/TMIST/doris malignant search/doris_download_2024-12-05_140400.csv"
output_file = "/mnt/d/Datasets/TMIST/doris dbt search/deidentified_output_dbt_search.csv"
# output_file = "/mnt/d/Datasets/TMIST/doris malignant search/deidentified_output_malignant.csv"
df = pd.read_csv(input_file)

# Function to hash sensitive information
def hash_value(value):
    return hashlib.sha256(str(value).encode()).hexdigest() if pd.notnull(value) else value

# Function to generate unique pseudonyms
def pseudonymize(value):
    return str(uuid.uuid4())[:8] if pd.notnull(value) else value

# Remove or hash identifiable information
columns_to_hash = ['mrn', 'attending_npi', 'dictating_npi']  # Columns to hash
columns_to_pseudonymize = ['pt', 'attending_name', 'dictating_name', 'ordmd', 'resource']  # Columns to pseudonymize

for col in columns_to_hash:
    if col in df.columns:
        df[col] = df[col].apply(hash_value)

for col in columns_to_pseudonymize:
    if col in df.columns:
        df[col] = df[col].apply(pseudonymize)

# handle reptext col
if 'reptext' in df.columns:
    df['reptext'] = df['reptext'].apply(
        lambda x: re.sub(r'Electronically signed and approved by: .*?, MD', 
                         'Electronically signed and approved by: XXXX, MD', 
                         x, flags=re.DOTALL) if pd.notnull(x) else x
    )

# Save the de-identified CSV
df.to_csv(output_file, index=False)

print(f"De-identified CSV saved to {output_file}")
