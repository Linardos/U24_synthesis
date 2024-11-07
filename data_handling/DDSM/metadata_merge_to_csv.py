import pandas as pd

# Define the source folder
source_folder = '/mnt/c/Datasets/DDSM/'

# Load the CSVs from the source folder
ddsm_description_df = pd.read_csv(f"{source_folder}ddsm_description.csv")
metadata_df = pd.read_csv(f"{source_folder}metadata.csv")

# Create a mapping from Subject ID to File Location
subject_to_file_location = metadata_df[['Subject ID', 'File Location']].set_index('Subject ID').to_dict()['File Location']

# Extract Subject ID from the image file path (the first folder in the path)
ddsm_description_df['Subject ID'] = ddsm_description_df['image file path'].str.split('/').str[0]

# Match the File Location from metadata.csv
ddsm_description_df['File Location'] = ddsm_description_df['Subject ID'].map(subject_to_file_location)

# Save the combined CSV in the source folder
output_csv = f"{source_folder}ddsm_metadata.csv"
ddsm_description_df.to_csv(output_csv, index=False)

print(f"Combined CSV saved to: {output_csv}")
