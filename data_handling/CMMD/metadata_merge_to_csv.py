import pandas as pd

# Step 1: Define the source folder path
source_folder = '/mnt/c/Datasets/CMMD/'

# Step 2: Read the CSV and Excel files from the source folder
metadata_df = pd.read_csv(f'{source_folder}metadata.csv')  # Path to metadata.csv
clinical_df = pd.read_excel(f'{source_folder}CMMD_clinicaldata_revision.xlsx')  # Path to CMMD_clinicaldata_revision.xlsx

# Step 3: Merge the datasets on 'Subject ID' (assuming 'Subject ID' is the column that matches in both files)
# We will assume 'Subject ID' exists in both datasets. If needed, modify column names accordingly.
merged_df = pd.merge(metadata_df[['File Location', 'Subject ID']], 
                     clinical_df, 
                     left_on='Subject ID', 
                     right_on='ID1', 
                     how='left')

# Step 4: Select the columns we want
# From metadata.csv: File Location and Subject ID
# From CMMD_clinicaldata_revision.xlsx: LeftRight, Age, number, abnormality, classification, subtype
final_df = merged_df[['File Location', 'Subject ID', 'LeftRight', 'Age', 'number', 'abnormality', 'classification', 'subtype']]

# Step 5: Export the combined dataframe to a new CSV file in the source folder
final_df.to_csv(f'{source_folder}combined_metadata.csv', index=False)

print("Data has been combined and saved to 'combined_metadata.csv'")
