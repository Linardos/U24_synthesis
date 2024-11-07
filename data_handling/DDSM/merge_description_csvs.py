import pandas as pd
import os

# Define the input files and corresponding prefixes
input_files = [
    'mass_case_description_train_set.csv',
    'calc_case_description_train_set.csv',
    'mass_case_description_test_set.csv',
    'calc_case_description_test_set.csv'
]

# Define the source folder for the file paths
source_folder = '/mnt/c/Datasets/DDSM'

# Read each file and concatenate into a single DataFrame
ddsm_description = pd.concat([pd.read_csv(os.path.join(source_folder, file)) for file in input_files], ignore_index=True)

# Save the combined DataFrame to a new CSV in the source folder
ddsm_description.to_csv(os.path.join(source_folder, 'ddsm_description.csv'), index=False)
