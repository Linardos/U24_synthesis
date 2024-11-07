import os
import shutil
import pandas as pd
from tqdm import tqdm

# Define paths
abnormal_dir = "/home/locolinux2/datasets/BCS-DBT-Szymon/original_256x256/abnormal"
csv_file = "/home/locolinux2/datasets/BCS-DBT-Szymon/BCS-DBT-labels-train-v2.csv"  # Update with the correct path to your CSV file
output_dir = os.path.dirname(abnormal_dir)  # Parent directory for new folders

# Create new folders for each category
for category in ["actionable", "benign", "cancer"]:
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Loop through each row in the CSV with tqdm progress bar
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Patients"):
    patient_id = row['PatientID']
    actionable = row['Actionable']
    benign = row['Benign']
    cancer = row['Cancer']
    
    # Find all folders in the abnormal directory that match this PatientID
    patient_folders = [folder for folder in os.listdir(abnormal_dir) if folder.startswith(patient_id)]
    
    # Copy patient folders to the corresponding category folder based on the label
    if actionable == 1:
        for folder in tqdm(patient_folders, desc=f"Copying actionable folders for {patient_id}", leave=False):
            dest_path = os.path.join(output_dir, "actionable", folder)
            if not os.path.exists(dest_path):  # Check if folder already exists
                shutil.copytree(os.path.join(abnormal_dir, folder), dest_path)
    if benign == 1:
        for folder in tqdm(patient_folders, desc=f"Copying benign folders for {patient_id}", leave=False):
            dest_path = os.path.join(output_dir, "benign", folder)
            if not os.path.exists(dest_path):  # Check if folder already exists
                shutil.copytree(os.path.join(abnormal_dir, folder), dest_path)
    if cancer == 1:
        for folder in tqdm(patient_folders, desc=f"Copying cancer folders for {patient_id}", leave=False):
            dest_path = os.path.join(output_dir, "cancer", folder)
            if not os.path.exists(dest_path):  # Check if folder already exists
                shutil.copytree(os.path.join(abnormal_dir, folder), dest_path)
