import os
import csv
from tqdm import tqdm

# Paths and labels
root_path = "/mnt/d/Datasets/EMBED/EMBED_clean/train/original"
output_csv = "dataset_summary.csv"
label_mapping = {
    "benign": 0,
    "malignant": 1,
    "probably_benign": 2,
    "suspicious": 3
}

# Prepare CSV data
csv_data = []

for class_folder_name, label in tqdm(label_mapping.items()):
    class_folder_path = os.path.join(root_path, class_folder_name)
    
    if not os.path.isdir(class_folder_path):
        print(f"Skipping {class_folder_name} as it's not a directory.")
        continue
    
    for subfolder in tqdm(os.listdir(class_folder_path)):
        subfolder_path = os.path.join(class_folder_path, subfolder)
        
        # Check if it is a folder and contains `slice.nii.gz`
        if os.path.isdir(subfolder_path):
            slice_path = os.path.join(subfolder_path, "slice.nii.gz")
            
            # Check if the `slice.nii.gz` file exists
            if os.path.exists(slice_path):
                csv_data.append([
                    slice_path,  # Channel_0 (single-channel input)
                    label,       # Label
                    class_folder_name  # LabelMapping
                ])
            else:
                print(f"Missing `slice.nii.gz` in {subfolder_path}. Skipping.")

# Write to CSV
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(["Channel_0", "Label", "LabelMapping"])
    # Write data
    writer.writerows(csv_data)

print(f"CSV file created: {output_csv}")
