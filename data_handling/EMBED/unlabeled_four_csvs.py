import os
import csv
from tqdm import tqdm

### Due to GaNDLF-synth constraints, we generate four csvs one for each label, so we may use the "unlabeled" paradigm for each one separately

# Paths and labels
root_path = "/mnt/d/Datasets/EMBED/EMBED_clean/train/original"
label_mapping = {
    "benign": 0,
    "malignant": 1,
    "probably_benign": 2,
    "suspicious": 3
}

for class_folder_name in tqdm(label_mapping.keys()):
    class_folder_path = os.path.join(root_path, class_folder_name)
    output_csv = f"summary_{class_folder_name}.csv"
    
    if not os.path.isdir(class_folder_path):
        print(f"Skipping {class_folder_name} as it's not a directory.")
        continue

    csv_data = []
    
    for subfolder in tqdm(os.listdir(class_folder_path), desc=f"Processing {class_folder_name}"):
        subfolder_path = os.path.join(class_folder_path, subfolder)
        
        # Check if it is a folder and contains `slice.nii.gz`
        if os.path.isdir(subfolder_path):
            slice_path = os.path.join(subfolder_path, "slice.nii.gz")
            
            # Check if the `slice.nii.gz` file exists
            if os.path.exists(slice_path):
                csv_data.append([slice_path])  # Only Channel_0 is recorded
            else:
                print(f"Missing `slice.nii.gz` in {subfolder_path}. Skipping.")

    # Write to CSV
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Channel_0"])
        # Write data
        writer.writerows(csv_data)

    print(f"CSV file created: {output_csv}")
