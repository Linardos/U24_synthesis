import os
import csv

# Define paths to your dataset folders
# dataset_root = "/mnt/c/Datasets/BCS-DBT-Szymon/original"
dataset_root = "/mnt/c/Datasets/BCS-DBT-Szymon/vqvae"
abnormal_root = os.path.join(dataset_root, "abnormal")
healthy_root = os.path.join(dataset_root, "healthy")

csv_file = "dataset.csv"
header = ["SubjectID", "Channel_0", "ValueToPredict"]

# Function to extract paths for each subject
def extract_subject_paths(root_path):
    subject_paths = []
    for subject_folder in os.listdir(root_path):
        subject_id = subject_folder  # Assuming subject_folder names are the IDs
        subject_folder_path = os.path.join(root_path, subject_folder)
        if "vqvae" in dataset_root:
            image_path = os.path.join(subject_folder_path, "slice_unclamped_reconstructed.nii.gz")
        elif "original" in dataset_root:
            image_path = os.path.join(subject_folder_path, "slice.nii.gz")
        subject_paths.append([subject_id, image_path])
    return subject_paths

# Get paths for abnormal and healthy subjects
abnormal_subjects = extract_subject_paths(abnormal_root)
healthy_subjects = extract_subject_paths(healthy_root)

# Function to determine ValueToPredict based on folder type
def determine_value_to_predict(subject_id):
    if os.path.exists(os.path.join(abnormal_root, subject_id)):
        return 1  # Abnormal
    elif os.path.exists(os.path.join(healthy_root, subject_id)):
        return 0  # Healthy
    else:
        return None

# Write to CSV
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    
    for subject_data in abnormal_subjects + healthy_subjects:
        subject_id = subject_data[0]
        image_path = subject_data[1]
        value_to_predict = determine_value_to_predict(subject_id)
        
        if value_to_predict is not None:
            row = [subject_id, image_path, value_to_predict]
            writer.writerow(row)
        else:
            print(f"Warning: Subject {subject_id} not found in abnormal or healthy folders.")

print(f"CSV file '{csv_file}' has been successfully created.")
