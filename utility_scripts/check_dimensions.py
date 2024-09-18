import os
import nibabel as nib

# Define the input directory
input_dir = '/mnt/c/Datasets/BCS-DBT-Szymon/original'

# Loop through each category (abnormal, healthy)
for category in ['abnormal', 'healthy']:
    category_input_dir = os.path.join(input_dir, category)
    
    # Loop through each subject directory
    for subject in os.listdir(category_input_dir):
        subject_input_dir = os.path.join(category_input_dir, subject)
        
        # Define the path to the slice.nii.gz file
        input_file = os.path.join(subject_input_dir, 'slice.nii.gz')
        
        if os.path.isfile(input_file):
            # Load the image
            img = nib.load(input_file)
            img_data = img.get_fdata()
            
            # Print the dimensions
            print(f"{category}/{subject}/slice.nii.gz - Dimensions: {img_data.shape}")
        else:
            print(f"File not found: {input_file}")
