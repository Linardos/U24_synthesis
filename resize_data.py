import os
import nibabel as nib
from scipy.ndimage import zoom
import numpy as np

# Define paths
input_dir = '/mnt/c/Datasets/BCS-DBT-Szymon/original'
output_dir = '/mnt/c/Datasets/BCS-DBT-Szymon/resized_original'

# Desired output size for 2D images
output_size = (512, 512)

# Ensure output directory structure mirrors the input directory structure
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each category (abnormal, healthy)
for category in ['abnormal', 'healthy']:
    category_input_dir = os.path.join(input_dir, category)
    category_output_dir = os.path.join(output_dir, category)
    
    if not os.path.exists(category_output_dir):
        os.makedirs(category_output_dir)
    
    # Loop through each subject directory
    for subject in os.listdir(category_input_dir):
        subject_input_dir = os.path.join(category_input_dir, subject)
        subject_output_dir = os.path.join(category_output_dir, subject)
        
        if not os.path.exists(subject_output_dir):
            os.makedirs(subject_output_dir)
        
        # Define the path to the slice.nii.gz file
        input_file = os.path.join(subject_input_dir, 'slice.nii.gz')
        output_file = os.path.join(subject_output_dir, 'slice.nii.gz')
        
        # Load the image
        img = nib.load(input_file)
        img_data = img.get_fdata()

        # Check if the image is 2D
        if img_data.ndim == 2:
            # Calculate the zoom factor for each dimension
            zoom_factors = [output_size[i] / img_data.shape[i] for i in range(2)]
            
            # Resample the image
            resized_img_data = zoom(img_data, zoom_factors, order=3)  # using cubic interpolation
            
            # Convert back to nibabel object
            resized_img = nib.Nifti1Image(resized_img_data, img.affine)

            # Save the resized image
            nib.save(resized_img, output_file)
            
            print(f"Resized and saved {output_file}")
        else:
            print(f"Skipping non-2D image: {input_file}")
