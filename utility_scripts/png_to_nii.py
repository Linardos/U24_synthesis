import os
import numpy as np
from PIL import Image
import nibabel as nib

# Define paths
input_dir = '/mnt/c/Datasets/BCS-DBT-Szymon/old_stylegan_256x256'
output_dir = '/mnt/c/Datasets/BCS-DBT-Szymon/stylegan_256x256'

# Ensure output directory structure mirrors the input directory structure
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each category (abnormal, healthy)
for category in ['abnormal', 'healthy']:
    category_input_dir = os.path.join(input_dir, category)
    category_output_dir = os.path.join(output_dir, category)
    
    if not os.path.exists(category_output_dir):
        os.makedirs(category_output_dir)
    
    # Loop through each image in the category directory
    for image_name in os.listdir(category_input_dir):
        image_input_path = os.path.join(category_input_dir, image_name)
        
        # Load the image using PIL
        with Image.open(image_input_path) as img:
            # Convert the image to grayscale (or keep RGB if needed)
            img = img.convert('L')  # Convert to grayscale
            
            # Convert the image to a numpy array
            img_np = np.array(img)
            
            # Expand dimensions to create a single-slice 3D volume
            img_np = np.expand_dims(img_np, axis=-1)
            
            # Create a NIfTI image
            nifti_img = nib.Nifti1Image(img_np, affine=np.eye(4))
            
            # Define the output file path
            output_file_name = os.path.splitext(image_name)[0] + '.nii.gz'
            image_output_path = os.path.join(category_output_dir, output_file_name)
            
            # Save the NIfTI image
            nib.save(nifti_img, image_output_path)
            
            print(f"Converted and saved {image_output_path}")
