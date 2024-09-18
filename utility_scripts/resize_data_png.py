import os
from PIL import Image
import numpy as np

# Define paths
input_dir = '/mnt/c/Datasets/BCS-DBT-Szymon/stylegan'
output_dir = '/mnt/c/Datasets/BCS-DBT-Szymon/stylegan_256x256'

# Desired output size for 2D images
output_size = (256, 256)  

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
        image_output_path = os.path.join(category_output_dir, image_name)
        
        # Load the image
        with Image.open(image_input_path) as img:            
            # Resize the image
            resized_img = img.resize(output_size, Image.LANCZOS)  # LANCZOS is a high-quality downsampling filter
            
            # Save the resized image
            resized_img.save(image_output_path)
            
            print(f"Resized and saved {image_output_path}")
