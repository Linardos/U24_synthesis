import os
import shutil
import random

source_folder = '/mnt/c/Datasets/CMMD/CMMD_clean'
train_folder = os.path.join(source_folder, 'train')
test_folder = os.path.join(source_folder, 'test')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

classes = ['benign', 'malignant']
train_ratio = 0.9  # 90% for training, 10% for testing

# Process each class
for class_name in classes:
    class_path = os.path.join(source_folder, class_name)
    train_class_path = os.path.join(train_folder, class_name)
    test_class_path = os.path.join(test_folder, class_name)

    # Ensure directories for train and test subsets exist for each class
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)

    # Get all subdirectories in the current class directory
    samples = [sample for sample in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, sample))]
    
    # Shuffle and split the samples
    random.shuffle(samples)
    train_count = int(len(samples) * train_ratio)
    
    train_samples = samples[:train_count]
    test_samples = samples[train_count:]

    # Move train samples
    for sample in train_samples:
        src_path = os.path.join(class_path, sample)
        dst_path = os.path.join(train_class_path, sample)
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)

    # Move test samples
    for sample in test_samples:
        src_path = os.path.join(class_path, sample)
        dst_path = os.path.join(test_class_path, sample)
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)

    print(f"Moved {len(train_samples)} {class_name} samples to train and {len(test_samples)} to test.")

print("Stratified train-test split completed.")
