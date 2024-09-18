import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the root directory (WSL format)
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)

class NiftiDataset(Dataset):
    def __init__(self, full_data_path, transform=None):
        self.full_data_path =full_data_path 
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for label in [['healthy',0], ['abnormal',1]]:
            class_dir = os.path.join(self.full_data_path, label[0])
            for subdir in os.listdir(class_dir):
                file_path = os.path.join(class_dir, subdir, 'slice.nii.gz')
                if os.path.exists(file_path):
                    samples.append((file_path, label[1]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        nifti_img = nib.load(file_path)
        img_array = nifti_img.get_fdata()
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a standard size (you can adjust this)
])

dataset = NiftiDataset(full_data_path=full_data_path, transform=transform)

# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for inputs, labels in tqdm(train_loader):
    # print(f'Batch of {len(inputs)} images loaded with labels: {labels}')