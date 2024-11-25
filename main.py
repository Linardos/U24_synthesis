import os
import yaml
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from models import get_model  # Assuming get_model is defined in models.py
from data_loaders import NiftiDataset  # Assuming NiftiDataset is defined in data_loaders_S.py
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import shutil
import csv  # Import the csv module

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract parameters from config
root_dir = config['root_dir']
data_dir = config['data_dir']
full_data_path = os.path.join(root_dir, data_dir)
batch_size = config['batch_size']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
num_epochs = config['num_epochs']
val_split = config['val_split']
model_names = config['model_names']
experiment_number = config['experiment_number']
experiment_name = config['experiment_name']
k_folds = config['k_folds']
store_sample_per_epoch = config['store_sample_per_epoch']
transform_check = config['transform_check']

early_stopping_patience = config['early_stopping_patience']
best_val_loss = float('inf')
no_improvement_epochs = 0

# Extract the final folder name from the data directory
final_folder = data_dir

# Store the config.yaml file in the current experiment folder
experiment_folder = f'{experiment_number:03d}__{experiment_name}_{final_folder}'
experiments_folder = 'experiments'
experiment_path = os.path.join(experiments_folder, experiment_folder)

print(f"Starting experiment for {data_dir}, whose variables will be stored at: {experiment_path}")

# Create experiment folder if it doesn't exist
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

# Save config.yaml in the experiment folder
shutil.copy('config.yaml', experiment_path)


# Transform with random flipping
if transform_check =='basic':
    transform = transforms.Compose([
        # transforms.Resize((128, 128)),  # Resize images to fixed dimensions (if needed)
        # transforms.ToTensor(), #scale to 0 1
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the grayscale channel
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
elif transform_check =='augmentations':
        transform = transforms.Compose([
        # transforms.Resize((128, 128)),  # Resize images to fixed dimensions (if needed)
        # transforms.ToTensor(), #scale to 0 1
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the grayscale channel
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),  # Add Gaussian noise
        transforms.RandomVerticalFlip()
    ])
else:
    print("Warning: no transformations are applied")
    transform = None

# Load the dataset
print(f"Loading data from {full_data_path}")
dataset = NiftiDataset(full_data_path=full_data_path, transform=transform)

print(f"Dataset contains {len(dataset)} samples.")

# Extract labels for stratified splitting and count the occurrences of each label for Data summary
labels = np.array([dataset[i][1] for i in range(len(dataset))])
unique_labels, label_counts = np.unique(labels, return_counts=True)
label_names = {0: 'Benign', 1: 'Malignant'}
label_summary = {label_names[label]: count for label, count in zip(unique_labels, label_counts)}
print("Dataset Summary:")
for label, count in tqdm(label_summary.items()):
    print(f"{label}: {count}")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {name: get_model(name, pretrained=False) for name in model_names}

# Move models to device
for model in models.values():
    model.to(device)

criterion = nn.CrossEntropyLoss()
optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) for name, model in models.items()}
schedulers = {name: CosineAnnealingLR(optimizer, T_max=num_epochs) for name, optimizer in optimizers.items()}
# CSV log file path
csv_log_file = os.path.join(experiment_path, 'logs.csv')

# Write CSV header
with open(csv_log_file, mode='w', newline='') as csvfile:
    fieldnames = ['fold', 'epoch', 'phase', 'model_name', 'loss', 'accuracy', 'AUC']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# If k_folds is 0, perform holdout training; else, perform k-fold cross-validation
if k_folds == 0:
    # Holdout validation
    # train_size = int((1 - val_split) * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

    train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=val_split, stratify=labels, random_state=random_seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Training and evaluation loop
    for epoch in range(num_epochs):
        batch_saved = False  # Visualize images and save only the first batch in each epoch
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            for model_name, model in models.items():
                if phase == 'train':
                    model.train()
                    data_loader = train_loader
                else:
                    model.eval()
                    data_loader = val_loader

                running_loss = 0.0
                correct_preds = 0
                all_labels = []
                all_preds = []


                for inputs, labels in tqdm(data_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if phase == 'train':
                        optimizers[model_name].zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizers[model_name].step()

                    running_loss += loss.item() * inputs.size(0)
                    correct_preds += torch.sum(preds == labels.data)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy()[:, 1])

                    # Save a grid of images for visual inspection (one batch per epoch)
                    if not batch_saved:
                        malignant_images = inputs[labels == 1][:6]  # Select up to 6 malignant samples
                        benign_images = inputs[labels == 0][:6]     # Select up to 6 benign samples
                        
                        if len(malignant_images) == 6 and len(benign_images) == 6:
                            grid_images = torch.cat((benign_images, malignant_images), 0)  # Stack benign and malignant
                            grid = make_grid(grid_images, nrow=6, padding=2, normalize=True)

                            # Save grid as an image file
                            img_path = os.path.join(experiment_path, f'epoch_{epoch + 1}_batch.png')
                            save_image(grid, img_path)
                            print(f'Saved image grid for epoch {epoch + 1} to {img_path}')
                            batch_saved = True  # Only save the first batch of each epoch

                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = correct_preds.double() / len(data_loader.dataset)
                epoch_auc = roc_auc_score(all_labels, all_preds)

                print(f'{model_name} - {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')

                with open(csv_log_file, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'fold': 'Holdout',
                        'epoch': epoch + 1,
                        'phase': phase,
                        'model_name': model_name,
                        'loss': epoch_loss,
                        'accuracy': epoch_acc.item(),
                        'AUC': epoch_auc
                    })

                # Early stopping check
                if phase == 'val':
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        no_improvement_epochs = 0  # Reset counter if there’s improvement
                    else:
                        no_improvement_epochs += 1  # Increment counter if no improvement

        for scheduler in schedulers.values():
            scheduler.step()
        
        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping triggered. No improvement for {early_stopping_patience} epochs.")
            break

        if config['sanity_check']:
            # Sanity check with probability output
            print("Running a sanity check for predictions and labels with probabilities. Aim of this check is to overfit training data.")
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                # probs = outputs.softmax(dim=1).cpu().detach().numpy()
                probs = torch.sigmoid(outputs).cpu().detach().numpy()
                # binary_preds = (probs > 0.5).astype(int)
                print(f'Preds: {preds}, Labels: {labels.cpu().numpy()}, Probabilities: {probs}')
                # print(f'Preds: {preds.cpu().numpy()}, Labels: {labels.cpu().numpy()}, Probabilities: {probs}')
                break

else:
    # k-Fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{k_folds}')
        print('-' * 10)

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Training and evaluation loop
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')

            for phase in ['train', 'val']:
                for model_name, model in models.items():
                    if phase == 'train':
                        model.train()
                        data_loader = train_loader
                    else:
                        model.eval()
                        data_loader = val_loader

                    running_loss = 0.0
                    correct_preds = 0
                    all_labels = []
                    all_preds = []

                    for inputs, labels in tqdm(data_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        if phase == 'train':
                            optimizers[model_name].zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)

                            if phase == 'train':
                                loss.backward()
                                optimizers[model_name].step()

                        running_loss += loss.item() * inputs.size(0)
                        correct_preds += torch.sum(preds == labels.data)
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy()[:, 1])

                    epoch_loss = running_loss / len(data_loader.dataset)
                    epoch_acc = correct_preds.double() / len(data_loader.dataset)
                    epoch_auc = roc_auc_score(all_labels, all_preds)

                    print(f'{model_name} - {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')

                    with open(csv_log_file, mode='a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({
                            'fold': fold + 1,
                            'epoch': epoch + 1,
                            'phase': phase,
                            'model_name': model_name,
                            'loss': epoch_loss,
                            'accuracy': epoch_acc.item(),
                            'AUC': epoch_auc
                        })