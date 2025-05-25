import os
import shutil, csv, pickle, yaml
import time, re, random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score

from scipy.interpolate import interp1d

from models import get_model  
from data_loaders import NiftiDataset, stratified_real_synth_mix 


# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set random seed for reproducibility
random_seed = config['seed']
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
# Extract parameters from config
root_dir = config['root_dir']
data_dir = config['data_dir']
synth_data_dir = config['synth_data_dir']
real_percentage = config.get('real_percentage', 1.0)

# full_data_path = os.path.join(root_dir, data_dir)
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

# Store the config.yaml file in the current experiment folder
# ── EXPERIMENT FOLDER ──────────────────────────────────────────────────────────

# 1) make sure the base directory exists
base_dir = Path("experiments")
base_dir.mkdir(exist_ok=True)

# 2) collect existing prefixes that match "000__something"
prefix_re = re.compile(r"^(\d{3})__")          # capture 3-digit prefix
prefixes  = [
    int(prefix_re.match(d.name).group(1))
    for d in base_dir.iterdir()
    if d.is_dir() and prefix_re.match(d.name)
]

# 3) choose the next number (start from 1 if folder is empty)
next_num = max(prefixes, default=0) + 1        # default=0 handles no experiments yet
next_tag = f"{next_num:03d}"                   # zero-pad to 3 digits

# 4) compose the folder name and create it
experiment_folder = f"{next_tag}_{model_names[0]}_{experiment_name}_seed{random_seed}_real_perc{real_percentage}"
experiment_path   = base_dir / experiment_folder
experiment_path.mkdir(exist_ok=False)          # error if duplicate

print(f"New experiment folder: {experiment_path}")

# Save config.yaml in the experiment folder
shutil.copy("config.yaml", experiment_path / "config.yaml")


# Histogram Standardization using a reference CDF
def histogram_standardization(img_tensor, ref_cdf, ref_bins):
    img_np = img_tensor.numpy().flatten()
    img_hist, img_bins = np.histogram(img_np, bins=256, range=(0, 1), density=True)
    img_cdf = np.cumsum(img_hist) / img_hist.sum()

    interp_values = np.interp(img_cdf, ref_cdf, ref_bins[:-1])
    img_standardized = np.interp(img_np, img_bins[:-1], interp_values)
    
    return torch.tensor(img_standardized.reshape(img_tensor.shape), dtype=torch.float32)

def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

# Transform with random flipping
if transform_check =='basic':

    transform = transforms.Compose([
        transforms.Lambda(lambda x: min_max_normalization(x)),  # Min-max normalization
        # transforms.Lambda(lambda x: histogram_standardization(x, ref_cdf, ref_bins)),  # Histogram Standardization
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the grayscale channel
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
elif transform_check =='augmentations':
        transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(degrees=15),
        transforms.Lambda(lambda x: min_max_normalization(x)),  # Min-max normalization,
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the grayscale channel
        transforms.Lambda(lambda x: apply_gaussian_denoise(x)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
else:
    print("Warning: no transformations are applied")
    transform = None

# Load the dataset with transformations applied
# ------------- paths -------------
real_path  = os.path.join(root_dir, data_dir)
synth_path = os.path.join(root_dir, synth_data_dir)

print(f"Loading data from:\n  • {real_path}\n  • {synth_path}")

# ------------- choose dataset(s) -------------
if real_percentage == 1.0:
    dataset = NiftiDataset(full_data_path=real_path, transform=transform)
    n_real, n_synth = len(dataset), 0

elif real_percentage == 0.0:
    dataset = NiftiDataset(full_data_path=synth_path, transform=transform)
    n_real, n_synth = 0, len(dataset)

else:
    real_ds  = NiftiDataset(full_data_path=real_path,  transform=transform)
    synth_ds = NiftiDataset(full_data_path=synth_path, transform=transform)

    dataset = stratified_real_synth_mix(real_ds, synth_ds,
                                        real_fraction=real_percentage,
                                        seed=random_seed)

    # The mix returns ConcatDataset([real_subset, synth_subset])
    n_real  = len(dataset.datasets[0])
    n_synth = len(dataset.datasets[1])

print(f"Final dataset: {len(dataset)}  (real {n_real}, synth {n_synth})")


labels = []
print("Processing for label summary...")
# for i in tqdm(range(0, len(dataset), 1000)):  # Process in chunks of 1000
#     start_time = time.time()
#     labels.extend([dataset[j][1] for j in range(i, min(i + 1000, len(dataset)))])
#     # print(f'Time for 1000 samples: {time.time() - start_time:.2f} seconds')
data_loader = DataLoader(dataset, batch_size=1000, num_workers=8, shuffle=False)
labels = []
for batch in tqdm(data_loader):
    labels.extend(batch[1].numpy())  # Assuming labels are the second item in the dataset
labels = np.array(labels)

unique_labels, label_counts = np.unique(labels, return_counts=True)

#--- Binary classification

# label_names = {0: 'Benign', 1: 'Malignant'}
# label_summary = {label_names[label]: count for label, count in zip(unique_labels, label_counts)}

#--- multi-classification
label_names = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Probably Benign',
    3: 'Suspicious'
}
label_summary = {label_names[label]: count for label, count in zip(unique_labels, label_counts)}


print("Dataset Summary:")
for label, count in label_summary.items():
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
csv_log_file = experiment_path / "logs.csv"

# Write CSV header
with open(csv_log_file, mode='w', newline='') as csvfile:
    fieldnames = [
        'fold', 'epoch', 'phase', 'model_name',
        'loss', 'accuracy', 'AUC', 'balanced_accuracy',
        'macro_precision', 'macro_recall', 'macro_f1'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# If k_folds is 0, perform holdout training; else, perform k-fold cross-validation
class_ratios = {0: 3, 2: 2, 3: 2, 1: 1}  # Benign: Probably Benign: Suspicious: Malignant
total_ratio = sum(class_ratios.values())

# Get indices for each class
class_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

# Compute total validation set size
total_val_size = int(len(dataset) * val_split)

# Compute how many samples per class should go to validation
val_class_sizes = {}
for label, indices in class_indices.items():
    if label == 1:  # Malignant, should be ~20% validation
        val_class_sizes[label] = int(len(indices) * 0.2)  # 20% of Malignant cases in validation
    else:  # Other classes follow 5:3:2
        val_class_sizes[label] = int((class_ratios[label] / total_ratio) * total_val_size)

train_indices, val_indices = [], []

# Stratified sampling for each class
for label, indices in class_indices.items():
    # Ensure validation set does not take too much from a class
    val_size = max(1, min(val_class_sizes[label], len(indices) - 1))

    # Convert val_size to fraction for train_test_split()
    test_fraction = val_size / len(indices) if len(indices) > 1 else 0.5

    # Split the class-specific data
    train_idx, val_idx = train_test_split(indices, test_size=test_fraction, random_state=random_seed)

    train_indices.extend(train_idx)
    val_indices.extend(val_idx)

# Convert to numpy arrays
train_indices = np.array(train_indices)
val_indices = np.array(val_indices)

# Create dataset subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Debugging Check: Print final label distribution
train_label_counts = np.bincount([labels[i] for i in train_indices])
val_label_counts = np.bincount([labels[i] for i in val_indices])

print("Training set distribution:", train_label_counts)
print("Validation set distribution:", val_label_counts)



def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    for model_name, model in models.items():
        model.load_state_dict(checkpoint['model_state_dict'][model_name])
    for model_name, optim in optimizers.items():
        optim.load_state_dict(checkpoint['optimizer_state_dict'][model_name])
    for model_name, sched in schedulers.items():
        sched.load_state_dict(checkpoint['scheduler_state_dict'][model_name])
    return checkpoint['epoch'], checkpoint['best_val_loss'], checkpoint['no_improvement_epochs']

# Load from checkpoint if exists
latest_checkpoint = os.path.join(experiment_path, 'latest_checkpoint.pth')
if os.path.exists(latest_checkpoint):
    start_epoch, best_val_loss, no_improvement_epochs = load_checkpoint(latest_checkpoint)
    print(f'Resuming training from epoch {start_epoch}')
else:
    start_epoch = 0
    best_val_loss = float('inf')
    no_improvement_epochs = 0


# Training and evaluation loop
for epoch in range(start_epoch, num_epochs):
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

                if config['num_classes'] > 2:
                    all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy())  # Store all class probabilities
                else:
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
            if config['num_classes'] > 2:

                epoch_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
            else:
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
    
    checkpoint_path = experiment_path / f"checkpoint_epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': {model_name: model.state_dict() for model_name, model in models.items()},
        'optimizer_state_dict': {model_name: optim.state_dict() for model_name, optim in optimizers.items()},
        'scheduler_state_dict': {model_name: sched.state_dict() for model_name, sched in schedulers.items()},
        'best_val_loss': best_val_loss,
        'no_improvement_epochs': no_improvement_epochs
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')


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