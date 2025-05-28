import os
import shutil, csv, pickle, yaml, json
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

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

# # 2) collect existing prefixes that match "000__something"
# prefix_re = re.compile(r"^(\d{3})__")          # capture 3-digit prefix
# prefixes  = [
#     int(prefix_re.match(d.name).group(1))
#     for d in base_dir.iterdir()
#     if d.is_dir() and prefix_re.match(d.name)
# ]

# # 3) choose the next number (start from 1 if folder is empty)
# next_num = max(prefixes, default=0) + 1        # default=0 handles no experiments yet
next_tag = f"{experiment_number:03d}"                   # zero-pad to 3 digits

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
        # transforms.Lambda(lambda x: apply_gaussian_denoise(x)),
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
    2: 'Suspicious'
}
CLASS_NAMES = ["Benign", "Malignant", "Suspicious"]
if config['num_classes'] == 4:
    CLASS_NAMES.append("Probably Benign")
    label_names[3] = "Probably Benign"
label_summary = {label_names[label]: count for label, count in zip(unique_labels, label_counts)}

print("Dataset Summary:")
for label, count in label_summary.items():
    print(f"{label}: {count}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {name: get_model(name, pretrained=True) for name in model_names}

# Move models to device
for model in models.values():
    model.to(device)

# Loss with inverse class frequency weights
class_counts = np.array([label_summary[c] for c in CLASS_NAMES])
# (Benign, Malignant, Probably Benign, Suspicious)

# inverse-freq weights → more weight for rare classes
# weights = 1.0 / class_counts
# weights = weights / weights.sum() * len(class_counts)   # re-scale so avg = 1
# weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

# criterion = nn.CrossEntropyLoss(weight=weight_tensor)
criterion = nn.CrossEntropyLoss()
# optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) for name, model in models.items()}
optimizers = {
    name: torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    for name, model in models.items()
}

schedulers = {name: CosineAnnealingLR(optimizer, T_max=num_epochs) for name, optimizer in optimizers.items()}
# CSV log file path
csv_log_file = experiment_path / "logs.csv"

# Write CSV header
with open(csv_log_file, mode='w', newline='') as csvfile:
    fieldnames = [
        'fold','epoch','phase','model_name',
        'loss','accuracy','AUC','balanced_accuracy',
        'sensitivity_benign','sensitivity_malignant',
        'sensitivity_suspicious',
        'specificity_benign','specificity_malignant',
        'specificity_suspicious'
    ]

    if config['num_classes'] == 4:
        fieldnames.extend(['sensitivity_prob_benign','specificity_prob_benign'])


    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# k-Fold cross-validation ------------
# 0/1 flag for synthetic vs real
is_synth = np.zeros(len(dataset), dtype=int)
if isinstance(dataset, ConcatDataset):
    # ConcatDataset([real_subset, synth_subset])
    is_synth[len(dataset.datasets[0]):] = 1        # mark synthetic part

# composite label: class_id * 2 + is_synth
strat_key = labels * 2 + is_synth                 # values 0..(2*num_classes-1)

# stratification logic:
# | clinical class | real / synth | composite value |
# | -------------- | ------------ | --------------- |
# | Benign         | real (0)     | 0               |
# | Benign         | synth (1)    | 1               |
# | Malignant      | real (0)     | 2               |
# | Malignant      | synth (1)    | 3               |
# | ...            | …            | …               |
# Because each pair has its own code, StratifiedKFold will treat them as separate “classes” and keep all eight (4 clinical × 2 sources) in equal proportion in every fold.

# Dataset composition	is_synth vector	strat_key values produced	What StratifiedKFold balances
# All real	all 0	labels * 2 → 0, 2, 4, 6…	Only the clinical classes (Benign-real, Malignant-real, …). There are no “synthetic” codes, so folds stay class-balanced exactly as in the ordinary case.
# All synthetic	all 1	labels * 2 + 1 → 1, 3, 5, 7…	Again you get one unique code per clinical class, just offset by +1. Folds are still class-balanced; the “source” dimension is moot because every sample shares the same source.

kfold = StratifiedKFold(n_splits=k_folds,
                        shuffle=True,
                        random_state=random_seed)
# kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

# for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
    
for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(strat_key)), strat_key)):
    print(f'Fold {fold + 1}/{k_folds}')
    print('-' * 10)

    # ------ identify real vs synthetic indices in this fold -------
    if isinstance(dataset, ConcatDataset):
        split = len(dataset.datasets[0])
        real_train   = train_idx[train_idx <  split]
        synth_train  = train_idx[train_idx >= split]
        real_val     = val_idx[val_idx <  split]
        synth_val    = val_idx[val_idx >= split]
    else:  # pure real or pure synthetic
        real_train, synth_train = (train_idx, np.array([], int)) if n_real else (np.array([], int), train_idx)
        real_val,   synth_val   = (val_idx,   np.array([], int)) if n_real else (np.array([], int), val_idx)

    idx_info = {
        "train_real" : real_train.tolist(),
        "train_synth": synth_train.tolist(),
        "val_real"   : real_val.tolist(),
        "val_synth"  : synth_val.tolist(),
    }
    with open(experiment_path / f"indices_fold{fold+1}.json", "w") as f:
        json.dump(idx_info, f, indent=2)
    # --------------------------------------------------------------

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    no_improvement_epochs = 0

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
                    # all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy()[:, 1])
                    
                    if config['num_classes'] > 2:
                        all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy())  # Store all class probabilities
                    else:
                        all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy()[:, 1])
                        
                # turn probabilities → class index once, reuse everywhere
                y_true = np.array(all_labels)
                y_pred = np.argmax(all_preds, axis=1)

                cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
                tp = np.diag(cm)
                fp = cm.sum(axis=0) - tp
                fn = cm.sum(axis=1) - tp
                tn = cm.sum() - (tp + fp + fn)

                per_class_sensitivity  = tp / (tp + fn + 1e-12)
                per_class_specificity  = tn / (tn + fp + 1e-12)

                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                epoch_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')


                # -----------------------------------------------------------------------

                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = correct_preds.double() / len(data_loader.dataset)


                print(f'{model_name} - {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')

                with open(csv_log_file, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    dict_to_log = {
                        'fold': fold + 1,
                        'epoch': epoch + 1,
                        'phase': phase,
                        'model_name': model_name,
                        'loss': epoch_loss,
                        'accuracy': epoch_acc.item(),
                        'AUC': epoch_auc,
                        'balanced_accuracy': balanced_acc,
                        'sensitivity_benign':       per_class_sensitivity[0],
                        'sensitivity_malignant':    per_class_sensitivity[1],
                        'sensitivity_suspicious':   per_class_sensitivity[2],
                        'specificity_benign':       per_class_specificity[0],
                        'specificity_malignant':    per_class_specificity[1],
                        'specificity_suspicious':  per_class_specificity[2],
                    }

                    if config['num_classes'] == 4:
                        dict_to_log ['sensitivity_prob_benign'] = per_class_sensitivity[3]
                        dict_to_log ['specificity_prob_benign'] = per_class_specificity[3]
                    writer.writerow(dict_to_log)


                if phase == 'val':
                    print("\nPer-class sensitivity || specificity:")
                    for i, cls in enumerate(CLASS_NAMES):
                        print(f"   {cls:>15}: {per_class_sensitivity[i]:.3f} || {per_class_specificity[i]:.3f}")

                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        # store the confusion matrix, in case we need another metric after training
                        cm_file = experiment_path / f"cm_fold{fold+1}_epoch{epoch+1}_{phase}.npy"
                        np.save(cm_file, cm)

                        no_improvement_epochs = 0  # Reset counter if there’s improvement
                    else:
                        no_improvement_epochs += 1  # Increment counter if no improvement
                    
        for scheduler in schedulers.values():
            scheduler.step()

        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping triggered. No improvement for {early_stopping_patience} epochs.")
            break
