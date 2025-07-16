import os, sys
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
# pass seed by command line if you want
if len(sys.argv) > 1:                         # a number was passed
    try:
        cli_seed = int(sys.argv[1])
        config["seed"] = cli_seed             # overwrite config value
        print(f"[INFO] Using CLI seed {cli_seed}")
    except ValueError:
        raise SystemExit("Seed must be an integer, e.g.  python main.py 42")
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
model_name = config["model_name"]        
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
experiment_folder = f"{next_tag}_{model_name}_{experiment_name}_seed{random_seed}_real_perc{real_percentage}"
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

import torchvision.transforms as T

train_tf = T.Compose([
    T.Lambda(min_max_normalization),
    # random augments only here
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

eval_tf = T.Compose([
    T.Lambda(min_max_normalization),
    T.Normalize(mean=[0.5], std=[0.5]),
])




# Load the dataset with transformations applied
# ------------- paths -------------
real_path  = os.path.join(root_dir, data_dir)
synth_path = os.path.join(root_dir, synth_data_dir)

print(f"Loading data from:\n  • {real_path}\n  • {synth_path}")

# ------------- choose dataset(s) -------------
# if real_percentage == 1.0:
#     dataset = NiftiDataset(full_data_path=real_path, transform=transform)
#     n_real, n_synth = len(dataset), 0

# elif real_percentage == 0.0:
#     dataset = NiftiDataset(full_data_path=synth_path, transform=transform)
#     n_real, n_synth = 0, len(dataset)

# else:
#     real_ds  = NiftiDataset(full_data_path=real_path,  transform=transform)
#     synth_ds = NiftiDataset(full_data_path=synth_path, transform=transform)

#     dataset = stratified_real_synth_mix(real_ds, synth_ds,
#                                         real_fraction=real_percentage,
#                                         seed=random_seed)

#     # The mix returns ConcatDataset([real_subset, synth_subset])
#     n_real  = len(dataset.datasets[0])
#     n_synth = len(dataset.datasets[1])

# ---------- get data labels once ----------
real_ds  = NiftiDataset(real_path,  train_tf)
synth_ds = NiftiDataset(synth_path, train_tf)   # augments OK on synth
val_ds   = NiftiDataset(real_path,  eval_tf)

# grab per-sample class ids
def ds_labels(ds):
    lab = []
    for _, y in DataLoader(ds, batch_size=1024, num_workers=8, shuffle=False):
        lab.extend(y.numpy())
    return np.asarray(lab)

real_labels  = ds_labels(real_ds)
synth_labels = ds_labels(synth_ds)

classes = np.unique(real_labels)
print(f"Real {len(real_ds)}  |  Synth {len(synth_ds)}")

#--- classification
label_names = {
    0: 'Benign',
    1: 'Malignant'
}
CLASS_NAMES = ["Benign", "Malignant"] #, "Suspicious"]
if config['num_classes'] >=3:
    CLASS_NAMES.append("Suspicious")
    label_names[2] = "Suspicious"
if config['num_classes'] == 4:
    CLASS_NAMES.append("Probably Benign")
    label_names[3] = "Probably Benign"
unique_labels, label_counts = np.unique(real_labels, return_counts=True)
label_summary = {label_names[l]: c for l, c in zip(unique_labels, label_counts)}

print("Dataset Summary:")
for label, count in label_summary.items():
    print(f"{label}: {count}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -------- model choice ----------------------------
model      = get_model(model_name, pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate, weight_decay=weight_decay)
scheduler  = CosineAnnealingLR(optimizer, T_max=num_epochs)

# CSV log file path
csv_log_file = experiment_path / "logs.csv"

# Write CSV header
with open(csv_log_file, mode='w', newline='') as csvfile:
    fieldnames = [
        'fold','epoch','phase','model_name',
        'loss','overall_accuracy','AUC','balanced_accuracy',
        'ACC_benign','ACC_malignant'
    ]
    if config['num_classes'] >= 3:
        fieldnames.append('ACC_suspicious')
    if config['num_classes'] == 4:
        fieldnames.append('ACC_prob_benign')

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# k-Fold cross-validation ------------

cv = StratifiedKFold(n_splits=k_folds,
                     shuffle=True, random_state=random_seed)
for fold, (train_real_idx, val_real_idx) in enumerate(
        cv.split(np.zeros(len(real_ds)), real_labels)):

    rng = np.random.default_rng(random_seed + fold)

    # ----- build mixed TRAIN subset -------------------------------
    synth_pool_by_cls = {c: np.where(synth_labels == c)[0] for c in classes}

    chosen_real, chosen_synth = [], []
    for c in classes:
        cls_idx = train_real_idx[real_labels[train_real_idx] == c]
        rng.shuffle(cls_idx)

        n_keep = int(round(len(cls_idx) * real_percentage))   # REAL_FRACTION
        n_need = len(cls_idx) - n_keep

        chosen_real.extend(cls_idx[:n_keep])

        if n_need and len(synth_pool_by_cls[c]):
            pick = rng.choice(synth_pool_by_cls[c], size=n_need, replace=False)
            chosen_synth.extend(pick)
        elif n_need:
            print(f"⚠️  class {c}: wanted {n_need} synth but pool empty.")

    train_subset = ConcatDataset([Subset(real_ds,  chosen_real),
                                  Subset(synth_ds, chosen_synth)])
    val_subset   = Subset(val_ds, val_real_idx)          # 100 % real

    #  PRINT split summary
    # print(f"\nFold {fold+1}/{k_folds}  │  "
    #       f"train  real={len(chosen_real):4d} / synth={len(chosen_synth):4d}   │  "
    #       f"val real={len(val_real_idx):4d}")

    print(f"Fold {fold+1}/{k_folds} │ "
      f"train  real={len(chosen_real)} / synth={len(chosen_synth)} "
      f"({100*len(chosen_real)/(len(chosen_real)+len(chosen_synth)):.1f}% real) │ "
      f"val real={len(val_real_idx)}")

    # save indices for reproducibility
    idx_json = {
        "train_real" : list(map(int, chosen_real)),
        "train_synth": list(map(int, chosen_synth)),
        "val_real"   : list(map(int, val_real_idx)),
    }
    (experiment_path / f"indices_fold{fold+1}.json").write_text(
        json.dumps(idx_json, indent=2))

    # # --------------------------------------------------------------
    '''
    StratifiedKFold keeps each clinical class balanced across the 5 folds;
    because we do the replacement inside each class after the split, class balance is preserved.
    '''

    # train_subset = Subset(dataset, train_idx)
    # val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    ckpt_dir = experiment_path / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_val_auc         = -float("inf")   
    best_val_loss        =  float("inf")   
    early_stop_metric    = "auc"           # or "loss"
    min_delta_auc        = 0.002
    no_improvement_epochs = 0


    # Training and evaluation loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            correct_preds = 0
            all_labels = []
            all_probs  = []           # ← continuous scores for AUC

            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)                      # logits
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct_preds += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy())
                #                     shape (B, num_classes)

            # ------------------------------------------------------------
            # epoch-level metrics
            y_true = np.asarray(all_labels)          # (N,)
            probs  = np.asarray(all_probs)           # (N, C)
            y_pred = probs.argmax(axis=1)

            cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
            tp = np.diag(cm)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            tn = cm.sum() - (tp + fp + fn)

            per_class_sensitivity = tp / (tp + fn + 1e-12)
            per_class_specificity = tn / (tn + fp + 1e-12)
            balanced_acc          = balanced_accuracy_score(y_true, y_pred)

            # ---- AUC -----------------------------------------------------
            if config['num_classes'] == 2:
                # use probability of the positive class (column 1)
                epoch_auc = roc_auc_score(y_true, probs[:, 1])
            else:
                epoch_auc = roc_auc_score(y_true, probs, multi_class='ovr')



            # -----------------------------------------------------------------------

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = correct_preds.double() / len(data_loader.dataset)


            print(f'{model_name} - {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')

            with open(csv_log_file, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                dict_to_log = {
                    'fold': fold+1,
                    'epoch': epoch+1,
                    'phase': phase,
                    'model_name': model_name,
                    'loss': epoch_loss,
                    'overall_accuracy': epoch_acc.item(),
                    'AUC': epoch_auc,
                    'balanced_accuracy': balanced_acc,
                    'ACC_benign':    per_class_sensitivity[0],
                    'ACC_malignant': per_class_sensitivity[1],
                }
                # append optional classes as above

                if config['num_classes'] > 2:
                    dict_to_log.update({
                        'ACC_suspicious': per_class_sensitivity[2],
                    })
                if config['num_classes'] == 4:
                    dict_to_log.update({
                        'ACC_prob_benign': per_class_sensitivity[3],
                    })
                writer.writerow(dict_to_log)


            if phase == 'val':
                print("\nPer-class accuracy:")
                for i, cls in enumerate(CLASS_NAMES):
                    print(f"   {cls:>15}: {per_class_sensitivity[i]:.3f}")

                improved = False
                if early_stop_metric == "auc":
                    if epoch_auc > best_val_auc + min_delta_auc:
                        best_val_auc = epoch_auc
                        improved = True
                else:  # loss
                    if epoch_loss < best_val_loss - 1e-4:
                        best_val_loss = epoch_loss
                        improved = True

                if improved:
                    cm_file = experiment_path / f"cm_fold{fold+1}_epoch{epoch+1}_{phase}.npy"
                    np.save(cm_file, cm)

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": epoch_loss,
                            "seed": random_seed,
                        },
                        ckpt_dir / f"best_{model_name}_fold{fold+1}.pt"
                    )
                    print(f"✓ saved new best checkpoint (loss {epoch_loss:.4f})")
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1

                    
        scheduler.step()

        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping triggered. No improvement for {early_stopping_patience} epochs.")
            break
