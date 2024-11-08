import os
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from models import get_model
from data_loaders_S import NiftiDataset 
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import shutil
import csv 

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
num_epochs = config['num_epochs']
val_split = config['val_split']
model_names = config['model_names']
experiment_number = config['experiment_number']
experiment_name = config['experiment_name']
resize_check = config['resize_check']
k_folds = config['k_folds']

# Extract the final folder name from the data directory
final_folder = data_dir

# Store the config.yaml file in the current experiment folder
hold_or_cv = "holdout" if k_folds <= 0 else "cv"
experiment_folder = f'{experiment_number:03d}__{experiment_name}_{hold_or_cv}_{final_folder}'
experiments_folder = 'experiments'
experiment_path = os.path.join(experiments_folder, experiment_folder)

print(f"Starting experiment for {data_dir}, whose variables will be stored at: {experiment_path}")

# Create experiment folder if it doesn't exist
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

# Save config.yaml in the experiment folder
shutil.copy('config.yaml', experiment_path)

# Transform with random flipping
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
models = {name: get_model(name, pretrained=config['pretrained']) for name in model_names}

# Move models to device
for model in models.values():
    model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate) for name, model in models.items()}

# CSV log file path
csv_log_file = os.path.join(experiment_path, 'logs.csv')

# Write CSV header
with open(csv_log_file, mode='w', newline='') as csvfile:
    fieldnames = ['fold', 'epoch', 'phase', 'model_name', 'loss', 'accuracy', 'AUC', 'F1']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


# If k_folds is 0, perform holdout training; else, perform k-fold cross-validation
if k_folds <= 0:
    # Holdout validation
    
    train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=val_split, stratify=labels, random_state=random_seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # train_size = int((1 - val_split) * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training and evaluation loop
    for epoch in range(num_epochs):
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
                    labels = labels.to(device).float().unsqueeze(1)


                    if phase == 'train':
                        optimizers[model_name].zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # print(f"outputs: {outputs.shape}, dtype: {outputs.dtype}")
                        # print(f"labels: {labels.shape}, dtype: {labels.dtype}")
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizers[model_name].step()

                    running_loss += loss.item() * inputs.size(0)
                    correct_preds += torch.sum(preds == labels.data)  # Count correct predictions
                    total_preds += labels.size(0)  # Track total number of predictions
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
                    # all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy())


                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = correct_preds.float() / total_preds.float()
                epoch_auc = roc_auc_score(all_labels, all_preds)
                # epoch_auc = roc_auc_score(all_labels, all_preds[:, 1])  # assuming the second column represents positive class probabilities
                binary_preds = (np.array(all_preds) > 0.5).astype(int)
                epoch_f1 = f1_score(all_labels, binary_preds)

                print(f'{model_name} - {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f} F1: {epoch_f1:.4f}')

                with open(csv_log_file, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'fold': 'Holdout',
                        'epoch': epoch + 1,
                        'phase': phase,
                        'model_name': model_name,
                        'loss': epoch_loss,
                        'accuracy': epoch_acc.item(),
                        'AUC': epoch_auc,
                        'F1': epoch_f1
                    })

        if config['sanity_check']:
            # Sanity check with probability output
            print("Running a sanity check for predictions and labels with probabilities...")
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                # preds = torch.argmax(outputs, dim=1)
                # probs = outputs.softmax(dim=1).cpu().detach().numpy()
                probs = torch.sigmoid(outputs).cpu().detach().numpy()
                binary_preds = (probs > 0.5).astype(int)
                print(f'Preds: {binary_preds}, Labels: {labels.cpu().numpy()}, Probabilities: {probs}')
                # print(f'Preds: {preds.cpu().numpy()}, Labels: {labels.cpu().numpy()}, Probabilities: {probs}')



else:
    # k-Fold cross-validation
    # kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    # for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
    for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):
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
                        all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy())


                    epoch_loss = running_loss / len(data_loader.dataset)
                    epoch_acc = correct_preds.double() / len(data_loader.dataset)
                    # epoch_auc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
                    # epoch_f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average="macro")
                    epoch_auc = roc_auc_score(all_labels, all_preds)
                    # epoch_auc = roc_auc_score(all_labels, all_preds[:, 1])  # assuming the second column represents positive class probabilities
                    binary_preds = (all_preds > 0.5).astype(int)
                    epoch_f1 = f1_score(all_labels, binary_preds)
                    

                    print(f'{model_name} - {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f} F1: {epoch_f1:.4f}')

                    with open(csv_log_file, mode='a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({
                            'fold': fold + 1,
                            'epoch': epoch + 1,
                            'phase': phase,
                            'model_name': model_name,
                            'loss': epoch_loss,
                            'accuracy': epoch_acc.item(),
                            'AUC': epoch_auc,
                            'F1': epoch_f1
                        })