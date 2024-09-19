import os
import yaml
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models import get_model  # Assuming get_model is defined in models.py
from data_loaders_S import NiftiDataset  # Assuming NiftiDataset is defined in data_loaders_S.py
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
num_epochs = config['num_epochs']
val_split = config['val_split']
model_names = config['model_names']
experiment_number = config['experiment_number']
resize_check = config['resize_check']

# Extract the final folder name from the data directory
final_folder = data_dir

# Store the config.yaml file in the current experiment folder
experiment_folder = f'{experiment_number:03d}_{model_names[0]}_{final_folder}'
experiments_folder = 'experiments'
experiment_path = os.path.join(experiments_folder, experiment_folder)

print(f"Starting experiment for {data_dir}, whose variables will be stored at: {experiment_path}")

# Create experiment folder if it doesn't exist
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

# Save config.yaml in the experiment folder
shutil.copy('config.yaml', experiment_path)

# Transform with random flipping
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
#     transforms.RandomVerticalFlip()     # Randomly flip images vertically
# ])

transform = None
# Load the dataset
print (f"loading data from {full_data_path}")
dataset = NiftiDataset(full_data_path=full_data_path, transform=transform)

print(f"Dataset contains {len(dataset)} samples.")

# Split dataset into training and validation sets
train_size = int((1 - val_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {name: get_model(name, pretrained=False) for name in model_names}

# Move models to device
for model in models.values():
    model.to(device)

criterion = nn.CrossEntropyLoss()
optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate) for name, model in models.items()}

metrics = {name: {'train_loss': [], 'train_acc': [], 'train_auc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []} for name in model_names}

metrics_file_name = os.path.join(experiment_path, f'{experiment_number:03d}_metrics.pkl')

# CSV log file path
csv_log_file = os.path.join(experiment_path, 'logs.csv')

# Write CSV header
with open(csv_log_file, mode='w', newline='') as csvfile:
    fieldnames = ['epoch', 'phase', 'model_name', 'loss', 'accuracy', 'AUC']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Training and evaluation loop
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        for model_name, model in models.items():
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                data_loader = val_loader

            running_loss = 0.0
            correct_preds = 0
            all_labels = []
            all_preds = []

            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                if phase == 'train':
                    optimizers[model_name].zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        # Backward pass and optimize
                        loss.backward()
                        optimizers[model_name].step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                correct_preds += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy()[:, 1])

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = correct_preds.double() / len(data_loader.dataset)
            epoch_auc = roc_auc_score(all_labels, all_preds)

            print(f'{model_name} - {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')

            # Store the metrics in the dictionary
            if phase == 'train':
                metrics[model_name]['train_loss'].append(epoch_loss)
                metrics[model_name]['train_acc'].append(epoch_acc.item())
                metrics[model_name]['train_auc'].append(epoch_auc)
            else:
                metrics[model_name]['val_loss'].append(epoch_loss)
                metrics[model_name]['val_acc'].append(epoch_acc.item())
                metrics[model_name]['val_auc'].append(epoch_auc)

            # Save metrics to pickle file
            with open(metrics_file_name, 'wb') as f:
                pickle.dump(metrics, f)

            # Log metrics to CSV file
            with open(csv_log_file, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'epoch': epoch + 1,
                    'phase': phase,
                    'model_name': model_name,
                    'loss': epoch_loss,
                    'accuracy': epoch_acc.item(),
                    'AUC': epoch_auc
                })

# Directory creation and saving models
for model_name, model in models.items():
    directory_name = os.path.join(experiment_path, f'{experiment_number:03d}_{model_name}_{final_folder}')
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    
    model_file_name = os.path.join(directory_name, f'{model_name}.pth')
    torch.save(model.state_dict(), model_file_name)

print("Training and evaluation complete. Models, metrics, and logs saved.")
