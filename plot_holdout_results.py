import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data
csv_file = './experiments/001__holdout_original_256x256/logs.csv'  # Replace with your actual CSV file path
df = pd.read_csv(csv_file)

# Separate data for training and validation
df_train = df[df['phase'] == 'train']
df_val = df[df['phase'] == 'val']

# Function to plot training metrics
def plot_train_metric(metric, ylabel, title, save_filename):
    plt.figure(figsize=(10, 6))

    # Plot for each model
    for model in df_train['model_name'].unique():
        plt.plot(df_train[df_train['model_name'] == model]['epoch'], 
                 df_train[df_train['model_name'] == model][metric], label=model)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'Train {title}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_filename)

# Function to plot validation metrics
def plot_val_metric(metric, ylabel, title, save_filename):
    plt.figure(figsize=(10, 6))

    # Plot for each model
    for model in df_val['model_name'].unique():
        plt.plot(df_val[df_val['model_name'] == model]['epoch'], 
                 df_val[df_val['model_name'] == model][metric], label=model)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'Validation {title}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_filename)

# Plot accuracy
plot_train_metric('accuracy', 'Accuracy', 'Model Accuracy per Epoch', './plots/train_accuracy_plot.png')
plot_val_metric('accuracy', 'Accuracy', 'Model Accuracy per Epoch', './plots/val_accuracy_plot.png')

# Plot loss
plot_train_metric('loss', 'Loss', 'Model Loss per Epoch', './plots/train_loss_plot.png')
plot_val_metric('loss', 'Loss', 'Model Loss per Epoch', './plots/val_loss_plot.png')

# Plot AUC
plot_train_metric('AUC', 'AUC', 'Model AUC per Epoch', './plots/train_auc_plot.png')
plot_val_metric('AUC', 'AUC', 'Model AUC per Epoch', './plots/val_auc_plot.png')
