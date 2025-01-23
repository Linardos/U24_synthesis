import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data
csv_file = './experiments/032__sanity_EMBED_multi_class_original/logs.csv'  # Update with actual CSV file path
df = pd.read_csv(csv_file)

# Function to plot combined metrics for train and validation
def plot_combined_metric(metric, ylabel, title, save_filename):
    plt.figure(figsize=(10, 6))

    # Plot for each model, combining train and validation data
    for model in df['model_name'].unique():
        train_data = df[(df['model_name'] == model) & (df['phase'] == 'train')]
        val_data = df[(df['model_name'] == model) & (df['phase'] == 'val')]

        plt.plot(train_data['epoch'], train_data[metric], label=f'{model} - Train', linestyle='-')
        plt.plot(val_data['epoch'], val_data[metric], label=f'{model} - Val', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_filename)
    plt.close()

# Plot accuracy
plot_combined_metric('accuracy', 'Accuracy', 'Model Accuracy per Epoch', './plots/combined_accuracy_plot.png')

# Plot loss
plot_combined_metric('loss', 'Loss', 'Model Loss per Epoch', './plots/combined_loss_plot.png')

# Plot AUC
plot_combined_metric('AUC', 'AUC', 'Model AUC per Epoch', './plots/combined_auc_plot.png')

print("Plots saved successfully.")
