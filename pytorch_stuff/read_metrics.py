import pickle
import os
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def read_metrics(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        metrics = pickle.load(file)
    return metrics

def main():
    # Load the config file to get the experiment number and other details
    config = load_config()
    experiment_number = config['experiment_number']
    data_dir = config['data_dir']

    # Construct the folder name and the pickle file path
    experiment_folder = "experiments/001_original/001_metrics_original"
    pickle_file_path = os.path.join(experiment_folder, 'metrics.pkl')

    # Check if the pickle file exists
    if not os.path.exists(pickle_file_path):
        print(f"Metrics file not found at {pickle_file_path}")
        return

    # Read the metrics from the pickle file
    metrics = read_metrics(pickle_file_path)

    # Display the metrics
    for i, x in enumerate(metrics.items()):
        model_name, model_metrics = x
        print(f"Metrics for {model_name}:")
        print(f"  Epoch {i + 1}:")
        print(f"    Training Loss: {model_metrics['train_loss']}")
        print(f"    Validation Loss: {model_metrics['val_loss']}")
        print(f"    Training Accuracy: {model_metrics['train_acc']}")
        print(f"    Validation Accuracy: {model_metrics['val_acc']}")
        print(f"    Training AUC: {model_metrics['train_auc']}")
        print(f"    Validation AUC: {model_metrics['val_auc']}")

if __name__ == "__main__":
    main()