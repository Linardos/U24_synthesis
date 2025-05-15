import pandas as pd

# Load the benign CSV
input_path = "/mnt/d/Datasets/TMIST/all_TMIST_radiology.csv"

# Read the CSV
df = pd.read_csv(input_path)

# -------------------
# Count unique rows per label
# -------------------

# Helper function to detect if a label appears in either column
def label_in_row(row, label):
    return (row['left_birads'] == label) or (row['right_birads'] == label)

# Initialize a dictionary to hold counts
label_counts = {}


all_birads = {'1','2','3', '4', '5', '6', '4A', '4B', '4C'}
# Check each label
for label in all_birads:
    label_counts[label] = df.apply(lambda row: label_in_row(row, label), axis=1).sum()

print("\nLabel counts (unique per row):")
for label, count in sorted(label_counts.items()):
    print(f"BiRADs {label}: {count} samples")