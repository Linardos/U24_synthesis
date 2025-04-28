import pandas as pd

# Load the malignant CSV
input_path = "/mnt/d/Datasets/TMIST/parsed_malignant.csv"
output_path = "/mnt/d/Datasets/TMIST/00_malignant.csv"

# Read the CSV
df = pd.read_csv(input_path)

# --- Force both columns to string type ---
df['left_birads'] = df['left_birads'].astype(str)
df['right_birads'] = df['right_birads'].astype(str)

# Define allowed BIRADS values (as strings)
allowed_birads = {'4', '5', '6', '4A', '4B', '4C'}

# Filter where either 'left_birads' or 'right_birads' is in the allowed set
filtered_df = df[
    df['left_birads'].isin(allowed_birads) | df['right_birads'].isin(allowed_birads)
]

# Save the filtered result
filtered_df.to_csv(output_path, index=False)

print(f"Filtered {len(filtered_df)} rows saved to {output_path}")

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
