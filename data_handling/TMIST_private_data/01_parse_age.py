import pandas as pd

# Load the CSV
input_path = "/mnt/d/Datasets/TMIST/parsed_all.csv"
output_path = "/mnt/d/Datasets/TMIST/01_benign_age_pruned.csv"
# input_path = "/mnt/d/Datasets/TMIST/parsed_malignant.csv"
# output_path = "/mnt/d/Datasets/TMIST/01_malignant_age_pruned.csv"
# Read the CSV
df = pd.read_csv(input_path)

# Filter the DataFrame
filtered_df = df[(df['age'] >= 40) & (df['age'] <= 74)]

# Save the filtered result
filtered_df.to_csv(output_path, index=False)

print(f"Filtered {len(filtered_df)} rows saved to {output_path}")
