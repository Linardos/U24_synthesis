import pandas as pd

# Load the CSV file
input_file = 'original_256_train_data.csv'
data = pd.read_csv(input_file)

# Calculate the number of rows
total_rows = len(data)

# Calculate the split indices
split_90_index = int(0.9 * total_rows)
split_5_index = int(0.05 * total_rows)

# Split the data
train_90 = data.iloc[:split_90_index]  # First 90%
val_90 = pd.concat([data.iloc[:split_5_index], data.iloc[-split_5_index:]])  # First 5% + Last 5% to take equal from both labels

# Save the data into two CSV files
train_90.to_csv('0.9_train_split.csv', index=False)
val_90.to_csv('0.1_val_split_holdout.csv', index=False)

print("Splitting completed.")
