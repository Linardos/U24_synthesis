import pandas as pd

input_path1 = "/mnt/d/Datasets/TMIST/parsed_all.csv"
input_path2 = "/mnt/d/Datasets/TMIST/parsed_malignant.csv"
output_path = "/mnt/d/Datasets/TMIST/all_TMIST_radiology.csv"

# read both files; pandas automatically uses the first line as the header
df1 = pd.read_csv(input_path1)
df2 = pd.read_csv(input_path2)

# append rows of df2 to df1
combined = pd.concat([df1, df2], ignore_index=True)

# save the result (header written once)
combined.to_csv(output_path, index=False)

print(f"Combined CSV saved to: {output_path}")
