#!/usr/bin/env python3
# ------------------------------------------------------------
#  Count benign vs malignant BEFORE any subsampling/splitting
#  (uses the same mapping/filters as your builder script)
# ------------------------------------------------------------
import pandas as pd

# --- Config -------------------------------------------------
csv_path = "/mnt/d/Datasets/EMBED/tables/EMBED_cleaned_metadata.csv"

# --- Load & map --------------------------------------------
data = pd.read_csv(csv_path)

birads_map = {
    "N": "benign",
    "B": "benign",
    # "P": "probably_benign",
    # "S": "malignant",
    "M": "malignant",
    "K": "malignant",
}

# Map BI-RADS → coarse category labels
data["category"] = data["asses"].map(birads_map).fillna("unknown")

# Apply the same filters you used elsewhere
#  • drop unknown category rows
#  • drop magnification views
filtered = data[(data["category"] != "unknown") & (data["spot_mag"] != 1)]

# --- Report -------------------------------------------------
print("=== Counts BEFORE any subsampling/splitting ===")
print(filtered["category"].value_counts().to_string())

print("\nTotal (benign + malignant):", len(filtered))

# Optional extra breakdowns (uncomment if useful):
# print("\nBy asses code (after filters):")
# print(filtered["asses"].value_counts().to_string())
#
# print("\nBy laterality (if column exists):")
# if "laterality" in filtered.columns:
#     print(filtered.groupby(["category", "laterality"]).size())
