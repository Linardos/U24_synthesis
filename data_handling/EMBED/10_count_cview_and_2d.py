import pandas as pd
from pathlib import Path

# USER: edit path before running
CSV_PATH = Path("/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_metadata_reduced.csv")

birads_map = {
    "N": "benign",
    "B": "benign",
    "M": "malignant",
    "K": "malignant",
}

def main():
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Normalize asses → benign/malignant, drop others
    df["label"] = df["asses"].map(birads_map)
    df = df.dropna(subset=["label"])

    # Normalize FinalImageType → "2d" / "cview" / ignore others
    ftype = df["FinalImageType"].astype(str).str.lower().str.strip()
    ftype = ftype.map(lambda s: "2d" if "2d" in s else ("cview" if "cview" in s else None))
    df["ftype"] = ftype
    df = df.dropna(subset=["ftype"])

    # Count by label + ftype
    counts = df.groupby(["label", "ftype"]).size().to_dict()

    print("\n=== FinalImageType counts (from CSV) ===")
    for key in [("malignant", "cview"), ("benign", "cview"),
                ("malignant", "2d"), ("benign", "2d")]:
        print(f"{key[0]:9s} {key[1]:5s} : {counts.get(key, 0)}")

if __name__ == "__main__":
    main()
