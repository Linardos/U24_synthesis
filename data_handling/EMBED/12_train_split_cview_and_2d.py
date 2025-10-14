import os
from pathlib import Path
import pandas as pd
import shutil

# USER: edit these two paths before running
CSV_PATH = Path("/mnt/d/Datasets/EMBED/tables/EMBED_OpenData_metadata_reduced.csv")
train_ROOT = Path("/mnt/d/Datasets/EMBED/EMBED_binary_256x256/train/original")

def load_uid_map(csv_path: Path):
    df = pd.read_csv(csv_path, low_memory=False)
    uids = df["anon_dicom_path"].astype(str).map(lambda p: os.path.splitext(os.path.basename(p))[0])
    final_type = df["FinalImageType"].astype(str).str.lower().str.strip()
    final_type = final_type.map(lambda s: "2d" if "2d" in s else ("cview" if "cview" in s else s))
    uid_map = {uid: ftype for uid, ftype in zip(uids, final_type)}
    return uid_map

def scan_train_set(train_root: Path):
    found = []
    for label in ["benign", "malignant"]:
        label_dir = train_root / label
        if not label_dir.is_dir():
            continue
        for uid_dir in label_dir.iterdir():
            if uid_dir.is_dir():
                uid = uid_dir.name
                nii = uid_dir / "slice.nii.gz"
                found.append((uid, label, nii))
    return found

def ensure_dirs(base: Path):
    for sub in ["benign", "malignant"]:
        (base / sub).mkdir(parents=True, exist_ok=True)

def main():
    uid_map = load_uid_map(CSV_PATH)
    samples = scan_train_set(train_ROOT)

    out_cview = Path(str(train_ROOT) + "_cview")
    out_2d = Path(str(train_ROOT) + "_2d")
    ensure_dirs(out_cview)
    ensure_dirs(out_2d)

    counts = {
        ("malignant", "cview"): 0,
        ("benign", "cview"): 0,
        ("malignant", "2d"): 0,
        ("benign", "2d"): 0,
    }

    for uid, label, nii in samples:
        ftype = uid_map.get(uid)
        if ftype not in ("2d", "cview"):
            continue
        counts[(label, ftype)] += 1

        dest_base = out_cview if ftype == "cview" else out_2d
        dest_dir = dest_base / label / uid
        dest_dir.mkdir(parents=True, exist_ok=True)
        if nii.is_file():
            shutil.copy2(nii, dest_dir / nii.name)

    print("\n=== FinalImageType counts (train SET) ===")
    print(f"malignant cview : {counts[('malignant','cview')]}")
    print(f"benign    cview : {counts[('benign','cview')]}")
    print(f"malignant 2D    : {counts[('malignant','2d')]}")
    print(f"benign    2D    : {counts[('benign','2d')]}")
    print(f"\nOutput dirs:\n  {out_cview}\n  {out_2d}\n")

if __name__ == "__main__":
    main()