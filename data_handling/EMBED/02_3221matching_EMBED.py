#!/usr/bin/env python3
import os, random, shutil
from tqdm import tqdm

def downsample_3221_inplace(source_base_dir, malignant_num, *, dry_run=False, seed=42):
    """
    Trim BENIGN / PROBABLY_BENIGN / SUSPICIOUS folders in `source_base_dir`
    so their counts match the 3 : 2 : 2 : 1 ratio w.r.t. `malignant_num`.

    Parameters
    ----------
    source_base_dir : str
        Path that contains the four category sub-folders.
        (e.g. ".../train/original" or ".../test")
    malignant_num   : int
        Number of malignant cases already present in that split.
    dry_run         : bool, optional
        If True, only print what would be removed; don't delete anything.
    seed            : int, optional
        RNG seed for reproducibility.
    """
    random.seed(seed)

    targets = {
        "benign":          3 * malignant_num,
        "probably_benign": 2 * malignant_num,
        "suspicious":      2 * malignant_num,
        "malignant":       None,                  # keep all
    }

    for cat, cap in targets.items():
        cat_dir = os.path.join(source_base_dir, cat)
        if not os.path.isdir(cat_dir):
            print(f"⚠️  Skipping '{cat_dir}' (not found)")
            continue

        folders = [f for f in os.listdir(cat_dir)
                   if os.path.isdir(os.path.join(cat_dir, f))]
        n_now = len(folders)

        if cap is None or n_now <= cap:
            print(f"[{cat:<16}] keep {n_now:>5} / {n_now:>5}")
            continue

        n_remove = n_now - cap
        to_remove = random.sample(folders, n_remove)

        print(f"[{cat:<16}] current {n_now:>5} remove {n_remove:>5} → target {cap:>5}")

        for folder in tqdm(to_remove, desc=f"Deleting {cat}", unit="folder"):
            path = os.path.join(cat_dir, folder)
            if dry_run:
                tqdm.write(f"DRY-RUN  would remove {path}")
            else:
                shutil.rmtree(path, ignore_errors=True)

    print("✅ Down-sampling complete.")

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
RESIZE_DIM = 512
EMBED_ds = f"EMBED_clean_{RESIZE_DIM}x{RESIZE_DIM}"

# TRAIN split
train_dir = f"/mnt/d/Datasets/EMBED/{EMBED_ds}/train/original"
downsample_3221_inplace(train_dir, malignant_num=1148, dry_run=False)

# TEST split
test_dir  = f"/mnt/d/Datasets/EMBED/{EMBED_ds}/test"
downsample_3221_inplace(test_dir,  malignant_num=324,  dry_run=False)
