#!/usr/bin/env python3
# make_balanced_test_split.py
# ------------------------------------------------------------------
# Hard-coded settings – change these four lines only
SOURCE_DIR     = "/mnt/d/Datasets/EMBED/EMBED_binary_12vs56_256x256/test"
DEST_DIR       = "/mnt/d/Datasets/EMBED/EMBED_binary_12vs56_256x256/test_balanced"
N_PER_CLASS    = 300          # folders to copy *for each* class
SEED           = 42
# ------------------------------------------------------------------

import os, shutil, random
from pathlib import Path
from tqdm import tqdm

def copy_subset(src: Path, dst: Path, n: int, seed: int = 0):
    """
    Copy `n` random sub-directories from `src` to `dst`.
    """
    folders = [p for p in src.iterdir() if p.is_dir()]
    if len(folders) < n:
        raise ValueError(f"{src} has only {len(folders)} folders (< {n})")

    random.seed(seed)
    selected = random.sample(folders, n)

    dst.mkdir(parents=True, exist_ok=True)
    for folder in tqdm(selected, desc=f"Copying {src.name}", unit="dir"):
        shutil.copytree(folder, dst / folder.name, dirs_exist_ok=True)

def main():
    src_root = Path(SOURCE_DIR)
    dst_root = Path(DEST_DIR)

    for cls in ("benign", "malignant"):
        src_cls = src_root / cls
        dst_cls = dst_root / cls

        if not src_cls.is_dir():
            print(f"✗ missing class folder: {src_cls}")
            continue

        copy_subset(src_cls, dst_cls, N_PER_CLASS, SEED)

    print("✅ Balanced split created at", dst_root)

if __name__ == "__main__":
    main()