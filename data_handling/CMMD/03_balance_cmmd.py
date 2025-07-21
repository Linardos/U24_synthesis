#!/usr/bin/env python
# balance_cmmd.py  ─ copy a class‑balanced subset of CMMD_clean_256x256
# ---------------------------------------------------------------------
import os, shutil, random
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------
# CONFIG  – adjust if your paths differ
# ---------------------------------------------------------------------
SOURCE_ROOT = Path("/mnt/d/Datasets/CMMD/CMMD_binary_256x256/train/original")
DEST_ROOT   = Path("/mnt/d/Datasets/CMMD/balanced_CMMD")      # → /CMMD/balanced_CMMD
RNG_SEED    = 42                                        # reproducible sampling

# ---------------------------------------------------------------------
def list_sample_dirs(class_dir: Path) -> List[Path]:
    """Return list of sub‑directories, one per slice (subject‑slice)."""
    return [p for p in class_dir.iterdir() if p.is_dir()]

def copy_sample(src_dir: Path, dst_dir: Path) -> None:
    """Copy an entire sample folder (subject‑slice) to balanced dataset."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)          # overwrite if stale
    shutil.copytree(src_dir, dst_dir)

def main() -> None:
    random.seed(RNG_SEED)

    # --- discover the two classes ------------------------------------------------
    class_dirs = {
        d.name.lower(): d
        for d in SOURCE_ROOT.iterdir()
        if d.is_dir() and d.name.lower() in {"benign", "malignant"}
    }
    if len(class_dirs) != 2:
        raise SystemExit("Expected exactly two folders: benign/ and malignant/")

    benign_samples    = list_sample_dirs(class_dirs["benign"])
    malignant_samples = list_sample_dirs(class_dirs["malignant"])

    n_benign, n_malignant = len(benign_samples), len(malignant_samples)
    n_target = min(n_benign, n_malignant)

    print(f"Found {n_benign} benign vs {n_malignant} malignant slices.")
    print(f"Creating a balanced set with {n_target} samples per class.")

    # --- pick a balanced subset --------------------------------------------------
    random.shuffle(benign_samples)
    random.shuffle(malignant_samples)

    keep_benign    = benign_samples[:n_target]
    keep_malignant = malignant_samples[:n_target]

    # --- copy to DEST_ROOT -------------------------------------------------------
    for cls, sample_dirs in [("benign", keep_benign), ("malignant", keep_malignant)]:
        for src_dir in sample_dirs:
            rel_path = src_dir.relative_to(SOURCE_ROOT)   # e.g. benign/123‑1
            dst_dir  = DEST_ROOT / rel_path
            copy_sample(src_dir, dst_dir)

    print(f"Done! Balanced dataset written to: {DEST_ROOT}")

if __name__ == "__main__":
    main()
