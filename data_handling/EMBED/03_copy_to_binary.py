#!/usr/bin/env python3
"""
Collapse a 4-class EMBED dataset into 2 classes
(benign / malignant) by **copying** the folders.

‣ Edit SRC_ROOT and DST_ROOT below, then:
      python collapse_to_binary.py
"""

from pathlib import Path
import shutil, sys

# ── EDIT THESE TWO LINES ONLY ────────────────────────────────
SRC_ROOT = Path("/mnt/d/Datasets/EMBED/EMBED_clean_256x256")
DST_ROOT = Path("/mnt/d/Datasets/EMBED/EMBED_clean_256x256_binary")
# ─────────────────────────────────────────────────────────────

MAP_4_TO_2 = {
    "benign":           "benign",
    "probably_benign":  "benign",
    "suspicious":       "malignant",
    "malignant":        "malignant",
}
SPLITS = ["train/original", "test"]          # add more if you have them

def copy_case(src_dir: Path, dst_dir: Path) -> None:
    """Copy an entire case directory (fast local copy)."""
    if dst_dir.exists():                              # avoid clobbering duplicates
        i = 1
        while (dst_dir.parent / f"{dst_dir.name}_{i}").exists():
            i += 1
        dst_dir = dst_dir.parent / f"{dst_dir.name}_{i}"
    shutil.copytree(src_dir, dst_dir)

def main() -> None:
    if not SRC_ROOT.is_dir():
        sys.exit(f"Source directory not found: {SRC_ROOT}")

    print(f"Source : {SRC_ROOT}")
    print(f"Dest   : {DST_ROOT}")
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for old_cls, new_cls in MAP_4_TO_2.items():
            src_cls_dir = SRC_ROOT / split / old_cls
            if not src_cls_dir.is_dir():
                continue

            dst_cls_dir_base = DST_ROOT / split / new_cls
            dst_cls_dir_base.mkdir(parents=True, exist_ok=True)

            for case_dir in src_cls_dir.iterdir():
                if case_dir.is_dir():
                    copy_case(case_dir, dst_cls_dir_base / case_dir.name)

            print(f"[{split.upper():5}] {old_cls:17} → {new_cls:10}  ✓")

    print("✅ Binary dataset ready.")

if __name__ == "__main__":
    main()
