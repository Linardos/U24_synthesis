# trim_synthetic.py  –  keep exactly N samples / class
import os, shutil, re
import numpy as np
from pathlib import Path

# ── edit these two blocks ──────────────────────────────────────────────
SYN_ROOT = Path("/mnt/d/Datasets/EMBED/EMBED_clean_256x256/train/synthetic_guide5.0")

target_counts = {
    'benign':          3444,
    'probably_benign': 2296,
    'suspicious':      2296,
    'malignant':       1148,
}
# ───────────────────────────────────────────────────────────────────────

folder_re = re.compile(r"^\d{4}$")   # 0001 … 9999

def numbered_subdirs(dir_path: Path):
    """return list of Path objects whose name is 4-digit number"""
    return [d for d in dir_path.iterdir() if d.is_dir() and folder_re.match(d.name)]

for cls, keep_n in target_counts.items():
    cls_dir = SYN_ROOT / cls
    if not cls_dir.exists():
        print(f"⚠  {cls}: directory missing – skipped")
        continue

    subs = numbered_subdirs(cls_dir)
    n_now = len(subs)

    if n_now <= keep_n:
        print(f"{cls:16}  {n_now} ≤ {keep_n}  → nothing to delete")
        continue

    # pick excess folders to delete
    # Option 1 (delete highest indices first = newest):
    excess = sorted(subs, key=lambda p: int(p.name), reverse=True)[keep_n:]

    # Option 2 (delete random ones):
    # excess = list(np.random.permutation(subs)[keep_n:])

    print(f"{cls:16}  {n_now} → {keep_n}   deleting {len(excess)} folders …")
    for p in excess:
        shutil.rmtree(p)

print("\n✅  Trimming finished.")
