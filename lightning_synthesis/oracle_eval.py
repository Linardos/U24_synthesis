# ─────────────────────────────────────────────────────────────────────────────
#  evaluate_oracle.py – Oracle-only accuracy on held‑out real validation set
# ─────────────────────────────────────────────────────────────────────────────
"""
Usage
-----
$ python evaluate_oracle.py

This script reproduces the exact 10 % validation split and per‑class sampling
used in *evaluate.py*, but **only** measures the Oracle classifier’s accuracy
on real data (no diffusion model, no FID).  All randomness is seeded so results
are reproducible as long as the underlying dataset is unchanged.
"""

import os, csv, random, yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Normalize
from monai import transforms as mt

# ── project‑level imports (make repo root importable first) ────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]  # one level up from this file
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from data_loaders_l import NiftiSynthesisDataset  # noqa: E402  (after sys.path fix)
from models import get_model                       # noqa: E402

# ── CONFIGURATION ──────────────────────────────────────────────────────────
SEED        = 42          # reproducibility everywhere
RESOLUTION  = 256         # image side length (for transforms)
BATCH       = 4           # Oracle batch size (fp16 OK)
N_EVAL      = 300         # images sampled *per class* to evaluate
CONFIG_YAML = "config_l.yaml"  # same config file as before

# Paths ----------------------------------------------------------------------------------
DATA_ROOT = "/mnt/d/Datasets/EMBED/EMBED_binary_12vs56_256x256/test_balanced"

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

if cfg["num_classes"] == 2:
    ORACLE_CKPT = (
        "/home/locolinux2/U24_synthesis/experiments/"
        "062_resnet50_binary_12vs56_seed44_real_perc1.0/"
        "checkpoints/best_resnet50_fold3.pt"
    )
elif cfg["num_classes"] == 4:
    ORACLE_CKPT = (
        "/home/locolinux2/U24_synthesis/experiments/"
        "048_resnet50_four_class_pretrainedImagenet_frozenlayers_seed44_real_perc1.0/"
        "checkpoints/best_resnet50_fold5.pt"
    )
else:
    raise ValueError("Unsupported num_classes in config_l.yaml – expected 2 or 4.")

# ── REPRODUCIBLE RNG SETUP ─────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── DATA TRANSFORMS & DATASET ─────────────────────────────────────────────
real_tf = mt.Compose([
    mt.LoadImaged(keys=["image"], image_only=True),
    mt.SqueezeDimd(keys=["image"], dim=-1),
    mt.EnsureChannelFirstd(keys=["image"]),
    # z‑score by slice, then [0,1] scaling so Oracle sees similar ranges
    mt.Lambdad(keys=["image"], func=lambda img: (img - img.mean()) / (img.std() + 1e-8)),
    mt.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    mt.ToTensord(keys=["image"]),
])

full_ds = NiftiSynthesisDataset(DATA_ROOT, transform=real_tf)

# ── BUILD 10 % VALIDATION SPLIT + PER‑CLASS SUBSETS ──────────────────────
g = torch.Generator().manual_seed(SEED)
val_len  = int(len(full_ds) * 0.1)
_, val_ds = torch.utils.data.random_split(full_ds, [len(full_ds) - val_len, val_len], generator=g)

# Category names must match training config to map label‑ids → strings
categories: List[str] = ["benign", "malignant"]
if cfg["num_classes"] >= 3:
    categories.append("probably_benign")
if cfg["num_classes"] == 4:
    categories.append("suspicious")

# Collect indices by class
idx_by_class: Dict[str, List[int]] = {c: [] for c in categories}
for idx in val_ds.indices:  # these are indices w.r.t. full_ds
    _, lbl = full_ds[idx]
    idx_by_class[categories[lbl]].append(idx)

# Sub‑sample exactly N_EVAL per class (shuffled deterministically)
rng = np.random.default_rng(SEED)
for c in categories:
    rng.shuffle(idx_by_class[c])
    idx_by_class[c] = idx_by_class[c][:N_EVAL]

val_loaders = {
    c: DataLoader(
        Subset(full_ds, idx_by_class[c]),
        batch_size=BATCH,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
    )
    for c in categories
}

# ── LOAD ORACLE (fp16 + GPU) ──────────────────────────────────────────────
oracle = get_model("resnet50", num_classes=len(categories), pretrained=False)
ckpt = torch.load(ORACLE_CKPT, map_location="cpu")
oracle.load_state_dict(ckpt["model_state_dict"])
oracle = oracle.half().to(DEVICE).eval()

oracle_norm = Normalize(mean=[0.5], std=[0.5])  # maps [0,1] → [‑1,1]

# ── EVALUATION ────────────────────────────────────────────────────────────
acc_cls: Dict[str, float] = {}

with torch.no_grad(), torch.autocast("cuda", torch.float16):
    for label_id, c in enumerate(categories):
        total = correct = 0
        for imgs, _ in val_loaders[c]:
            imgs = imgs.to(DEVICE)
            logits = oracle(oracle_norm(imgs))
            preds = logits.argmax(dim=1)
            correct += (preds == label_id).sum().item()
            total   += len(imgs)
        acc_cls[c] = correct / total if total else 0.0

acc_cls["mean"] = sum(acc_cls.values()) / len(categories)

print("\nOracle ACC :", {k: f"{v:.3f}" for k, v in acc_cls.items()})

# ── LOGGING (CSV) ─────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"logs/oracle_metrics_{ts}.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"ACC_{c}" for c in categories] + ["ACC_mean"])
    writer.writerow([acc_cls[c] for c in categories] + [acc_cls["mean"]])

print(f"Saved results to {csv_path}\n")
