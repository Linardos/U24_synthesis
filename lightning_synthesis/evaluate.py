# ─────────────────────────────────────────────────────────────────────────────
#  evaluate.py – per-class and overall FID / MS-SSIM on the held-out val split
# ─────────────────────────────────────────────────────────────────────────────
import os, csv, random, math, json
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from data_loaders_l import NiftiSynthesisDataset
from model_architectures import MonaiDDPM
from monai import transforms as mt
from torchmetrics.image import (
    FrechetInceptionDistance as FID,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM, # after all doesn't make much sense to use this since it's unpaired data.
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# CKPT_PATH  = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/049_cDDPM_depth5_fixedScaling_256x256/checkpoints/epoch=22-step=7843.ckpt" # FID = 8.18 at guidance scale 3
CKPT_PATH  = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/053_DDPM__DataArtifactsRemoved___256x256/checkpoints/epoch=04-step=1435.ckpt"
CKPT_PATH  = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/054_DDPM_default512_256x256/checkpoints/epoch=04-step=1435.ckpt" # FID @ GS 0: 7.38
CKPT_PATH  = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/057_DDPM_seed2025_cropped_256x256/checkpoints/epoch=04-step=1435.ckpt"
CKPT_PATH = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/063_DDPM_contrast-aug-20percent_256x256/checkpoints/epoch=04-step=1435.ckpt"
CKPT_PATH = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/092_DDPM_MS-SSIM_10perc_HF_5perc_256x256/checkpoints/epoch=18-step=2736.ckpt" # ACTUAL GOLD

"""
GS @ 0
FID   : {'benign': '1.47', 'probably_benign': '2.95', 'suspicious': '3.06', 'malignant': '2.31', 'mean': '2.45'}
MS-SSIM: {'benign': '0.2246', 'probably_benign': '0.2250', 'suspicious': '0.2130', 'malignant': '0.2142', 'mean': '0.2192'}

GS @ 3
FID   : {'benign': '1.49', 'probably_benign': '0.93', 'suspicious': '0.60', 'malignant': '0.82', 'mean': '0.96'}
MS-SSIM: {'benign': '0.2244', 'probably_benign': '0.1999', 'suspicious': '0.2220', 'malignant': '0.2055', 'mean': '0.2129'}

GS @ 5
FID   : {'benign': '1.80', 'probably_benign': '0.85', 'suspicious': '1.27', 'malignant': '1.98', 'mean': '1.48'}
MS-SSIM: {'benign': '0.2195', 'probably_benign': '0.1874', 'suspicious': '0.1755', 'malignant': '0.2145', 'mean': '0.1992'}

GS @ 6
FID   : {'benign': '1.55', 'probably_benign': '1.62', 'suspicious': '1.63', 'malignant': '2.65', 'mean': '1.86'}
MS-SSIM: {'benign': '0.1956', 'probably_benign': '0.2192', 'suspicious': '0.2286', 'malignant': '0.1884', 'mean': '0.2080'}

GS @ 7
FID   : {'benign': '1.62', 'probably_benign': '1.36', 'suspicious': '2.29', 'malignant': '3.90', 'mean': '2.29'}
MS-SSIM: {'benign': '0.2233', 'probably_benign': '0.1891', 'suspicious': '0.2239', 'malignant': '0.1671', 'mean': '0.2009'}
"""

# CKPT_PATH = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/094_DDPM_MS-SSIM_10perc_HF_5perc_val/checkpoints/epoch=12-step=1690.ckpt" # ACTUAL GOLD

print(f"Evaluating {CKPT_PATH}")
NAME_TAG = "003_10percMS-SSIM_5percHF_thebest"

RESOLUTION = 256
BATCH      = 16
N_EVAL     = 100                       # samples / class
SCALES     = [0, 3, 5, 6, 7]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); random.seed(42); np.random.seed(42)

# ── LOAD MODEL (fp16) ─────────────────────────────────────────────────────────
model = (MonaiDDPM
         .load_from_checkpoint(CKPT_PATH, map_location="cpu")
         .half().to(device).eval())

# ── DATASETS ────────────────────────────────────────────────────────────────
root_dir   = "/mnt/d/Datasets/EMBED/EMBED_clean_256x256/train/original"
categories = ["benign", "probably_benign", "suspicious", "malignant"]

real_tf = mt.Compose([
    mt.LoadImaged(keys=["image"], image_only=True),
    mt.SqueezeDimd(keys=["image"], dim=-1),
    mt.EnsureChannelFirstd(keys=["image"]),
    mt.Lambdad(keys=["image"],
               func=lambda img: (img - img.mean()) / (img.std() + 1e-8)),
    mt.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    mt.ToTensord(keys=["image"]),
])

full_ds  = NiftiSynthesisDataset(root_dir, transform=real_tf)

# ---------- reproducible 10 % val split then 128-per-class subset ----------
g = torch.Generator().manual_seed(42)
val_len  = int(len(full_ds) * 0.1)
train_len = len(full_ds) - val_len
_, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len], generator=g)

# build class-specific index lists
idx_by_class = {c: [] for c in categories}
for idx in val_ds.indices:                           # original indices
    _, label = full_ds[idx]
    idx_by_class[categories[label]].append(idx)

rng = np.random.default_rng(42)
for c in categories:
    rng.shuffle(idx_by_class[c])
    idx_by_class[c] = idx_by_class[c][:N_EVAL]      

val_subsets = {c: Subset(full_ds, idx_by_class[c]) for c in categories}
val_loaders = {
    c: DataLoader(sub, batch_size=BATCH, shuffle=False,
                  num_workers=4, persistent_workers=True, drop_last=False)
    for c, sub in val_subsets.items()
}

to_rgb = lambda x: x.repeat(1, 3, 1, 1)              # 1-ch → 3-ch

# ── METRIC HELPERS ──────────────────────────────────────────────────────────
def metrics_for_scale(scale: int):
    fid_cls, mssim_cls = {}, {}

    for c, label_id in zip(categories, range(len(categories))):
        # ---------- FID ----------
        fid = FID(feature=64, normalize=True).to(device)
        for imgs, _ in val_loaders[c]:
            fid.update(to_rgb(imgs.to(device)), real=True)

        synth_remaining = N_EVAL
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            while synth_remaining:
                cur = min(BATCH, synth_remaining)
                synth = model.sample(
                    label=label_id, N=cur, size=RESOLUTION,
                    guidance_scale=scale
                )
                fid.update(to_rgb(synth.to(device)), real=False)
                synth_remaining -= cur
        fid_cls[c] = fid.compute().item()

        # ---------- MS-SSIM ----------
        ms = MS_SSIM(data_range=1.0, gaussian_kernel=False).to(device)
        real_iter = iter(val_loaders[c])
        synth_remaining = N_EVAL
        while synth_remaining:
            cur   = min(BATCH, synth_remaining)
            real  = next(real_iter)[0].to(device)
            synth = model.sample(
                label=label_id, N=cur, size=RESOLUTION,
                guidance_scale=scale
            )
            ms.update(to_rgb(synth), to_rgb(real))
            synth_remaining -= cur
        mssim_cls[c] = ms.compute().item()

    # overall = mean of the four class scores
    fid_mean   = sum(fid_cls.values())   / len(fid_cls)
    mssim_mean = sum(mssim_cls.values()) / len(mssim_cls)

    return fid_cls | {"mean": fid_mean}, mssim_cls | {"mean": mssim_mean}

# ── SWEEP ──────────────────────────────────────────────────────────────────
results_fid, results_ms = {}, {}
for gs in SCALES:
    print(f"\n>>> guidance scale {gs}")
    fid_c, ms_c = metrics_for_scale(gs)
    results_fid[gs], results_ms[gs] = fid_c, ms_c
    print("FID   :", {k:f"{v:.2f}"  for k,v in fid_c.items()})
    print("MS-SSIM:", {k:f"{v:.4f}" for k,v in ms_c.items()})

# ── LOGGING ────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
csv_path = f"logs/{NAME_TAG}_metrics_{ts}.csv"

header = ["Guidance", *[f"FID_{c}" for c in categories+['mean']],
                     *[f"MSSSIM_{c}" for c in categories+['mean']]]

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(header)
    for gs in SCALES:
        fid_row = [results_fid[gs][c] for c in categories+['mean']]
        ms_row  = [results_ms [gs][c] for c in categories+['mean']]
        w.writerow([gs, *fid_row, *ms_row])

print(f"\nSaved results to {csv_path}")