# ──────────────────────────────────────────────────────────────────────────────
#  evaluate.py  –  sample a fraction and score
# ──────────────────────────────────────────────────────────────────────────────
import os, torch, random
import csv
from datetime import datetime
from tqdm import tqdm
import torchmetrics
from data_loaders_l import NiftiSynthesisDataset          # your dataset
from model_architectures import MonaiDDPM                 # your model
from monai.transforms import (
    LoadImaged, SqueezeDimd, EnsureChannelFirstd,
    Resized, ScaleIntensityd, ToTensord, Compose
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CKPT_PATH   = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/049_cDDPM_depth5_fixedScaling_256x256/checkpoints/epoch=22-step=7843.ckpt"
GUIDE_SCALE = 3.0
RESOLUTION  = 256
BATCH       = 4         # GPU-friendly
N_EVAL      = 128       # how many synthetic images PER class to score
NUM_STEPS   = 40        # fast sampler steps

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(2025)
random.seed(2025)

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_txt_path = os.path.join(log_dir, f"eval_log_{timestamp}.txt")
log_csv_path = os.path.join(log_dir, f"eval_metrics_{timestamp}.csv")

# ── LOAD MODEL (fp16) ─────────────────────────────────────────────────────────
model = (MonaiDDPM
         .load_from_checkpoint(CKPT_PATH, map_location="cpu")
         .half().to(device).eval())

# ── REAL IMAGE LOADER (scaled to [0,1]) ───────────────────────────────────────
root_dir   = "/mnt/d/Datasets/EMBED/EMBED_clean_512x512/train_3221/original"
split      = "train"
categories = ["benign","probably_benign","suspicious","malignant"]

real_tf = Compose([
    LoadImaged(keys=["image"], image_only=True),
    SqueezeDimd(keys=["image"], dim=-1),
    EnsureChannelFirstd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(RESOLUTION, RESOLUTION)),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    ToTensord(keys=["image"]),
])

real_ds  = NiftiSynthesisDataset(root_dir, transform=real_tf)
real_ld  = torch.utils.data.DataLoader(real_ds, batch_size=BATCH,
                                       shuffle=True, num_workers=4,
                                       drop_last=True, persistent_workers=True)
real_iter = iter(real_ld)
# ── METRICS ───────────────────────────────────────────────────────────
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.kid import KernelInceptionDistance  as KID

# Inception-V3 64-d feature layer → 1 GB VRAM instead of full 2048-d
fid = FID(feature=64, normalize=True).to(device)
kid = KID(subset_size=100, normalize=True).to(device)

# ── EVALUATION LOOP (unchanged except metrics) ────────────────────────
for cls_name, label_id in zip(categories, range(len(categories))):
    remaining = N_EVAL
    pbar = tqdm(total=N_EVAL, desc=f"Eval {cls_name:16}")

    while remaining > 0:
        cur = min(BATCH, remaining)

        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            synth = model.sample(
                label=label_id,
                N=cur,
                size=RESOLUTION,
                guidance_scale=GUIDE_SCALE
            )

        # real batch (already [0,1] from transform)
        try:
            real, _ = next(real_iter)
        except StopIteration:
            real_iter = iter(real_ld)
            real, _ = next(real_iter)

        real  = real[:cur].to(device)
        synth = synth.to(device)

        fid.update(real,  real=True)
        fid.update(synth, real=False)

        kid.update(real,  real=True)
        kid.update(synth, real=False)

        remaining -= cur
        pbar.update(cur)

    pbar.close()
    torch.cuda.empty_cache()

# ── SUMMARY ───────────────────────────────────────────────────────────
fid_val = fid.compute().item()
kid_mean, kid_std = kid.compute()          # returns mean ± std
kid_val = kid_mean.item()

print("\n─────  FID / KID evaluation  ───────────────────────")
print(f"FID  : {fid_val:6.2f}")
print(f"KID  : {kid_val*1000:6.2f} × 1e-3")   # common formatting
print("✅  done (no files written)")

# ─ Text log ───────────────────────────────────────────────────────────
with open(log_txt_path, "w") as f:
    f.write("─────  FID / KID evaluation  ───────────────────────\n")
    f.write(f"FID  : {fid_val:6.2f}\n")
    f.write(f"KID  : {kid_val*1000:6.2f} × 1e-3\n")

# ─ CSV log ────────────────────────────────────────────────────────────
with open(log_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["FID",  round(fid_val, 2)])
    writer.writerow(["KID(×1e-3)", round(kid_val*1000, 2)])
