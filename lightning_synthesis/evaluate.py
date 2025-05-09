# ──────────────────────────────────────────────────────────────────────────────
#  evaluate.py  –  sample a fraction and score over different guidance scales
# ──────────────────────────────────────────────────────────────────────────────
import os, torch, random, csv
from datetime import datetime
from tqdm import tqdm
from data_loaders_l import NiftiSynthesisDataset
from model_architectures import MonaiDDPM
from monai.transforms import (
    LoadImaged, SqueezeDimd, EnsureChannelFirstd,
    Resized, ScaleIntensityd, ToTensord, Compose
)
from torchmetrics.image import FrechetInceptionDistance as FID


# ── CONFIG ────────────────────────────────────────────────────────────────────
# CKPT_PATH  = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/049_cDDPM_depth5_fixedScaling_256x256/checkpoints/epoch=22-step=7843.ckpt" # FID = 8.18 at guidance scale 3
CKPT_PATH  = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/053_DDPM__DataArtifactsRemoved___256x256/checkpoints/epoch=04-step=1435.ckpt"
CKPT_PATH  = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/054_DDPM_default512_256x256/checkpoints/epoch=04-step=1435.ckpt"
CKPT_PATH = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/063_DDPM_contrast-aug-20percent_256x256/checkpoints/epoch=04-step=1435.ckpt"

RESOLUTION = 256
BATCH      = 4
N_EVAL     = 128            # per class, per scale
NUM_STEPS  = 40
SCALES     = range(0, 6)    # 0,1,2,3,4,5

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(2025); random.seed(2025)
random.seed(2025)

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
log_txt_path = os.path.join(log_dir, f"eval_log_{timestamp}.txt")
log_csv_path = os.path.join(log_dir, f"eval_metrics_{timestamp}.csv")

# ── LOAD MODEL (fp16) ─────────────────────────────────────────────────────────
model = (MonaiDDPM
         .load_from_checkpoint(CKPT_PATH, map_location="cpu")
         .half().to(device).eval())

# ── REAL IMAGE LOADER (scaled to [0,1]) ───────────────────────────────────────
# root_dir   = "/mnt/d/Datasets/EMBED/EMBED_clean_512x512/train_3221/original"
root_dir   = '/mnt/d/Datasets/EMBED/EMBED_clean_256x256/train/original'
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

def to_rgb(x): return x.repeat(1,3,1,1)

def accumulate_real_features(fid_metric):
    for _ in range((N_EVAL * len(categories)) // BATCH + 1):
        real, _ = next(iter(real_ld))
        fid_metric.update(to_rgb(real.to(device)), real=True)

# ── EVALUATION FUNCTION ──────────────────────────────────────────────
def fid_for_scale(scale: int) -> float:
    fid = FID(feature=64, normalize=True).to(device)
    accumulate_real_features(fid)

    for cls_name, label_id in zip(categories, range(len(categories))):
        remaining = N_EVAL
        pbar = tqdm(total=N_EVAL, desc=f"GS={scale}  {cls_name:16}", leave=False)
        while remaining > 0:
            cur = min(BATCH, remaining)
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                synth = model.sample(label=label_id, N=cur, size=RESOLUTION,
                                     guidance_scale=scale,
                                     )
            fid.update(to_rgb(synth.to(device)), real=False)
            remaining -= cur; pbar.update(cur)
        pbar.close(); torch.cuda.empty_cache()
    return fid.compute().item()

# ── SWEEP ────────────────────────────────────────────────────────────
results = {}
for gs in SCALES:
    print(f"\n>>> evaluating guidance scale {gs}")
    results[gs] = fid_for_scale(gs)
    print(f"FID @ GS {gs}: {results[gs]:.2f}")

# ── LOGGING ──────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
with open(f"logs/fid_vs_guidance_{ts}.csv", "w", newline="") as f:
    csv.writer(f).writerows([["GuidanceScale","FID"]] +
                            [[k, round(v,2)] for k,v in results.items()])

print("\n──── FID vs. guidance scale ────")
for k,v in results.items():
    print(f"GS {k:>2}:  FID {v:6.2f}")