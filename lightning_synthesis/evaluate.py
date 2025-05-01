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

# ── METRICS ───────────────────────────────────────────────────────────────────
ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.).to(device)
psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.).to(device)

# track per-class results
class_stats = {c: {"ssim": [], "psnr": []} for c in categories}

# ── EVALUATION LOOP ───────────────────────────────────────────────────────────
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

        # get a random real batch (drop-in replacement each call)
        try:
            real, _ = next(real_iter)
        except StopIteration:
            real_iter = iter(real_ld)
            real, _ = next(real_iter)

        real = real[:cur].to(device)              # match shape
        ssim_val = ssim(synth.float(), real.float()).item()
        psnr_val = psnr(synth.float(), real.float()).item()

        class_stats[cls_name]["ssim"].append(ssim_val)
        class_stats[cls_name]["psnr"].append(psnr_val)

        remaining -= cur
        pbar.update(cur)

    pbar.close()
    torch.cuda.empty_cache()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n─────  quick evaluation  ───────────────────────────")
for c in categories:
    ssim_mean = sum(class_stats[c]["ssim"]) / len(class_stats[c]["ssim"])
    psnr_mean = sum(class_stats[c]["psnr"]) / len(class_stats[c]["psnr"])
    print(f"{c:16}  SSIM {ssim_mean:.4f} | PSNR {psnr_mean:.2f} dB")

overall_ssim = sum(v for cls in class_stats.values() for v in cls["ssim"]) / \
               sum(len(cls["ssim"]) for cls in class_stats.values())
overall_psnr = sum(v for cls in class_stats.values() for v in cls["psnr"]) / \
               sum(len(cls["psnr"]) for cls in class_stats.values())

print(f"\nOVERALL         SSIM {overall_ssim:.4f} | PSNR {overall_psnr:.2f} dB")
print("✅  done (no files written)")

# ─ Text log (same as what’s printed) ──────────────────────────────────────────
with open(log_txt_path, "w") as f:
    f.write("─────  quick evaluation  ───────────────────────────\n")
    for c in categories:
        ssim_mean = sum(class_stats[c]["ssim"]) / len(class_stats[c]["ssim"])
        psnr_mean = sum(class_stats[c]["psnr"]) / len(class_stats[c]["psnr"])
        f.write(f"{c:16}  SSIM {ssim_mean:.4f} | PSNR {psnr_mean:.2f} dB\n")
    f.write(f"\nOVERALL         SSIM {overall_ssim:.4f} | PSNR {overall_psnr:.2f} dB\n")
    f.write("✅  done\n")

# ─ CSV structured summary ─────────────────────────────────────────────────────
with open(log_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "SSIM_Mean", "PSNR_Mean"])
    for c in categories:
        ssim_mean = sum(class_stats[c]["ssim"]) / len(class_stats[c]["ssim"])
        psnr_mean = sum(class_stats[c]["psnr"]) / len(class_stats[c]["psnr"])
        writer.writerow([c, round(ssim_mean, 4), round(psnr_mean, 2)])
    writer.writerow(["OVERALL", round(overall_ssim, 4), round(overall_psnr, 2)])