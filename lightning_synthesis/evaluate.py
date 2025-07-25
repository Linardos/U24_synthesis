# ─────────────────────────────────────────────────────────────────────────────
#  evaluate.py – per-class and overall FID / Oracle Acc on the held-out val split
# ─────────────────────────────────────────────────────────────────────────────
import os, csv, random, yaml, json
from datetime import datetime
from itertools import combinations
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from data_loaders_l import NiftiSynthesisDataset
from model_architectures import MonaiDDPM # synthesis models
from monai import transforms as mt
from torchmetrics.image import FrechetInceptionDistance as FID
import sys
from pathlib import Path

# ── make ~/U24_synthesis importable ─────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]   # one level up
if str(ROOT_DIR) not in sys.path:                # idempotent
    sys.path.insert(0, str(ROOT_DIR))

# now "from models import get_model" works
from models import get_model # classification models
from torchvision.transforms import Normalize               # oracle preprocessing

with open("config_l.yaml") as f:
    config = yaml.safe_load(f)


# ── CONFIG ────────────────────────────────────────────────────────────────────

root = Path("/home/locolinux2/U24_synthesis/lightning_synthesis/experiments")
# ckpt  = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/049_cDDPM_depth5_fixedScaling_256x256/checkpoints/epoch=22-step=7843.ckpt" # FID = 8.18 at guidance scale 3
ckpt  = "053_DDPM__DataArtifactsRemoved___256x256/checkpoints/epoch=04-step=1435.ckpt"
ckpt  = "054_DDPM_default512_256x256/checkpoints/epoch=04-step=1435.ckpt" # FID @ GS 0: 7.38
ckpt  = "057_DDPM_seed2025_cropped_256x256/checkpoints/epoch=04-step=1435.ckpt"
ckpt = "063_DDPM_contrast-aug-20percent_256x256/checkpoints/epoch=04-step=1435.ckpt"
ckpt = "092_DDPM_MS-SSIM_10perc_HF_5perc_256x256/checkpoints/epoch=18-step=2736.ckpt" # GOLD
ckpt = "106_DDPM_3loss_binary_try/checkpoints/epoch=18-step=2736.ckpt" # functional binary
ckpt = "107_DDPM_3loss_4class_retry/checkpoints/epoch=17-step=3456.ckpt"
ckpt = "113_DDPM_binary_epochwise_balanced/checkpoints/epoch=30-step=1116.ckpt" # dynamic epoch-wise data balancing 
ckpt = "114_DDPM_binary_epochwise_balanced_12vs56/checkpoints/epoch=22-step=1771.ckpt" # more data
ckpt = "121_DDPM_binary_21fixedmatching_12vs56/checkpoints/epoch=20-step=1827.ckpt" #  just to check how epoch-wise sampling affects things
ckpt = "120_DDPM_binary_21perepoch_12vs56/checkpoints/epoch=19-step=1740.ckpt" # seems decent! Even guidance scale 4 gets 70% accuracy.
ckpt = "123_DDPM_binary_11_5perepoch_12vs56/checkpoints/epoch=20-step=1512.ckpt" # 
ckpt = "124_DDPM_binary_11perepoch_12vs56/checkpoints/epoch=26-step=1566.ckpt"
# ckpt = "140_DDPM_augmentationsgeometric_binary_11perepoch_12vs56/checkpoints/epoch=26-step=1566.ckpt" # AUGMENTATIONS ARE AWESOME! 1:1
# ckpt = "141_DDPM_augmentationsgeometric_binary_11_5perepoch_12vs56/checkpoints/epoch=20-step=1512.ckpt" # augmentations, but 1.5:1
# ckpt = "142_DDPM_augmentationsall_binary_11perepoch_12vs56/checkpoints/epoch=21-step=1276.ckpt"
# ckpt = "143_DDPM_augmentationsgeometric_binary_31fixed_12vs56/checkpoints/epoch=10-step=957.ckpt" # just a test for some metrics
# ckpt = "144_CMMD_DDPM_augmentationsgeometric_binary_13fixed_12vs56/checkpoints/epoch=12-step=962.ckpt"
# ckpt = "145_CMMD_DDPM_augmentationsNone_binary_11balancing_12vs56/checkpoints/epoch=29-step=990.ckpt"
# ----- EMBED
ckpt = "147_EMBED_DDPM_augmentationsNone_binary_31fixedmatching_12vs56/checkpoints/epoch=26-step=1566.ckpt" # simple, no augs no balancing

CKPT_PATH = root / ckpt

# CKPT_PATH = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/094_DDPM_MS-SSIM_10perc_HF_5perc_val/checkpoints/epoch=12-step=1690.ckpt" # ACTUAL GOLD

print(f"Evaluating {CKPT_PATH}")
NAME_TAG = f"{ckpt[:4]}"

RESOLUTION = 256
BATCH      = 16
N_EVAL     = 200                       # samples / class
SCALES     = [0,4,5,6,7,8] #,9,10, 0,4]#[0, 4, 7, 8]
EVAL_SET = 'test'
# ORACLE_DIR  = "072_resnet50_CMMD_binary_12vs56_seed44_real_perc1.0" # "062_resnet50_binary_12vs56_seed44_real_perc1.0"
ORACLE_DIR  = "073_resnet50_CMMD_balanced_binary_12vs56_seed44_real_perc1.0"
ORACLE_DIR  = "074_CMMD_binary_256x256_resnet50_EMBED_binary_12vs56_dynamic11_seed44_real_perc1.0"
ORACLE_DIR  = "075_CMMD_binary_256x256_resnet50_EMBED_binary_12vs56_dynamic21_seed44_real_perc1.0"
ORACLE_DIR  = "088_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsgeometric_real_perc1.0" # GOLDEN EMBED, on test: Oracle ACC : {'benign': '0.875', 'malignant': '0.825', 'mean': '0.850'}
DATASET = 'EMBED' # CMMD or EMBED
MODEL_NAME = 'convnext_tiny'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); random.seed(42); np.random.seed(42)

if config["num_classes"] == 2:
    ORACLE_CKPT = ("/home/locolinux2/U24_synthesis/experiments/"
                f"{ORACLE_DIR}/last.pt")
                # "checkpoints/best_resnet50_fold5.pt")
elif config["num_classes"] == 4:
    ORACLE_CKPT = ("/home/locolinux2/U24_synthesis/experiments/"
                "048_resnet50_four_class_pretrainedImagenet_frozenlayers_seed44_real_perc1.0/"
                "checkpoints/best_resnet50_fold5.pt")
               
# ── DATASET ───────────────────────────────────────────────────────────────────
if EVAL_SET == 'val':
    DATA_ROOT = f"/mnt/d/Datasets/{DATASET}/{DATASET}_binary_256x256/train/original"
elif EVAL_SET =='test':
    DATA_ROOT = f"/mnt/d/Datasets/{DATASET}/{DATASET}_binary_256x256/test"

print(f"Evaluating synthesis on dataset: {DATASET} at path: {DATA_ROOT}")
categories = ["benign", "malignant"]
if config["num_classes"] >= 3:   categories.append("probably_benign")
if config["num_classes"] == 4:   categories.append("suspicious")
num_classes = len(categories)     # <-- used by oracle
# ── caching ─────────────────────────────────────────────────────────────
CACHE_SYNTH   = True                     # flip off if you really want fresh draws
CACHE_ROOT    = Path(f"/mnt/d/Datasets/{DATASET}/synth_cache")      # anything inside .gitignore
ckpt_tag      = CKPT_PATH.parent.name    # e.g. 140_DDPM_augmentationsgeometric…
def load_or_generate(model, label_id, class_name, scale, n):
    save_dir  = CACHE_ROOT / ckpt_tag / f"gs{scale}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f"{class_name}.pt"

    # 1) already cached?
    if save_file.exists():
        imgs = torch.load(save_file, map_location="cpu")
        cur  = imgs.shape[0]

        if cur >= n:                       # we have enough — trim if asked for less
            return imgs[:n]

        # need to generate the *missing* part
        missing, bank = n - cur, []
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            while missing:
                k = min(BATCH, missing)
                bank.append(
                    model.sample(label=label_id,
                                 N=k,
                                 size=RESOLUTION,
                                 guidance_scale=scale).cpu()
                )
                missing -= k

        imgs = torch.cat([imgs] + bank, dim=0)  # (n,1,H,W)
        torch.save(imgs, save_file)             # overwrite with the larger set
        return imgs

    # 2) file didn’t exist → create it from scratch
    bank, remaining = [], n
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        while remaining:
            k = min(BATCH, remaining)
            bank.append(
                model.sample(label=label_id,
                             N=k,
                             size=RESOLUTION,
                             guidance_scale=scale).cpu()
            )
            remaining -= k

    imgs = torch.cat(bank, dim=0)
    torch.save(imgs, save_file)
    return imgs


# ── LOAD MODEL (fp16) ─────────────────────────────────────────────────────────
model = (MonaiDDPM
         .load_from_checkpoint(CKPT_PATH, map_location="cpu")
         .half().to(device).eval())
# ── LOAD ORACLE  (right after you load the diffusion model) ────────────────
oracle = get_model(MODEL_NAME,
                   num_classes=len(categories),   # ⬅  matches current eval set
                   pretrained=False)
ckpt   = torch.load(ORACLE_CKPT, map_location="cpu", weights_only=False)
# oracle.load_state_dict(ckpt["model_state_dict"])
oracle.load_state_dict(ckpt["model_state"])
oracle = oracle.half().to(device).eval()                   # keep fp16 + GPU

# image → oracle preprocessing (expects 1-channel × 256 ×256 in **[-1,1]**)
oracle_norm = Normalize(mean=[0.5], std=[0.5])             # (x-0.5)/0.5 → [-1,1]

# ──────────────────────────────────────────────────────────────────

print(f"Evaluating categories {categories}, for scales {SCALES}...")

real_tf = mt.Compose([
    mt.LoadImaged(keys=["image"], image_only=True),
    mt.SqueezeDimd(keys=["image"], dim=-1),
    mt.EnsureChannelFirstd(keys=["image"]),
    mt.Lambdad(keys=["image"],
               func=lambda img: (img - img.mean()) / (img.std() + 1e-8)),
    mt.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    mt.ToTensord(keys=["image"]),
])

full_ds  = NiftiSynthesisDataset(DATA_ROOT, transform=real_tf)

# ---------- reproducible 10 % val split then 128-per-class subset ----------
if EVAL_SET=='val':
    # Folder that contains best.pt and indices.json (adjust if needed)
    TRAIN_EXP_DIR = f"/home/locolinux2/U24_synthesis/experiments/{ORACLE_DIR}"

    with open(os.path.join(TRAIN_EXP_DIR, "indices.json")) as jf:
        idx_dict = json.load(jf)
    val_idx = np.asarray(idx_dict["val_real"], dtype=int)     # same order as real_ds
    val_ds = Subset(full_ds, val_idx) 
else:
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
def metrics_for_scale(model, scale):
    fid_cls, acc_cls = {}, {}
    fid_global = FID(feature=64, normalize=True).to(device)
    synth_bank = {c: [] for c in categories}

    for c, label_id in zip(categories, range(len(categories))):
        # ── prepare “real” side ─────────────────────────────────────────
        fid = FID(feature=64, normalize=True).to(device)
        for imgs, _ in val_loaders[c]:
            fid.update(to_rgb(imgs.to(device)), real=True)
            fid_global.update(to_rgb(imgs.to(device)), real=True)

        # ── synthetic side (cache-aware) ────────────────────────────────
        synth = load_or_generate(model, label_id, c, scale, N_EVAL)      # (N_EVAL,1,H,W)
        synth_bank[c].append(synth)                               # later BCFID

        fid.update(to_rgb(synth.to(device)), real=False)
        fid_global.update(to_rgb(synth.to(device)), real=False)

        # oracle accuracy ------------------------------------------------
        inp = oracle_norm(synth.to(device)).half()   # keep/restore FP16
        logits = oracle(inp)                         # (N_EVAL, C)
        preds   = logits.argmax(dim=1)
        correct = (preds == label_id).sum().item()

        fid_cls[c] = fid.compute().item()
        acc_cls[c] = correct / N_EVAL

    # ── BCFID (between-class FID) ───────────────────────────────────────
    bcfids = []
    for c1, c2 in combinations(categories, 2):
        fid_pair = FID(feature=64, normalize=True).to(device)
        for t in synth_bank[c1]:
            fid_pair.update(to_rgb(t.to(device)), real=True)
        for t in synth_bank[c2]:
            fid_pair.update(to_rgb(t.to(device)), real=False)
        bcfids.append(fid_pair.compute().item())
    fid_cls["bcfid"] = sum(bcfids) / len(bcfids)

    # summaries ---------------------------------------------------------
    fid_cls["global"] = fid_global.compute().item()
    fid_cls["mean"]   = sum(fid_cls[k] for k in categories) / len(categories)
    acc_cls["mean"]   = sum(acc_cls.values()) / len(categories)
    return fid_cls, acc_cls


# ── SWEEP ──────────────────────────────────────────────────────────────────
results_fid, results_acc = {}, {}
for gs in SCALES:
    print(f"\n>>> guidance scale {gs}")
    fid_c, acc_c = metrics_for_scale(model, gs)
    results_fid[gs], results_acc[gs] = fid_c, acc_c
    print("FID :", {k:f"{v:.2f}" for k,v in fid_c.items()})
    print("Oracle ACC :", {k:f"{v:.3f}" for k,v in acc_c.items()})

# ── LOGGING ───────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
csv_path = f"logs/{NAME_TAG}{EVAL_SET}_metrics_{ts}.csv"

header = (["Guidance",
           "FID_global", "FID_bcfid",
           *[f"FID_{c}" for c in categories + ['mean']],
           *[f"ACC_{c}" for c in categories + ['mean']]])

with open(csv_path, "w", newline="") as f:        # ← open file
    writer = csv.writer(f)
    writer.writerow(header)

    for gs in SCALES:
        fid_row = [results_fid[gs]["global"],
                   results_fid[gs]["bcfid"],
                   *[results_fid[gs][c] for c in categories + ['mean']]]
        acc_row = [results_acc[gs][c] for c in categories + ['mean']]
        writer.writerow([gs, *fid_row, *acc_row])

print(f"\nSaved results to {csv_path}")
