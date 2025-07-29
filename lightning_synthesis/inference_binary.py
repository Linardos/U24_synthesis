import os, uuid, numpy as np, torch, nibabel as nib
import glob
import re
from tqdm import tqdm
from pathlib import Path
from model_architectures import MonaiDDPM
device = "cuda:0" #if torch.cuda.is_available() else "cpu"

torch.manual_seed(2025)   # reproducible noise

RESOLUTION = 256
BATCH = 16       # keep RAM/VRAM sane; adjust to your GPU
GUIDE_SCALE = 5.0
T = 1_000     

root = Path("/home/locolinux2/U24_synthesis/lightning_synthesis/experiments")
ckpt = "099_DDPM_3loss_binary_FalsexFalse/checkpoints/epoch=12-step=3731.ckpt"
ckpt = "106_DDPM_3loss_binary_try/checkpoints/epoch=18-step=2736.ckpt"
ckpt = "147_EMBED_DDPM_augmentationsNone_binary_31fixedmatching_12vs56/checkpoints/epoch=26-step=1566.ckpt"
CKPT_PATH = root / ckpt

model = (MonaiDDPM
         .load_from_checkpoint(CKPT_PATH, map_location="cpu") 
         .half()                                                # weights → fp16
         .to(device)
         .eval())

# --- where to put the synthetic data ---------------------------------
# ---------------------------------------------------------------------
SYN_ROOT = f"/mnt/d/Datasets/EMBED/EMBED_clean_256x256_binary/train/synthetic_guide{GUIDE_SCALE}"

os.makedirs(SYN_ROOT, exist_ok=True)

print(f"Running inference for {SYN_ROOT}")
# --- how many of each label do we want? ------------------------------
# target_counts = {
#     'benign':          5740,
#     'malignant':       3444,
# }
# target_counts = {
#     'benign':          3444,
#     'malignant':       1148,
# }
target_counts = {
    'benign':          5514,
    'malignant':       1838,
}


# --- integer id mapping you trained with -----------------------------
class_labels = {
    'benign': 0,
    'malignant': 1,
}
# helper ─────────────────────────────────────────────────────────────
def existing_count(dir_path):
    """
    Counts how many <class>/<XXXX>/slice.nii.gz exist.
    Returns (count, highest_idx) so we can resume numbering.
    """
    pattern = os.path.join(dir_path, "[0-9]"*4, "slice.nii.gz")
    files   = glob.glob(pattern)
    if not files:
        return 0, 0
    # grab folder names like “…/0123/slice.nii.gz” → 123
    idxs = [int(re.search(r"/([0-9]{4})/slice", f).group(1)) for f in files]
    return len(idxs), max(idxs)
# ---------------------------------------------------------------------
#  GENERATION LOOP
# ---------------------------------------------------------------------
for cls_name, n_total in target_counts.items():
    label_id = class_labels[cls_name]
    out_class_dir = os.path.join(SYN_ROOT, cls_name)
    os.makedirs(out_class_dir, exist_ok=True)

    have, last_idx = existing_count(out_class_dir)
    need           = max(0, n_total - have)
    if need == 0:
        print(f"{cls_name}: already has ≥{n_total} samples – skipping.")
        continue

    print(f"{cls_name}: {have} present, need {need} more.")

    generated = 0
    pbar = tqdm(total=n_total, desc=f"Synth {cls_name:16}")

    while generated < n_total:
        n_batch = min(BATCH, n_total - generated)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            imgs = model.sample(
                label=label_id,
                N=n_batch,
                size=RESOLUTION,
                guidance_scale=GUIDE_SCALE,
            )                     # (N,1,H,W) in [0,1] float32

        # -----------------------------------------------------------------
        # Save each tensor to .../<class>/<000X>/slice.nii.gz
        # -----------------------------------------------------------------
        for img in imgs:
            idx      = generated + 1          # 1-based running index
            folder   = f"{idx:04d}"           # 0001, 0002, …
            sample_dir = os.path.join(out_class_dir, folder)
            os.makedirs(sample_dir, exist_ok=True)

            # convert to uint8 [0,255] and drop channel dim
            arr = (img.squeeze(0).cpu().numpy() * 255).round().astype(np.uint8)
            nifti = nib.Nifti1Image(arr, affine=np.eye(4))      # identity affine
            nib.save(nifti, os.path.join(sample_dir, "slice.nii.gz"))

            generated += 1
            pbar.update(1)

            if generated >= n_total:
                break  # safety when batch > remaining
    
    torch.cuda.empty_cache()  # call once per outer loop
    pbar.close()

print("✅  Synthetic dataset complete!")