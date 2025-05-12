import os, uuid, numpy as np, torch, nibabel as nib
from tqdm import tqdm
from model_architectures import MonaiDDPM
device = "cuda:0" #if torch.cuda.is_available() else "cpu"

torch.manual_seed(2025)   # reproducible noise

CONDITIONAL = True
RESOLUTION = 256
BATCH = 1       # keep RAM/VRAM sane; adjust to your GPU
GUIDE_SCALE = 5.0
T = 1_000     

# ckpt_path = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/049_cDDPM_depth5_fixedScaling_256x256/checkpoints/epoch=22-step=7843.ckpt"
ckpt_path = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/054_DDPM_default512_256x256/checkpoints/epoch=04-step=1435.ckpt"
ckpt_path = "/home/locolinux2/U24_synthesis/lightning_synthesis/experiments/063_DDPM_contrast-aug-20percent_256x256/checkpoints/epoch=04-step=1435.ckpt"

model = (MonaiDDPM
         .load_from_checkpoint(ckpt_path, map_location="cpu")   # keep GPU free
         .half()                                                # weights → fp16
         .to(device)
         .eval())

# --- where to put the synthetic data ---------------------------------
# ---------------------------------------------------------------------
if "contrast" in ckpt_path:
    SYN_ROOT = f"/mnt/d/Datasets/EMBED/EMBED_clean_256x256/train/synthetic_guide{GUIDE_SCALE}_contrast-enhanced"
else:
    SYN_ROOT = f"/mnt/d/Datasets/EMBED/EMBED_clean_256x256/train/synthetic_guide{GUIDE_SCALE}"

os.makedirs(SYN_ROOT, exist_ok=True)

print(f"Running inference for {SYN_ROOT}")
# --- how many of each label do we want? ------------------------------
target_counts = {
    'benign':          4092,
    'probably_benign': 2728,
    'suspicious':      2728,
    'malignant':       1364,
}

# --- integer id mapping you trained with -----------------------------
class_labels = {
    'benign': 0,
    'malignant': 1,
    'probably_benign': 2,
    'suspicious': 3
}

# ---------------------------------------------------------------------
#  GENERATION LOOP
# ---------------------------------------------------------------------
for cls_name, n_total in target_counts.items():
    label_id = class_labels[cls_name]
    out_class_dir = os.path.join(SYN_ROOT, cls_name)
    os.makedirs(out_class_dir, exist_ok=True)

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