# ---------------------------------------------------------------------------
# 5-frame forward-diffusion ladder (benign) – progressively noisier PNGs
# ---------------------------------------------------------------------------
import os, numpy as np, nibabel as nib
from PIL import Image

# paths
root      = '/mnt/d/Datasets/EMBED/EMBED_clean_256x256/train'
orig_path = os.path.join(root, 'original')
out_dir   = '/mnt/d/Datasets/EMBED/EMBED_clean_256x256/paper_sample_images'
os.makedirs(out_dir, exist_ok=True)

# helpers ----------------------------------------------------------
def load_first_images(base_dir, class_name, n=1):
    case = sorted(os.listdir(os.path.join(base_dir, class_name)))[0]
    nii  = nib.load(os.path.join(base_dir, class_name, case, 'slice.nii.gz'))
    return [np.squeeze(nii.get_fdata())]

def save_gray(arr, path, vmin, vmax):
    arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1)         # SAME scale!
    Image.fromarray((arr * 255).astype(np.uint8), 'L').save(path)

# 1. clean slice ---------------------------------------------------
x0         = load_first_images(orig_path, 'benign')[0].astype(np.float32)
vmin, vmax = x0.min(), x0.max()                                # global scale

# 2. DDPM β schedule ---------------------------------------------
num_steps  = 1000
betas      = np.linspace(1e-4, 2e-2, num_steps, dtype=np.float32)
alpha_bar  = np.cumprod(1.0 - betas)

# 3. pick timesteps + save ---------------------------------------
t_choices = [0, 100, 300, 600, 999]        # feel free to tweak
for i, t in enumerate(t_choices):
    eps  = np.random.randn(*x0.shape).astype(np.float32)
    xt   = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    save_gray(xt, os.path.join(out_dir, f'benign_forward_t{i}.png'), vmin, vmax)
