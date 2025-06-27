# ---------------------------------------------------------------------------
# 5-STEP FORWARD DIFFUSION, GLOBAL CONTRAST SCALING
# ---------------------------------------------------------------------------
import os, torch, numpy as np, nibabel as nib
from PIL import Image
from generative.networks.schedulers import DDPMScheduler

# ── config / paths ───────────────────────────────────────────────────────────
root      = '/mnt/d/Datasets/EMBED/EMBED_clean_256x256/train'
orig_path = os.path.join(root, 'original')
out_dir   = '/mnt/d/Datasets/EMBED/EMBED_clean_256x256/paper_sample_images'
os.makedirs(out_dir, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def load_one_slice(cls='benign'):
    case  = sorted(os.listdir(os.path.join(orig_path, cls)))[0]
    f     = os.path.join(orig_path, cls, case, 'slice.nii.gz')
    img   = np.squeeze(nib.load(f).get_fdata()).astype(np.float32)
    return img

def save_png(arr, fn):
    Image.fromarray((arr * 255).astype(np.uint8), 'L').save(os.path.join(out_dir, fn))

# ── 1. clean slice → tensor in [-1,1] ────────────────────────────────────────
x0 = load_one_slice('benign')
x0 = (x0 - x0.min()) / (x0.max() - x0.min())        # [0,1]
x0 = torch.from_numpy(x0 * 2 - 1).unsqueeze(0).unsqueeze(0)  # [-1,1]  (1,1,H,W)

# ── 2. scheduler identical to training ───────────────────────────────────────
sched = DDPMScheduler(num_train_timesteps=1000)
timesteps = [0, 200, 400, 700, 999]

# ── 3. make all frames first so we can find a global min/max ────────────────
frames = []
for t in timesteps:
    eps = torch.randn_like(x0)
    xt  = sched.add_noise(x0, eps, torch.tensor([t]))
    frames.append(xt.squeeze().numpy())             # (H,W)

# include the clean image to keep its contrast too
frames.insert(0, x0.squeeze().numpy())

gmin, gmax = np.min(frames), np.max(frames)

# ── 4. save with one consistent window ──────────────────────────────────────
for i, arr in enumerate(frames):                    # i=0 is clean image
    arr = np.clip((arr - gmin) / (gmax - gmin), 0, 1)   # common scale
    save_png(arr, f'benign_forward_t{i}.png')
