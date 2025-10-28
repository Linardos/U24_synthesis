#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
#  evaluate_oracle_metrics.py – Balanced Acc, Sensitivity, Specificity, AUC
# ─────────────────────────────────────────────────────────────────────────────
"""
Usage
-----
$ python evaluate_oracle_metrics.py
"""

import os, csv, yaml, json, math, random
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset   # ← (B) use Dataset here
from torchvision.transforms import Normalize
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score

# ── project-level imports (make repo root importable first) ────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]  # one level up from this file
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from data_loaders import NiftiDataset  # noqa: E402
from models import get_model           # noqa: E402

# ── CONFIGURATION ──────────────────────────────────────────────────────────
SEED         = 42
RESOLUTION   = 256
BATCH        = 32          # larger batch for faster eval; adjust as needed
NUM_WORKERS  = 4
CONFIG_YAML  = "config.yaml"

# Choose your model experiment directory and checkpoint (mirrors your pattern)
MODEL_DIR = "088_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsgeometric_real_perc1.0" # Oracle is the baseline
MODEL_DIR = "132_EMBED_binary_256x256_holdout_convnext_tiny_syngscale3_seed44_Augsbasic_real1.0_syn0.0_fTune_130_real1.0_syn0.75" # GSCALE 3
MODEL_DIR = "119_EMBED_binary_256x256_holdout_convnext_tiny_FTune_Freeze5_seed44_Augsgeometric_real1.0_syn0.0_fTune_097_real1.0_syn0.75" # GSCALE 5
MODEL_DIR = "097_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsgeometric_real1.0_syn0.75"
MODEL_DIR = "111_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_101_real1.0_syn0.75"
MODEL_DIR = "121_EMBED_binary_256x256_holdout_convnext_tiny_FTune_Freeze5_seed44_Augsgeometric_real1.0_syn0.0_fTune_096_real1.0_syn0.25"
MODEL_DIR = "109_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_099_real1.0_syn0.25"
MODEL_CKPT = (
    "/home/locolinux2/U24_synthesis/experiments/"
    f"{MODEL_DIR}/last.pt"
)

# Dataset root (parent of the three splits below)
DATASET     = "EMBED"
DATA_ROOT   = f"/mnt/d/Datasets/{DATASET}/{DATASET}_binary_256x256"

# Which subfolders to evaluate
SPLITS = [
    ("test",       "Full test set"),
    ("test_cview", "C-View only"),
    ("test_2d",    "2D only"),
]

# ── REPRODUCIBLE RNG SETUP ─────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── HELPERS ────────────────────────────────────────────────────────────────
# (A) tensor→tensor transform (no MONAI dict ops)
def build_transforms():
    """Expects a torch.Tensor shaped [1,H,W] or [1,H,W,D] from NiftiDataset.
    Applies per-image z-score, then min-max to [0,1]."""
    def _tf(x: torch.Tensor) -> torch.Tensor:
        # If 3D with trailing depth, collapse to 2D by mean over depth
        if x.ndim == 4:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)      # [1,H,W]
            else:
                x = x.mean(dim=-1)     # [1,H,W]
        x = x.to(torch.float32)
        # z-score
        m, s = x.mean(), x.std()
        x = (x - m) / (s + 1e-8)
        # min-max → [0,1]
        xmin, xmax = x.min(), x.max()
        x = (x - xmin) / (xmax - xmin + 1e-8)
        return x
    return _tf

def load_oracle(num_classes: int):
    model = get_model("convnext_tiny", num_classes=num_classes, pretrained=False)
    ckpt = torch.load(MODEL_CKPT, map_location="cpu", weights_only=False)
    # Support either "model_state" or "model_state_dict"
    state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint missing 'model_state' or 'model_state_dict'.")
    model.load_state_dict(state)
    model = model.half().to(DEVICE).eval()
    return model

# (B) type hint uses Dataset; safe autocast only on CUDA
def collect_logits_labels(
    ds: Dataset,
    batch_size: int,
    num_workers: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the model over the dataset; return logits [N,C] and labels [N]."""
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    norm = Normalize(mean=[0.5], std=[0.5])  # [0,1] -> [-1,1]
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    use_cuda_amp = DEVICE.startswith("cuda")
    if use_cuda_amp:
        ctx = torch.autocast("cuda", dtype=torch.float16)
        ctx.__enter__()
    try:
        with torch.no_grad():
            for imgs, labels in dl:
                imgs = imgs.to(DEVICE, non_blocking=True)
                logits = oracle(norm(imgs))
                logits_list.append(logits.float().cpu())
                labels_list.append(labels.long().cpu())
    finally:
        if use_cuda_amp:
            ctx.__exit__(None, None, None)

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)

def metrics_binary(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute Balanced Acc, Sensitivity (TPR for malignant=1),
    Specificity (TNR for benign=0), and ROC-AUC (prob of class 1).
    """
    if logits.ndim == 1 or logits.shape[1] == 1:
        probs1 = torch.sigmoid(logits.squeeze(1)).numpy()
    else:
        probs = torch.softmax(logits, dim=1).numpy()
        probs1 = probs[:, 1]

    y_true = labels.numpy()
    y_pred = (probs1 >= 0.5).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, probs1)
    except ValueError:
        auc = float("nan")

    return {
        "balanced_acc": bal_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": int(labels.numel()),
        "pos": int((y_true == 1).sum()),
        "neg": int((y_true == 0).sum()),
    }

# ── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Evaluating on dataset root: {DATA_ROOT}")

    with open(CONFIG_YAML) as f:
        cfg = yaml.safe_load(f)

    num_classes = int(cfg["num_classes"])
    assert num_classes == 2, "This script is set up for binary classification (num_classes==2)."

    categories = ["benign", "malignant"]  # 0/1 expected from dataset
    tfm = build_transforms()

    # Load Oracle once
    oracle = load_oracle(num_classes=num_classes)

    # Prepare logging
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"logs/oracle_metrics_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "description", "n_total", "n_benign", "n_malignant",
            "TP", "TN", "FP", "FN",
            "BalancedAccuracy", "Sensitivity_TPR_malignant", "Specificity_TNR_benign", "ROC_AUC"
        ])

        for split_name, split_desc in SPLITS:
            split_path = Path(DATA_ROOT) / split_name
            if not split_path.exists():
                print(f"[WARN] Split path missing: {split_path}. Skipping.")
                continue

            ds = NiftiDataset(str(split_path), transform=tfm)
            if len(ds) == 0:
                print(f"[WARN] Split {split_name} is empty. Skipping.")
                continue

            print(f"\n==> Split: {split_name}  ({split_desc})  @ {split_path}")
            logits, labels = collect_logits_labels(ds, BATCH, NUM_WORKERS)
            m = metrics_binary(logits, labels)

            # Pretty print
            print(f"  N={m['n']}  benign={m['neg']}  malignant={m['pos']}")
            print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
            print(f"  BalancedAcc = {m['balanced_acc']:.3f}")
            print(f"  Sensitivity = {m['sensitivity']:.3f} (malignant)")
            print(f"  Specificity = {m['specificity']:.3f} (benign)")
            auc_str = f"{m['auc']:.3f}" if not math.isnan(m['auc']) else "NaN"
            print(f"  ROC-AUC     = {auc_str}")

            # Write one row
            writer.writerow([
                split_name, split_desc, m["n"], m["neg"], m["pos"],
                m["tp"], m["tn"], m["fp"], m["fn"],
                f"{m['balanced_acc']:.6f}",
                f"{m['sensitivity']:.6f}" if not math.isnan(m["sensitivity"]) else "NaN",
                f"{m['specificity']:.6f}" if not math.isnan(m["specificity"]) else "NaN",
                f"{m['auc']:.6f}" if not math.isnan(m["auc"]) else "NaN",
            ])

    print(f"\nSaved results to {csv_path}\n")
