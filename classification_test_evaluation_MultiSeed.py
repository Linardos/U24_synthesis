#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# evaluate_oracle_metrics_multi.py
#   Multi-seed test-set evaluation for Balanced Acc, Sensitivity, Specificity, AUC
# ─────────────────────────────────────────────────────────────────────────────
"""
Usage
-----
$ python evaluate_oracle_metrics_multi.py
"""

import os
import csv
import yaml
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score

# ── project-level imports (make repo root importable first) ────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from data_loaders import NiftiDataset  # noqa: E402
from models import get_model           # noqa: E402

# ── CONFIGURATION ──────────────────────────────────────────────────────────
SEED = 42
RESOLUTION = 256
BATCH = 32
NUM_WORKERS = 4
CONFIG_YAML = "config.yaml"

# Put the 5 seed runs here

MODEL_DIRS = [
    "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed42_Augsgeometric_real1.0_syn0.0",
    "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed43_Augsgeometric_real1.0_syn0.0",
    "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real1.0_syn0.0",
    "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed45_Augsgeometric_real1.0_syn0.0",
    "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed46_Augsgeometric_real1.0_syn0.0",
]
MODEL_DIRS = [
    "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
    "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
    "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
    "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
    "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
]

MODEL_DIRS = [
    "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
    "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
    "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
    "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
    "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
]

# Dataset root (parent of the three splits below)
DATASET = "EMBED"
# DATA_ROOT = f"/mnt/d/Datasets/{DATASET}/{DATASET}_binary_256x256"
DATA_ROOT = f"/mnt/d/Datasets/{DATASET}/{DATASET}_binary_clean"

# Which subfolders to evaluate
SPLITS = [
    ("test", "Full test set"),
    ("test_cview", "C-View only"),
    ("test_2d", "2D only"),
]

# ── REPRODUCIBLE RNG SETUP ─────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ── HELPERS ────────────────────────────────────────────────────────────────
def build_transforms():
    """Expects a torch.Tensor shaped [1,H,W] or [1,H,W,D] from NiftiDataset.
    Applies per-image z-score, then min-max to [0,1]."""
    def _tf(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            else:
                x = x.mean(dim=-1)

        x = x.to(torch.float32)

        m, s = x.mean(), x.std()
        x = (x - m) / (s + 1e-8)

        xmin, xmax = x.min(), x.max()
        x = (x - xmin) / (xmax - xmin + 1e-8)
        return x

    return _tf


def load_model(ckpt_path: Path, num_classes: int):
    model = get_model("convnext_tiny", num_classes=num_classes, pretrained=False)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError(f"Checkpoint missing 'model_state' or 'model_state_dict': {ckpt_path}")

    model.load_state_dict(state)
    model = model.to(DEVICE).eval()

    # Keep float16 on CUDA for speed; float32 on CPU
    if DEVICE.startswith("cuda"):
        model = model.half()

    return model


def collect_logits_labels(
    model: torch.nn.Module,
    ds: Dataset,
    batch_size: int,
    num_workers: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the model over the dataset; return logits [N,C] and labels [N]."""
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE.startswith("cuda"),
        drop_last=False,
    )

    norm = Normalize(mean=[0.5], std=[0.5])  # [0,1] -> [-1,1]
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if DEVICE.startswith("cuda")
        else nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            for imgs, labels in dl:
                imgs = imgs.to(DEVICE, non_blocking=True)
                imgs = norm(imgs)

                logits = model(imgs)
                logits_list.append(logits.float().cpu())
                labels_list.append(labels.long().cpu())

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def metrics_binary(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute Balanced Acc, Sensitivity (TPR for malignant=1),
    Specificity (TNR for benign=0), and ROC-AUC.
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
        "balanced_acc": float(bal_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc": float(auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n": int(labels.numel()),
        "pos": int((y_true == 1).sum()),
        "neg": int((y_true == 0).sum()),
    }


# def safe_mean_std(values: List[float]) -> Tuple[float, float]:
#     arr = np.array(values, dtype=float)
#     arr = arr[~np.isnan(arr)]
#     if len(arr) == 0:
#         return float("nan"), float("nan")
#     if len(arr) == 1:
#         return float(arr[0]), 0.0
#     return float(arr.mean()), float(arr.std(ddof=1))

def safe_stats(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1)), float(arr.var(ddof=1))
# ── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Evaluating on dataset root: {DATA_ROOT}")

    with open(CONFIG_YAML) as f:
        cfg = yaml.safe_load(f)

    num_classes = int(cfg["num_classes"])
    assert num_classes == 2, "This script is set up for binary classification (num_classes==2)."

    tfm = build_transforms()

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_run_csv = f"logs/oracle_metrics_multiseed_per_run_{ts}.csv"
    summary_csv = f"logs/oracle_metrics_multiseed_summary_{ts}.csv"

    per_run_rows = []
    summary_bucket = {}  # (split_name, split_desc) -> list of metric dicts

    for model_dir in MODEL_DIRS:
        ckpt_path = Path("/home/locolinux2/U24_synthesis/experiments") / model_dir / "last.pt"
        if not ckpt_path.exists():
            print(f"[WARN] Missing checkpoint: {ckpt_path}. Skipping.")
            continue

        print(f"\n============================================================")
        print(f"Model: {model_dir}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"============================================================")

        model = load_model(ckpt_path, num_classes=num_classes)

        for split_name, split_desc in SPLITS:
            split_path = Path(DATA_ROOT) / split_name
            if not split_path.exists():
                print(f"[WARN] Split path missing: {split_path}. Skipping.")
                continue

            ds = NiftiDataset(str(split_path), transform=tfm)
            if len(ds) == 0:
                print(f"[WARN] Split {split_name} is empty. Skipping.")
                continue

            print(f"\n==> Split: {split_name} ({split_desc}) @ {split_path}")
            logits, labels = collect_logits_labels(model, ds, BATCH, NUM_WORKERS)
            m = metrics_binary(logits, labels)

            print(f"  N={m['n']}  benign={m['neg']}  malignant={m['pos']}")
            print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
            print(f"  BalancedAcc = {m['balanced_acc']:.3f}")
            print(f"  Sensitivity  = {m['sensitivity']:.3f} (malignant)")
            print(f"  Specificity  = {m['specificity']:.3f} (benign)")
            auc_str = f"{m['auc']:.3f}" if not math.isnan(m["auc"]) else "NaN"
            print(f"  ROC-AUC      = {auc_str}")

            row = {
                "model_dir": model_dir,
                "checkpoint": str(ckpt_path),
                "split": split_name,
                "description": split_desc,
                "n_total": m["n"],
                "n_benign": m["neg"],
                "n_malignant": m["pos"],
                "TP": m["tp"],
                "TN": m["tn"],
                "FP": m["fp"],
                "FN": m["fn"],
                "BalancedAccuracy": m["balanced_acc"],
                "Sensitivity_TPR_malignant": m["sensitivity"],
                "Specificity_TNR_benign": m["specificity"],
                "ROC_AUC": m["auc"],
            }
            per_run_rows.append(row)
            summary_bucket.setdefault((split_name, split_desc), []).append(row)

    # Write per-run CSV
    with open(per_run_csv, "w", newline="") as f:
        fieldnames = [
            "model_dir", "checkpoint", "split", "description",
            "n_total", "n_benign", "n_malignant",
            "TP", "TN", "FP", "FN",
            "BalancedAccuracy", "Sensitivity_TPR_malignant",
            "Specificity_TNR_benign", "ROC_AUC",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_run_rows:
            writer.writerow(row)

    # Write summary CSV
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "split", "description",
            "n_runs",
            "BalancedAccuracy_mean", "BalancedAccuracy_std", "BalancedAccuracy_var",
            "Sensitivity_TPR_malignant_mean", "Sensitivity_TPR_malignant_std", "Sensitivity_TPR_malignant_var",
            "Specificity_TNR_benign_mean", "Specificity_TNR_benign_std", "Specificity_TNR_benign_var",
            "ROC_AUC_mean", "ROC_AUC_std", "ROC_AUC_var",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (split_name, split_desc), rows in summary_bucket.items():
            bal = [r["BalancedAccuracy"] for r in rows]
            sen = [r["Sensitivity_TPR_malignant"] for r in rows]
            spe = [r["Specificity_TNR_benign"] for r in rows]
            auc = [r["ROC_AUC"] for r in rows]

            bal_m, bal_s, bal_v = safe_stats(bal)
            sen_m, sen_s, sen_v = safe_stats(sen)
            spe_m, spe_s, spe_v = safe_stats(spe)
            auc_m, auc_s, auc_v = safe_stats(auc)

            writer.writerow({
                "split": split_name,
                "description": split_desc,
                "n_runs": len(rows),
                "BalancedAccuracy_mean": bal_m,
                "BalancedAccuracy_std": bal_s,
                "BalancedAccuracy_var": bal_v,
                "Sensitivity_TPR_malignant_mean": sen_m,
                "Sensitivity_TPR_malignant_std": sen_s,
                "Sensitivity_TPR_malignant_var": sen_v,
                "Specificity_TNR_benign_mean": spe_m,
                "Specificity_TNR_benign_std": spe_s,
                "Specificity_TNR_benign_var": spe_v,
                "ROC_AUC_mean": auc_m,
                "ROC_AUC_std": auc_s,
                "ROC_AUC_var": auc_v,
            })
    print(f"\nSaved per-run results to {per_run_csv}")
    print(f"Saved summary results to {summary_csv}\n")


print("\n=== SUMMARY (mean ± std) ===\n")
for (split_name, split_desc), rows in summary_bucket.items():
    bal = [r["BalancedAccuracy"] for r in rows]
    sen = [r["Sensitivity_TPR_malignant"] for r in rows]
    spe = [r["Specificity_TNR_benign"] for r in rows]

    bal_m, bal_s, _ = safe_stats(bal)
    sen_m, sen_s, _ = safe_stats(sen)
    spe_m, spe_s, _ = safe_stats(spe)

    print(f"{split_name} ({split_desc}):")
    print(f"  Balanced Acc: {bal_m:.4f} ± {bal_s:.4f}")
    print(f"  Sensitivity : {sen_m:.4f} ± {sen_s:.4f}")
    print(f"  Specificity : {spe_m:.4f} ± {spe_s:.4f}")
    print()