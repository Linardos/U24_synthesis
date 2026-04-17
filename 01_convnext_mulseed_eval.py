#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# compare_groups_metrics.py
#   Compare baseline vs syn025 vs syn075 across multiple seeds
#   Outputs mean ± std and paired significance tests vs baseline
# ─────────────────────────────────────────────────────────────────────────────

import os
import csv
import yaml
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score
from scipy.stats import ttest_rel, wilcoxon

# ── project-level imports ───────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from data_loaders import NiftiDataset  # noqa: E402
from models import get_model           # noqa: E402

# ── CONFIGURATION ──────────────────────────────────────────────────────────
SEED = 42
BATCH = 32
NUM_WORKERS = 4
CONFIG_YAML = "config.yaml"

DATASET = "EMBED"
DATA_ROOT = f"/mnt/d/Datasets/{DATASET}/{DATASET}_binary_clean"

SPLITS = [
    ("test", "Full test set"),
    ("test_cview", "C-View only"),
    ("test_2d", "2D only"),
]

GROUPS = {
    "baseline": [
        "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed42_Augsgeometric_real1.0_syn0.0",
        "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed43_Augsgeometric_real1.0_syn0.0",
        "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real1.0_syn0.0",
        "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed45_Augsgeometric_real1.0_syn0.0",
        "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed46_Augsgeometric_real1.0_syn0.0",
    ],
    "syn025": [
        "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
        "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
        "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
        "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
        "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
    ],
    "syn075": [
        "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
        "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
        "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
        "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
        "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
    ],
}

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── HELPERS ────────────────────────────────────────────────────────────────
def extract_seed(run_dir: str):
    m = re.search(r"seed(\d+)", run_dir)
    return int(m.group(1)) if m else None


def build_transforms():
    """Expects a torch.Tensor shaped [1,H,W] or [1,H,W,D]."""
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
    if DEVICE.startswith("cuda"):
        model = model.half()
    return model


def collect_logits_labels(
    model: torch.nn.Module,
    ds: Dataset,
    batch_size: int,
    num_workers: int
):
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE.startswith("cuda"),
        drop_last=False,
    )

    norm = Normalize(mean=[0.5], std=[0.5])
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


def metrics_binary(logits: torch.Tensor, labels: torch.Tensor):
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


def safe_stats(values: List[float]):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1)), float(arr.var(ddof=1))


def paired_tests(a: List[float], b: List[float]) -> Dict[str, float]:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]

    out = {"n_pairs": int(len(a)), "t_stat": float("nan"), "p_value": float("nan"), "wilcoxon_p": float("nan")}

    if len(a) >= 2:
        t_res = ttest_rel(a, b)
        out["t_stat"] = float(t_res.statistic)
        out["p_value"] = float(t_res.pvalue)

        try:
            w_res = wilcoxon(a, b)
            out["wilcoxon_p"] = float(w_res.pvalue)
        except ValueError:
            pass

    return out


def evaluate_group(group_name: str, model_dirs: List[str], num_classes: int, tfm):
    group_results = {}
    flat_rows = []

    for model_dir in model_dirs:
        seed = extract_seed(model_dir)
        ckpt_path = Path("/home/locolinux2/U24_synthesis/experiments") / model_dir / "last.pt"

        if seed is None:
            print(f"[WARN] Could not parse seed from: {model_dir}. Skipping.")
            continue
        if not ckpt_path.exists():
            print(f"[WARN] Missing checkpoint: {ckpt_path}. Skipping.")
            continue

        print(f"\n============================================================")
        print(f"Group: {group_name} | seed={seed}")
        print(f"Model: {model_dir}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"============================================================")

        model = load_model(ckpt_path, num_classes=num_classes)
        group_results[seed] = {}

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

            group_results[seed][split_name] = m
            flat_rows.append({
                "group": group_name,
                "seed": seed,
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
                "Sensitivity": m["sensitivity"],
                "Specificity": m["specificity"],
                "ROC_AUC": m["auc"],
            })

    return group_results, flat_rows


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
    per_run_csv = f"logs/group_metrics_per_run_{ts}.csv"
    summary_csv = f"logs/group_metrics_summary_{ts}.csv"
    sig_csv = f"logs/group_metrics_significance_{ts}.csv"

    all_group_results = {}
    per_run_rows = []

    for group_name, dirs in GROUPS.items():
        group_results, flat_rows = evaluate_group(group_name, dirs, num_classes, tfm)
        all_group_results[group_name] = group_results
        per_run_rows.extend(flat_rows)

    # Write per-run CSV
    with open(per_run_csv, "w", newline="") as f:
        fieldnames = [
            "group", "seed", "model_dir", "checkpoint", "split", "description",
            "n_total", "n_benign", "n_malignant", "TP", "TN", "FP", "FN",
            "BalancedAccuracy", "Sensitivity", "Specificity", "ROC_AUC",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_run_rows)

    # Group summaries
    summary_rows = []
    print("\n=== SUMMARY (mean ± std) ===\n")
    for group_name, seed_map in all_group_results.items():
        print(f"{group_name}:")
        for split_name, split_desc in SPLITS:
            metrics = [seed_map[s][split_name] for s in sorted(seed_map.keys()) if split_name in seed_map[s]]
            bal = [m["balanced_acc"] for m in metrics]
            sen = [m["sensitivity"] for m in metrics]
            spe = [m["specificity"] for m in metrics]

            bal_m, bal_s, bal_v = safe_stats(bal)
            sen_m, sen_s, sen_v = safe_stats(sen)
            spe_m, spe_s, spe_v = safe_stats(spe)

            print(f"  {split_name} ({split_desc}):")
            print(f"    Balanced Acc: {bal_m:.4f} ± {bal_s:.4f}")
            print(f"    Sensitivity : {sen_m:.4f} ± {sen_s:.4f}")
            print(f"    Specificity : {spe_m:.4f} ± {spe_s:.4f}")

            summary_rows.append({
                "group": group_name,
                "split": split_name,
                "description": split_desc,
                "n_runs": len(metrics),
                "BalancedAccuracy_mean": bal_m,
                "BalancedAccuracy_std": bal_s,
                "BalancedAccuracy_var": bal_v,
                "Sensitivity_mean": sen_m,
                "Sensitivity_std": sen_s,
                "Sensitivity_var": sen_v,
                "Specificity_mean": spe_m,
                "Specificity_std": spe_s,
                "Specificity_var": spe_v,
            })
        print()

    # Significance tests: baseline vs syn025 and baseline vs syn075
    significance_rows = []
    print("=== SIGNIFICANCE TESTS vs baseline ===\n")
    comparisons = [("baseline", "syn025"), ("baseline", "syn075")]
    metrics_to_test = [
        ("BalancedAccuracy", "balanced_acc"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
    ]

    for split_name, split_desc in SPLITS:
        print(f"{split_name} ({split_desc}):")
        base_seeds = set(all_group_results["baseline"].keys())
        for other_group in ["syn025", "syn075"]:
            other_seeds = set(all_group_results[other_group].keys())
            shared_seeds = sorted(base_seeds & other_seeds)

            for pretty_name, key in metrics_to_test:
                base_vals = [all_group_results["baseline"][s][split_name][key] for s in shared_seeds if split_name in all_group_results["baseline"][s]]
                other_vals = [all_group_results[other_group][s][split_name][key] for s in shared_seeds if split_name in all_group_results[other_group][s]]

                test = paired_tests(base_vals, other_vals)
                print(
                    f"  baseline vs {other_group} | {pretty_name}: "
                    f"n={test['n_pairs']}, p={test['p_value']:.4g}"
                )

                significance_rows.append({
                    "split": split_name,
                    "description": split_desc,
                    "comparison": f"baseline_vs_{other_group}",
                    "metric": pretty_name,
                    "n_pairs": test["n_pairs"],
                    "paired_t_stat": test["t_stat"],
                    "paired_t_pvalue": test["p_value"],
                    "wilcoxon_pvalue": test["wilcoxon_p"],
                })
        print()

    # Write summary CSV
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "group", "split", "description", "n_runs",
            "BalancedAccuracy_mean", "BalancedAccuracy_std", "BalancedAccuracy_var",
            "Sensitivity_mean", "Sensitivity_std", "Sensitivity_var",
            "Specificity_mean", "Specificity_std", "Specificity_var",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Write significance CSV
    with open(sig_csv, "w", newline="") as f:
        fieldnames = [
            "split", "description", "comparison", "metric",
            "n_pairs", "paired_t_stat", "paired_t_pvalue", "wilcoxon_pvalue",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(significance_rows)

    print(f"Saved per-run results to {per_run_csv}")
    print(f"Saved summary results to {summary_csv}")
    print(f"Saved significance results to {sig_csv}")