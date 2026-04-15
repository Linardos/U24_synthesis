import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- ROOT DIRECTORY ---
ROOT = Path("/home/locolinux2/U24_synthesis/experiments")

# --- HARDCODED RUNS ---
PHASE1_DIRS = [
    "151_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed42_Augsgeometric_real0.75_syn0.25",
    "151_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed43_Augsgeometric_real0.75_syn0.25",
    "151_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real0.75_syn0.25",
    "151_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed45_Augsgeometric_real0.75_syn0.25",
    "151_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed46_Augsgeometric_real0.75_syn0.25",
]

PHASE2_DIRS = [
    "152_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed42_Augsgeometric_real1.0_syn0.0_fTune_151_eal0.75_syn0.25",
    "152_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed43_Augsgeometric_real1.0_syn0.0_fTune_151_eal0.75_syn0.25",
    "152_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real1.0_syn0.0_fTune_151_eal0.75_syn0.25",
    "152_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed45_Augsgeometric_real1.0_syn0.0_fTune_151_eal0.75_syn0.25",
    "152_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed46_Augsgeometric_real1.0_syn0.0_fTune_151_eal0.75_syn0.25",
]

BASELINE_DIRS = [
    "153_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed42_Augsgeometric_real1.0_syn0.0",
    "153_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed43_Augsgeometric_real1.0_syn0.0",
    "153_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real1.0_syn0.0",
    "153_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed45_Augsgeometric_real1.0_syn0.0",
    "153_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed46_Augsgeometric_real1.0_syn0.0",
]

def extract_best_metrics(run_dir):
    logs_path = ROOT / run_dir / "logs.csv"
    df = pd.read_csv(logs_path)

    val_df = df[df["phase"] == "val"]

    # Pick best epoch by balanced accuracy
    best = val_df.loc[val_df["balanced_acc"].idxmax()]

    return {
        "balanced_acc": best["balanced_acc"],
        "specificity": best["ACC_class0"],
        "sensitivity": best["ACC_class1"],
    }


def process_group(dirs, label):
    results = []

    for d in dirs:
        metrics = extract_best_metrics(d)
        results.append(metrics)

    df = pd.DataFrame(results)

    summary = {
        "phase": label,
        "balanced_acc_mean": df["balanced_acc"].mean(),
        "balanced_acc_std": df["balanced_acc"].std(ddof=1),
        "sensitivity_mean": df["sensitivity"].mean(),
        "sensitivity_std": df["sensitivity"].std(ddof=1),
        "specificity_mean": df["specificity"].mean(),
        "specificity_std": df["specificity"].std(ddof=1),
    }

    return df, summary


# --- PROCESS ---
baseline_df, baseline_summary = process_group(BASELINE_DIRS, "baseline")
phase1_df, phase1_summary = process_group(PHASE1_DIRS, "phase1")
phase2_df, phase2_summary = process_group(PHASE2_DIRS, "phase2")

summary_df = pd.DataFrame([
    baseline_summary,
    phase1_summary,
    phase2_summary
])

# --- PRINT RESULTS ---
print("\n=== SUMMARY (mean ± std) ===\n")
for _, row in summary_df.iterrows():
    print(f"{row['phase']}:")
    print(f"  Balanced Acc: {row['balanced_acc_mean']:.4f} ± {row['balanced_acc_std']:.4f}")
    print(f"  Sensitivity : {row['sensitivity_mean']:.4f} ± {row['sensitivity_std']:.4f}")
    print(f"  Specificity : {row['specificity_mean']:.4f} ± {row['specificity_std']:.4f}")
    print()

# --- SAVE CSV ---
summary_df.to_csv("summary_results.csv", index=False)