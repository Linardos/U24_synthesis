#!/usr/bin/env python3
# collect_metrics.py -------------------------------------------------------
"""
Summarise validation metrics across experiments.

Selection rule (matches EarlyStopping on val loss):
  • filter phase == "val"
  • pick epoch with MIN validation 'loss'
  • report that epoch’s metrics

Outputs:
  • prints a markdown table (copy-paste ready)
  • writes oracle_summary.csv in the current working directory
"""

from __future__ import annotations

import pathlib
import pandas as pd

# --------------------------- CONFIG (EDIT ME) -----------------------------

EXPERIMENTS = [
    # "./experiments/090_EMBED_binary_clean_holdout_convnext_tiny_Oracle_seed44_Augsgeometric_real1.0_syn0.0",
    # "./experiments/091_EMBED_binary_256x256_holdout_convnext_tiny_Oracle_seed44_Augsgeometric_real1.0_syn0.0",
    # "./experiments/141_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real0.75_syn0.25",
    # "./experiments/142_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real0.5_syn0.5",
    # "./experiments/143_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real0.25_syn0.75",
    # "./experiments/144_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real0.0_syn1.0",
    # "./experiments/144_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real1.0_syn0.0",
    # new January 7 runs
    "./experiments/145_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real0.75_syn0.25",
    "./experiments/146_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real0.5_syn0.5",
    "./experiments/147_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real0.25_syn0.75",
    "./experiments/148_EMBED_binary_clean_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real0.0_syn1.0",
]


OUT_CSV = "oracle_summary.csv"

# -------------------------------------------------------------------------


def _require_cols(df: pd.DataFrame, cols: list[str], log_path: pathlib.Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{log_path}: missing columns: {missing}. Available: {list(df.columns)}")


def pick_best_val_row_by_loss(log_path: pathlib.Path) -> pd.Series:
    df = pd.read_csv(log_path)

    # required for early-stopping-on-loss semantics
    _require_cols(df, ["epoch", "phase", "loss"], log_path)

    df = df[df["phase"] == "val"].copy()
    if df.empty:
        raise ValueError(f"{log_path}: no val rows")

    # ensure numeric
    df["epoch"] = pd.to_numeric(df["epoch"], errors="raise").astype(int)
    df["loss"] = pd.to_numeric(df["loss"], errors="raise")

    # EarlyStopping behavior: pick the MIN val loss epoch
    # If ties, take the earliest epoch (stable / conservative)
    min_loss = df["loss"].min()
    tied = df[df["loss"] == min_loss].sort_values("epoch")
    chosen = tied.iloc[0]

    return chosen


def summarise_experiment(exp_dir: str) -> dict:
    log_path = pathlib.Path(exp_dir) / "logs.csv"
    if not log_path.exists():
        raise FileNotFoundError(str(log_path))

    best = pick_best_val_row_by_loss(log_path)

    # Pull whichever metrics exist (won't crash if some are missing)
    def get_num(col: str):
        return float(best[col]) if col in best.index and pd.notna(best[col]) else float("nan")

    return {
        "experiment": log_path.parent.name,
        "epoch": int(best["epoch"]),
        "val_loss": round(get_num("loss"), 6),
        "AUC": round(get_num("AUC"), 4),
        "Accuracy": round(get_num("overall_acc"), 4),
        "BalancedAcc": round(get_num("balanced_acc"), 4),
        "Specificity": round(get_num("ACC_class0"), 4),
        "Sensitivity": round(get_num("ACC_class1"), 4),
    }


def main():
    recs = []
    for d in EXPERIMENTS:
        try:
            recs.append(summarise_experiment(d))
        except Exception as e:
            print(f"⚠️  {d}: {e}")

    if not recs:
        print("No valid experiments found.")
        return

    df = pd.DataFrame(recs).set_index("experiment")

    # Sort by val_loss ascending (best first)
    df = df.sort_values("val_loss", ascending=True)

    print("\n=== Summary (best epoch = min val loss) ===")
    print(df.to_markdown())

    df.to_csv(OUT_CSV)
    print(f"\nWrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
