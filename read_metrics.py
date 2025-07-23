#!/usr/bin/env python
# collect_metrics.py -------------------------------------------------------
"""
Summarise oracle-training metrics across experiments.

For each experiment folder given on the CLI:
  • read logs.csv
  • find max validation AUC (AUC_max)
  • find the LAST epoch whose AUC ≥ AUC_max − 0.001
  • report that epoch’s metrics
"""

import sys, csv, pathlib
import pandas as pd

TOL = 1e-3          # 0.001 AUC tolerance

def pick_epoch(log_path: pathlib.Path):
    with log_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["phase"] == "val"]

    # float-ify once
    for r in rows:
        r["AUC"]         = float(r["AUC"])
        r["overall_acc"] = float(r["overall_acc"])
        r["ACC_class0"]  = float(r["ACC_class0"])
        r["ACC_class1"]  = float(r["ACC_class1"])
        r["epoch"]       = int(r["epoch"])

    auc_max = max(r["AUC"] for r in rows)
    # keep LAST epoch within tolerance
    best = max((r for r in rows if r["AUC"] >= auc_max - TOL),
               key=lambda r: r["epoch"])

    return {
        "experiment":  log_path.parent.name,
        "epoch":       best["epoch"],
        "AUC":         round(best["AUC"], 3),
        "Accuracy":    round(best["overall_acc"], 3),
        "Sensitivity": round(best["ACC_class1"], 3),
        "Specificity": round(best["ACC_class0"], 3),
    }

def main(exp_dirs):
    recs = []
    for d in exp_dirs:
        log = pathlib.Path(d) / "logs.csv"
        if not log.exists():
            print(f"⚠️  {log} missing – skipped")
            continue
        recs.append(pick_epoch(log))

    if not recs:
        print("No valid experiments found."); return

    df = (pd.DataFrame(recs)
            .set_index("experiment")
            .sort_values("AUC", ascending=False))

    print("\n=== Oracle summary (Δ≤0.001 criterion) ===")
    print(df.to_markdown())        # easy copy-paste
    # also dump CSV if you like
    df.to_csv("oracle_summary.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect_metrics.py EXP_DIR [EXP_DIR ...]")
        sys.exit(1)
    main(sys.argv[1:])
