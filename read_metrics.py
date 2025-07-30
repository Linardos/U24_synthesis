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

import csv, pathlib
import pandas as pd

TOL = 1e-3          # 0.001 AUC tolerance


def pick_epoch(log_path: pathlib.Path):
    with log_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["phase"] == "val"]

    # float‑ify once
    for r in rows:
        r["AUC"]           = float(r["AUC"])
        r["overall_acc"]   = float(r["overall_acc"])
        r["balanced_acc"]  = float(r["balanced_acc"])          # ❶ NEW
        r["ACC_class0"]    = float(r["ACC_class0"])
        r["ACC_class1"]    = float(r["ACC_class1"])
        r["epoch"]         = int(r["epoch"])

    auc_max = max(r["AUC"] for r in rows)
    best = max((r for r in rows if r["AUC"] >= auc_max - TOL),
               key=lambda r: r["epoch"])

    return {
        "experiment":   log_path.parent.name,
        "epoch":        best["epoch"],
        "AUC":          round(best["AUC"],          4),
        "Accuracy":     round(best["overall_acc"],  4),
        "BalancedAcc":  round(best["balanced_acc"], 4),        # ❷ NEW
        "Sensitivity":  round(best["ACC_class1"],   4),
        "Specificity":  round(best["ACC_class0"],   4),
    }

def main(exp_dirs):
    recs = []
    for d in exp_dirs:
        log = pathlib.Path(d) / "logs.csv"
        if not log.exists():
            print(f"⚠️  {log} missing – skipped"); continue
        recs.append(pick_epoch(log))

    if not recs:
        print("No valid experiments found."); return

    df = (pd.DataFrame(recs)
            .set_index("experiment")
            .sort_values("AUC", ascending=False)
            .round(4))                                         # ❸ NEW

    print("\n=== Oracle summary (Δ≤0.001 criterion) ===")
    print(df.to_markdown())          # ready for copy‑paste
    df.to_csv("oracle_summary.csv")

if __name__ == "__main__":
    
    experiments_list = [

        # "./experiments/083__EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test04lowerLR_seed44_real_perc1.0",
        # "./experiments/087_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsbasic_real_perc1.0",
        # "./experiments/088_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsgeometric_real_perc1.0",
        # "./experiments/089_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsintensity_real_perc1.0",
        #,        
        # "./experiments/088_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsgeometric_real_perc1.0",
        # "./experiments/094_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsgeometric_real1.0_syn1.0",
        # "./experiments/095_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsgeometric_real1.0_syn0.5",
        # "./experiments/096_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsgeometric_real1.0_syn0.25",
        # "./experiments/097_EMBED_binary_256x256_holdout_convnext_tiny_model_regularizations_test06smoothingLoss_seed44_Augsgeometric_real1.0_syn0.75",
        # #,        
        # "./experiments/098_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsintensity_real1.0_syn0.0",
        # "./experiments/099_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsintensity_real1.0_syn0.25",
        # "./experiments/100_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsintensity_real1.0_syn0.5",
        # "./experiments/101_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsintensity_real1.0_syn0.75",
        # "./experiments/102_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsintensity_real1.0_syn1.0",
        # #,
        # "./experiments/104_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_096_eal1.0_syn0.25",
        # "./experiments/105_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_095_real1.0_syn0.5",
        # "./experiments/106_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_097_eal1.0_syn0.75/",
        # "./experiments/107_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_094_real1.0_syn1.0",
        # "./experiments/108_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_098__real1.0_syn0.0",
        # #,
        # "./experiments/109_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_099_real1.0_syn0.25",
        # "./experiments/110_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_100__real1.0_syn0.5",
        # "./experiments/111_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_101_real1.0_syn0.75",
        # "./experiments/112_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsgeometric_real1.0_syn0.0_fTune_102__real1.0_syn1.0",
        # #,
        # "./experiments/113_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsintensity_real1.0_syn0.0_fTune_099_real1.0_syn0.25",
        # "./experiments/114_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsintensity_real1.0_syn0.0_fTune_100__real1.0_syn0.5",
        # "./experiments/115_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsintensity_real1.0_syn0.0_fTune_101_real1.0_syn0.75",
        # "./experiments/116_EMBED_binary_256x256_holdout_convnext_tiny_augExperiments_seed44_Augsintensity_real1.0_syn0.0_fTune_102__real1.0_syn1.0"
        # #,
        "./experiments/117_EMBED_binary_256x256_holdout_convnext_tiny_FTune_Freeze5_seed44_Augsgeometric_real1.0_syn0.0_fTune_096_real1.0_syn0.25",
        "./experiments/118_EMBED_binary_256x256_holdout_convnext_tiny_FTune_Freeze5_seed44_Augsgeometric_real1.0_syn0.0_fTune_095__real1.0_syn0.5",
        "./experiments/119_EMBED_binary_256x256_holdout_convnext_tiny_FTune_Freeze5_seed44_Augsgeometric_real1.0_syn0.0_fTune_097_real1.0_syn0.75",
        "./experiments/120_EMBED_binary_256x256_holdout_convnext_tiny_FTune_Freeze5_seed44_Augsgeometric_real1.0_syn0.0_fTune_094__real1.0_syn1.0",
        
    ]
    main(experiments_list)
