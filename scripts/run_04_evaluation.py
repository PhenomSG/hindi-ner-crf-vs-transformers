#!/usr/bin/env python3
"""
Step 4: Comparative Evaluation (CRF vs mBERT)

Loads test metrics and predictions for both models, computes entity-level
comparison tables, entity length breakdown, and comparative plots.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    build_comparison_table,
    plot_f1_comparison,
    compute_entity_confusion_matrix,
    plot_confusion_matrix,
    performance_by_entity_length,
)

def main():
    print("=" * 70)
    print("STEP 4: COMPARATIVE EVALUATION (CRF vs mBERT)")
    print("=" * 70)

    results_dir = PROJECT_ROOT / "results"
    fig_dir = results_dir / "figures"
    tab_dir = results_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    crf_metrics_file = results_dir / "crf_test_metrics.json"
    mbert_metrics_file = results_dir / "mbert_test_metrics.json"

    if not crf_metrics_file.exists():
        print(f"✗ CRF metrics not found at {crf_metrics_file}. Run Step 2 first!")
        return

    if not mbert_metrics_file.exists():
        print(f"✗ mBERT metrics not found at {mbert_metrics_file}. Run Step 3 first!")
        return

    with open(crf_metrics_file, "r") as f:
        crf_metrics = json.load(f)

    with open(mbert_metrics_file, "r") as f:
        mbert_metrics = json.load(f)

    # 1. Comparison Table
    print("\n--- Model Comparison Table ---")
    comp_df = build_comparison_table(crf_metrics, mbert_metrics)
    print(comp_df.to_string(index=False))

    comp_csv = tab_dir / "crf_vs_mbert_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n✓ Saved comparison table to: {comp_csv}")

    # 2. Plot F1 comparison
    print("\nGenerating F1 comparison plot...")
    f1_plot = fig_dir / "f1_comparison.png"
    plot_f1_comparison(crf_metrics, mbert_metrics, save_path=str(f1_plot))

    # 3. Predictions comparison & Confusion Matrices
    crf_pred_file = results_dir / "predictions" / "crf_predictions.json"
    mbert_pred_file = results_dir / "predictions" / "mbert_predictions.json"

    if crf_pred_file.exists() and mbert_pred_file.exists():
        with open(crf_pred_file, "r") as f:
            crf_preds = json.load(f)
        with open(mbert_pred_file, "r") as f:
            mbert_preds = json.load(f)

        print("\nComputing Confusion Matrices...")
        crf_conf = compute_entity_confusion_matrix(crf_preds)
        plot_confusion_matrix(crf_conf, model_name="CRF", save_path=str(fig_dir / "crf_confusion_matrix.png"))

        mbert_conf = compute_entity_confusion_matrix(mbert_preds)
        plot_confusion_matrix(mbert_conf, model_name="mBERT", save_path=str(fig_dir / "mbert_confusion_matrix.png"))

        # Length analysis
        print("\nEntity Length Breakdown (mBERT):")
        length_df = performance_by_entity_length(mbert_preds)
        print(length_df.to_string(index=False))
        length_df.to_csv(tab_dir / "mbert_length_analysis.csv", index=False)

    print("\n✅ Step 4 Evaluation complete!")

if __name__ == "__main__":
    main()
