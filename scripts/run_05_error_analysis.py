#!/usr/bin/env python3
"""
Step 5: In-Depth Error Analysis & Hindi Linguistic Failure Modes

Analyzes errors made by both models, categorizes error taxonomy,
extracts concrete illustrative Hindi examples, and saves report tables.
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

from src.error_analysis import (
    analyze_all_errors,
    error_statistics,
    print_error_summary,
    get_interesting_examples,
    format_error_example,
)

def main():
    print("=" * 70)
    print("STEP 5: ERROR ANALYSIS & HINDI LINGUISTIC CASE STUDIES")
    print("=" * 70)

    pred_dir = PROJECT_ROOT / "results" / "predictions"
    tab_dir = PROJECT_ROOT / "results" / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    crf_pred_file = pred_dir / "crf_predictions.json"
    mbert_pred_file = pred_dir / "mbert_predictions.json"

    if not crf_pred_file.exists() and not mbert_pred_file.exists():
        print("✗ No predictions found. Run Step 2 (CRF) and Step 3 (mBERT) first!")
        return

    # Analyze CRF errors if available
    if crf_pred_file.exists():
        with open(crf_pred_file, "r") as f:
            crf_preds = json.load(f)
        crf_errors = analyze_all_errors(crf_preds)
        crf_stats = error_statistics(crf_errors)
        print_error_summary(crf_stats, model_name="CRF")

        # Select illustrative CRF examples
        interesting_crf = get_interesting_examples(crf_errors, n=5)
        print("\n" + "=" * 60)
        print("ILLUSTRATIVE CRF ERROR EXAMPLES")
        print("=" * 60)
        for i, ex in enumerate(interesting_crf):
            print(f"\n[Case {i+1}]")
            print(format_error_example(ex))

    # Analyze mBERT errors if available
    if mbert_pred_file.exists():
        with open(mbert_pred_file, "r") as f:
            mbert_preds = json.load(f)
        mbert_errors = analyze_all_errors(mbert_preds)
        mbert_stats = error_statistics(mbert_errors)
        print_error_summary(mbert_stats, model_name="mBERT")

        # Select illustrative examples
        interesting = get_interesting_examples(mbert_errors, n=8)
        print("\n" + "=" * 60)
        print("ILLUSTRATIVE HINDI ERROR EXAMPLES (FOR PPT / REPORT)")
        print("=" * 60)
        for i, ex in enumerate(interesting):
            print(f"\n[Case {i+1}]")
            print(format_error_example(ex))

        # Save examples table
        ex_rows = []
        for ex in interesting:
            ex_rows.append({
                "Error Type": ex["error_type"],
                "True Entity": f"{ex['true_entity']} ({ex['true_type']})" if ex["true_entity"] else "None",
                "Predicted Entity": f"{ex['pred_entity']} ({ex['pred_type']})" if ex["pred_entity"] else "None",
                "Context Snippet": ex["context"][:60] + "..." if len(ex["context"]) > 60 else ex["context"]
            })
        ex_df = pd.DataFrame(ex_rows)
        ex_csv = tab_dir / "illustrative_error_examples.csv"
        ex_df.to_csv(ex_csv, index=False)
        print(f"\n✓ Saved illustrative error examples table to: {ex_csv}")

    print("\n✅ Step 5 Error Analysis complete!")

if __name__ == "__main__":
    main()
