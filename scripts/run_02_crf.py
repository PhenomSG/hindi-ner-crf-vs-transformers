#!/usr/bin/env python3
"""
Step 2: Train and Evaluate CRF Baseline

Extracts handcrafted Hindi features, trains sklearn-crfsuite CRF model,
evaluates on test split, and saves predictions and metrics.
"""

import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_hiner_dataset, format_for_crf
from src.crf_model import HindiCRFModel
from src.evaluation import save_results

def main():
    print("=" * 70)
    print("STEP 2: CRF BASELINE TRAINING & EVALUATION")
    print("=" * 70)

    # Output paths
    pred_dir = PROJECT_ROOT / "results" / "predictions"
    models_dir = PROJECT_ROOT / "models"
    pred_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\nLoading dataset...")
    dataset = load_hiner_dataset()

    train_sents, train_labels = format_for_crf(dataset["train"])
    val_key = "validation" if "validation" in dataset else "dev"
    val_sents, val_labels = format_for_crf(dataset[val_key])
    test_sents, test_labels = format_for_crf(dataset["test"])

    print(f"Train sentences: {len(train_sents):,}")
    print(f"Val sentences:   {len(val_sents):,}")
    print(f"Test sentences:  {len(test_sents):,}")

    # 2. Train CRF model
    print("\nTraining CRF model (lbfgs, c1=0.1, c2=0.1, max_iter=100)...")
    crf_model = HindiCRFModel(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        context_window=2,
    )

    crf_model.train(train_sents, train_labels, val_sents, val_labels)

    # 3. Save model checkpoint
    model_path = models_dir / "crf_model.pkl"
    crf_model.save(str(model_path))

    # 4. Evaluate on Test split
    print("\nEvaluating on Test Set...")
    test_metrics = crf_model.evaluate(test_sents, test_labels)

    print("\n" + "=" * 60)
    print("CRF TEST EVALUATION REPORT (Entity-Level seqeval)")
    print("=" * 60)
    print(f"Overall Precision: {test_metrics['overall_precision']:.4f}")
    print(f"Overall Recall:    {test_metrics['overall_recall']:.4f}")
    print(f"Overall F1 Score:  {test_metrics['overall_f1']:.4f}")
    print("\nDetailed Per-Entity Breakdown:")
    print(test_metrics["report_str"])

    # 5. Save metrics and predictions
    metrics_path = PROJECT_ROOT / "results" / "crf_test_metrics.json"
    save_results(test_metrics, str(metrics_path))

    test_predictions = crf_model.predict(test_sents)
    preds_output = []
    for i in range(len(test_sents)):
        preds_output.append({
            "tokens": test_sents[i],
            "true_labels": test_labels[i],
            "pred_labels": test_predictions[i]
        })

    pred_file = pred_dir / "crf_predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(preds_output, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved CRF predictions to: {pred_file}")

    # 6. Show top features
    print("\nTop 5 Learned State Features:")
    crf_model.print_top_features(top_n=5)

    print("\n✅ Step 2 CRF Baseline complete!")

if __name__ == "__main__":
    main()
