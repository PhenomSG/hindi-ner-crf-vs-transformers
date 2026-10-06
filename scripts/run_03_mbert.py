#!/usr/bin/env python3
"""
Step 3: Fine-tune mBERT for Hindi NER

Uses HuggingFace Trainer to fine-tune bert-base-multilingual-cased,
saves the model checkpoint, evaluates on test split, and saves predictions.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_hiner_dataset
from src.transformer_ner import HindiTransformerNER
from src.evaluation import save_results

def main():
    print("=" * 70)
    print("STEP 3: mBERT FINE-TUNING FOR HINDI NER")
    print("=" * 70)

    # Output paths
    output_model_dir = PROJECT_ROOT / "models" / "mbert_hindi_ner"
    pred_dir = PROJECT_ROOT / "results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware accelerator detected: {device.upper()}")
    if device == "cpu":
        print("⚠ Running on CPU. Training will take longer. Consider reducing epochs for a quick test run.")

    # 1. Load dataset
    print("\nLoading dataset...")
    dataset = load_hiner_dataset()

    val_key = "validation" if "validation" in dataset else "dev"

    # 2. Initialize mBERT model wrapper
    print("\nInitializing HindiTransformerNER (bert-base-multilingual-cased)...")
    ner_model = HindiTransformerNER(
        model_name="bert-base-multilingual-cased",
        max_length=128,
        seed=42,
    )

    # 3. Train
    print("\nStarting training...")
    train_results = ner_model.train(
        train_dataset=dataset["train"],
        val_dataset=dataset[val_key],
        output_dir=str(output_model_dir),
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
    )

    print("\nSaving final model checkpoint...")
    ner_model.save(str(output_model_dir))

    # 4. Evaluate on test split
    print("\nEvaluating on Test Set...")
    test_metrics = ner_model.evaluate(dataset["test"])

    print("\n" + "=" * 60)
    print("mBERT TEST EVALUATION REPORT (Entity-Level seqeval)")
    print("=" * 60)
    print(f"Overall Precision: {test_metrics['overall_precision']:.4f}")
    print(f"Overall Recall:    {test_metrics['overall_recall']:.4f}")
    print(f"Overall F1 Score:  {test_metrics['overall_f1']:.4f}")
    print("\nDetailed Per-Entity Breakdown:")
    print(test_metrics["report_str"])

    # 5. Save metrics and predictions
    metrics_path = PROJECT_ROOT / "results" / "mbert_test_metrics.json"
    save_results(test_metrics, str(metrics_path))

    test_tokens = [ex["tokens"] for ex in dataset["test"]]
    test_labels = [[ner_model.model.config.id2label[t] for t in ex["ner_tags"]] for ex in dataset["test"]]
    test_preds = ner_model.predict(test_tokens)

    preds_output = []
    for i in range(len(test_tokens)):
        preds_output.append({
            "tokens": test_tokens[i],
            "true_labels": test_labels[i],
            "pred_labels": test_preds[i],
        })

    pred_file = pred_dir / "mbert_predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(preds_output, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved mBERT predictions to: {pred_file}")

    print("\n✅ Step 3 mBERT Fine-tuning complete!")

if __name__ == "__main__":
    main()
