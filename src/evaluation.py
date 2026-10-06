"""
Evaluation utilities for Hindi NER.

Provides entity-level metrics, comparison tables, and visualization
for CRF vs mBERT results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from seqeval.metrics import classification_report, f1_score

# Use a font that supports Devanagari if available
matplotlib.rcParams['font.family'] = 'sans-serif'


# ──────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────

def compute_entity_metrics(
    true_labels: List[List[str]], 
    pred_labels: List[List[str]]
) -> dict:
    """
    Compute entity-level precision, recall, F1 using seqeval.
    
    Args:
        true_labels: Ground truth BIO label sequences
        pred_labels: Predicted BIO label sequences
        
    Returns:
        Dictionary with overall and per-entity metrics
    """
    from seqeval.metrics import precision_score, recall_score, f1_score
    
    return {
        "overall_precision": precision_score(true_labels, pred_labels),
        "overall_recall": recall_score(true_labels, pred_labels),
        "overall_f1": f1_score(true_labels, pred_labels),
        "report": classification_report(true_labels, pred_labels, output_dict=True),
        "report_str": classification_report(true_labels, pred_labels),
    }


def build_comparison_table(
    crf_metrics: dict, 
    mbert_metrics: dict
) -> pd.DataFrame:
    """
    Build a comparison table between CRF and mBERT results.
    
    Returns a DataFrame like:
    
    | Entity | CRF P | CRF R | CRF F1 | mBERT P | mBERT R | mBERT F1 |
    |--------|-------|-------|--------|---------|---------|----------|
    | PER    | ...   | ...   | ...    | ...     | ...     | ...      |
    | LOC    | ...   | ...   | ...    | ...     | ...     | ...      |
    | ORG    | ...   | ...   | ...    | ...     | ...     | ...      |
    | Overall| ...   | ...   | ...    | ...     | ...     | ...      |
    """
    rows = []
    
    entities = ["PER", "LOC", "ORG"]
    
    for entity in entities:
        crf_report = crf_metrics.get("report", {})
        mbert_report = mbert_metrics.get("report", {})
        
        crf_entity = crf_report.get(entity, {})
        mbert_entity = mbert_report.get(entity, {})
        
        rows.append({
            "Entity": entity,
            "CRF Precision": round(crf_entity.get("precision", 0) * 100, 2),
            "CRF Recall": round(crf_entity.get("recall", 0) * 100, 2),
            "CRF F1": round(crf_entity.get("f1-score", 0) * 100, 2),
            "mBERT Precision": round(mbert_entity.get("precision", 0) * 100, 2),
            "mBERT Recall": round(mbert_entity.get("recall", 0) * 100, 2),
            "mBERT F1": round(mbert_entity.get("f1-score", 0) * 100, 2),
        })
    
    # Overall row
    rows.append({
        "Entity": "Overall",
        "CRF Precision": round(crf_metrics.get("overall_precision", 0) * 100, 2),
        "CRF Recall": round(crf_metrics.get("overall_recall", 0) * 100, 2),
        "CRF F1": round(crf_metrics.get("overall_f1", 0) * 100, 2),
        "mBERT Precision": round(mbert_metrics.get("overall_precision", 0) * 100, 2),
        "mBERT Recall": round(mbert_metrics.get("overall_recall", 0) * 100, 2),
        "mBERT F1": round(mbert_metrics.get("overall_f1", 0) * 100, 2),
    })
    
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Entity-level confusion analysis
# ──────────────────────────────────────────────

def extract_entities(tokens: List[str], labels: List[str]) -> List[Tuple[str, str, int, int]]:
    """
    Extract named entities from BIO-tagged sequence.
    
    Returns list of (entity_text, entity_type, start_idx, end_idx).
    """
    entities = []
    current_entity = None
    current_type = None
    start_idx = None
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            # Save previous entity if exists
            if current_entity is not None:
                entities.append((
                    " ".join(current_entity), 
                    current_type, 
                    start_idx, 
                    i - 1
                ))
            # Start new entity
            current_entity = [token]
            current_type = label[2:]
            start_idx = i
            
        elif label.startswith("I-") and current_entity is not None:
            # Continue current entity
            current_entity.append(token)
            
        else:
            # O tag or I- without B-
            if current_entity is not None:
                entities.append((
                    " ".join(current_entity), 
                    current_type, 
                    start_idx, 
                    i - 1
                ))
                current_entity = None
                current_type = None
                start_idx = None
    
    # Don't forget the last entity
    if current_entity is not None:
        entities.append((
            " ".join(current_entity), 
            current_type, 
            start_idx, 
            len(tokens) - 1
        ))
    
    return entities


def compute_entity_confusion_matrix(
    all_predictions: List[Dict],
) -> pd.DataFrame:
    """
    Compute confusion between entity types.
    
    For each predicted entity, check what the true label was.
    
    Returns:
        Confusion matrix as DataFrame (rows=true, cols=predicted)
    """
    entity_types = ["PER", "LOC", "ORG", "O"]
    confusion = defaultdict(lambda: defaultdict(int))
    
    for example in all_predictions:
        tokens = example["tokens"]
        true_labels = example["true_labels"]
        pred_labels = example["pred_labels"]
        
        true_entities = extract_entities(tokens, true_labels)
        pred_entities = extract_entities(tokens, pred_labels)
        
        # Match predicted entities to true entities by position overlap
        for pred_text, pred_type, pred_start, pred_end in pred_entities:
            matched = False
            for true_text, true_type, true_start, true_end in true_entities:
                # Check if there's overlap
                if pred_start <= true_end and pred_end >= true_start:
                    confusion[true_type][pred_type] += 1
                    matched = True
                    break
            if not matched:
                confusion["O"][pred_type] += 1  # False positive
        
        # Find missed entities (false negatives)
        for true_text, true_type, true_start, true_end in true_entities:
            matched = False
            for pred_text, pred_type, pred_start, pred_end in pred_entities:
                if pred_start <= true_end and pred_end >= true_start:
                    matched = True
                    break
            if not matched:
                confusion[true_type]["MISSED"] += 1
    
    # Convert to DataFrame
    all_cols = ["PER", "LOC", "ORG", "MISSED"]
    all_rows = ["PER", "LOC", "ORG", "O"]
    
    matrix = []
    for row in all_rows:
        matrix.append([confusion[row][col] for col in all_cols])
    
    return pd.DataFrame(matrix, index=all_rows, columns=all_cols)


# ──────────────────────────────────────────────
# Performance by entity length
# ──────────────────────────────────────────────

def performance_by_entity_length(
    all_predictions: List[Dict],
) -> pd.DataFrame:
    """
    Analyze how model performance varies with entity length.
    
    Returns DataFrame with columns: length, total, correct, f1
    """
    length_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for example in all_predictions:
        tokens = example["tokens"]
        true_entities = extract_entities(tokens, example["true_labels"])
        pred_entities = extract_entities(tokens, example["pred_labels"])
        
        for true_text, true_type, true_start, true_end in true_entities:
            length = true_end - true_start + 1
            length_stats[length]["total"] += 1
            
            # Check if correctly predicted (exact match)
            for pred_text, pred_type, pred_start, pred_end in pred_entities:
                if (pred_start == true_start and 
                    pred_end == true_end and 
                    pred_type == true_type):
                    length_stats[length]["correct"] += 1
                    break
    
    rows = []
    for length in sorted(length_stats.keys()):
        stats = length_stats[length]
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        rows.append({
            "Entity Length (tokens)": length,
            "Total Entities": stats["total"],
            "Correctly Predicted": stats["correct"],
            "Accuracy": round(accuracy * 100, 2),
        })
    
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_f1_comparison(
    crf_metrics: dict, 
    mbert_metrics: dict, 
    save_path: Optional[str] = None
):
    """
    Bar chart comparing CRF vs mBERT F1 scores per entity type.
    """
    entities = ["PER", "LOC", "ORG", "Overall"]
    
    crf_f1s = []
    mbert_f1s = []
    
    for entity in ["PER", "LOC", "ORG"]:
        crf_f1s.append(
            crf_metrics.get("report", {}).get(entity, {}).get("f1-score", 0) * 100
        )
        mbert_f1s.append(
            mbert_metrics.get("report", {}).get(entity, {}).get("f1-score", 0) * 100
        )
    
    crf_f1s.append(crf_metrics.get("overall_f1", 0) * 100)
    mbert_f1s.append(mbert_metrics.get("overall_f1", 0) * 100)
    
    x = np.arange(len(entities))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, crf_f1s, width, label='CRF', color='#4C72B0', alpha=0.85)
    bars2 = ax.bar(x + width/2, mbert_f1s, width, label='mBERT', color='#DD8452', alpha=0.85)
    
    ax.set_xlabel('Entity Type', fontsize=12)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('CRF vs mBERT: Entity-level F1 Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(entities)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    confusion_df: pd.DataFrame, 
    model_name: str = "",
    save_path: Optional[str] = None
):
    """Plot entity confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        confusion_df, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    title = f'Entity Confusion Matrix'
    if model_name:
        title += f' — {model_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_training_curves(
    log_history: List[dict], 
    save_path: Optional[str] = None
):
    """Plot training and validation loss/F1 curves from Trainer log history."""
    train_loss = []
    eval_loss = []
    eval_f1 = []
    epochs_train = []
    epochs_eval = []
    
    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_loss.append(entry["loss"])
            epochs_train.append(entry["epoch"])
        if "eval_loss" in entry and "epoch" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_f1.append(entry.get("eval_f1", 0))
            epochs_eval.append(entry["epoch"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs_train, train_loss, 'b-', alpha=0.5, label='Train Loss')
    ax1.plot(epochs_eval, eval_loss, 'r-o', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # F1 curve
    ax2.plot(epochs_eval, [f * 100 for f in eval_f1], 'g-o', label='Validation F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score (%)')
    ax2.set_title('Validation F1 Score', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle('mBERT Fine-tuning Progress', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def save_results(results: dict, filepath: str):
    """Save results dictionary to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"✓ Results saved to {filepath}")
