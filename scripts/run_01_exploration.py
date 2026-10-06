#!/usr/bin/env python3
"""
Step 1: Dataset Exploration and Statistics

Loads the Hindi NER dataset, computes statistics, creates visualizations,
and exports summary tables for reports.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data_utils import (
    load_hiner_dataset,
    compute_dataset_statistics,
    print_statistics,
    ID_TO_LABEL,
    LABEL_LIST,
)

def main():
    print("=" * 70)
    print("STEP 1: HINDI NER DATASET EXPLORATION")
    print("=" * 70)

    # Ensure output directories exist
    fig_dir = PROJECT_ROOT / "results" / "figures"
    tab_dir = PROJECT_ROOT / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    dataset = load_hiner_dataset()
    print(f"\n✓ Dataset loaded. Available splits: {list(dataset.keys())}")

    # 2. Compute statistics for all splits
    stats = {}
    for split_name in ["train", "validation", "test"]:
        if split_name in dataset:
            stats[split_name] = compute_dataset_statistics(dataset[split_name])
            print_statistics(stats[split_name], split_name)

    train_stats = stats.get("train", list(stats.values())[0])

    # 3. Entity Distribution Plot
    print("\nGenerating Entity Distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    entities = train_stats["entity_counts"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = axes[0].bar(list(entities.keys()), list(entities.values()), color=colors, alpha=0.85)
    axes[0].set_title("Entity Counts (Training Set)", fontweight="bold")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, entities.values()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{val:,}", ha="center", va="bottom", fontsize=10)

    label_dist = train_stats["label_distribution"]
    label_colors = {
        "O": "#999999",
        "B-PER": "#4C72B0", "I-PER": "#7BA3D4",
        "B-LOC": "#55A868", "I-LOC": "#88CC99",
        "B-ORG": "#C44E52", "I-ORG": "#E08888",
    }
    label_keys = [l for l in LABEL_LIST if l in label_dist]
    label_vals = [label_dist[l] for l in label_keys]
    bar_colors = [label_colors.get(l, "#999999") for l in label_keys]

    axes[1].bar(label_keys, label_vals, color=bar_colors, alpha=0.85)
    axes[1].set_title("Token-level Label Distribution", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    ent_plot_path = fig_dir / "entity_distribution.png"
    plt.savefig(ent_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {ent_plot_path}")

    # 4. Script Distribution Plot
    print("Generating Script Distribution plot...")
    script_dist = train_stats["script_distribution"]
    fig, ax = plt.subplots(figsize=(7, 5))
    script_colors = {
        "devanagari": "#FF6B35",
        "latin": "#004E89",
        "numeric": "#1A936F",
        "other": "#999999"
    }
    s_keys = list(script_dist.keys())
    s_vals = list(script_dist.values())
    s_colors = [script_colors.get(k, "#999999") for k in s_keys]

    ax.pie(s_vals, labels=s_keys, colors=s_colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 11})
    ax.set_title("Script Distribution of Tokens", fontweight="bold", fontsize=13)
    plt.tight_layout()
    script_plot_path = fig_dir / "script_distribution.png"
    plt.savefig(script_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {script_plot_path}")

    # 5. Sentence Length Distribution Plot
    print("Generating Sentence Length Distribution plot...")
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    sent_lengths = [len(ex["tokens"]) for ex in dataset[split_name]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sent_lengths, bins=50, color="#4C72B0", alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(sent_lengths), color="red", linestyle="--",
               label=f"Mean: {np.mean(sent_lengths):.1f}")
    ax.axvline(np.median(sent_lengths), color="orange", linestyle="--",
               label=f"Median: {np.median(sent_lengths):.1f}")
    ax.set_xlabel("Sentence Length (tokens)")
    ax.set_ylabel("Frequency")
    ax.set_title("Sentence Length Distribution (Training Set)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    length_plot_path = fig_dir / "sentence_lengths.png"
    plt.savefig(length_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {length_plot_path}")

    # 6. Summary Table
    summary_rows = []
    for s_name, s in stats.items():
        summary_rows.append({
            "Split": s_name.capitalize(),
            "Sentences": f"{s['num_sentences']:,}",
            "Tokens": f"{s['num_tokens']:,}",
            "Avg Length": s["avg_sentence_length"],
            "PER": f"{s['entity_counts']['PER']:,}",
            "LOC": f"{s['entity_counts']['LOC']:,}",
            "ORG": f"{s['entity_counts']['ORG']:,}",
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n📋 Dataset Summary Table:")
    print(summary_df.to_string(index=False))

    summary_csv = tab_dir / "dataset_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved table to: {summary_csv}")
    print("\n✅ Step 1 Data Exploration complete!")

if __name__ == "__main__":
    main()
