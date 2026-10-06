#!/usr/bin/env python3
"""
Master runner for the Hindi NER project pipeline.

Usage:
    python scripts/run_all.py --step 1     # Run Data Exploration
    python scripts/run_all.py --step 2     # Train & Evaluate CRF
    python scripts/run_all.py --step 3     # Fine-tune & Evaluate mBERT
    python scripts/run_all.py --step 4     # Run Comparative Evaluation
    python scripts/run_all.py --step 5     # Run Qualitative Error Analysis
    python scripts/run_all.py --all        # Run all steps sequentially
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Run Hindi NER Project Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], help="Run a specific pipeline step")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline end-to-end")

    args = parser.parse_args()

    if not args.step and not args.all:
        parser.print_help()
        print("\nQuick start recommendation:")
        print("  python scripts/run_01_exploration.py  # Start by exploring the dataset")
        print("  python scripts/run_02_crf.py          # Fast CPU baseline training")
        return

    steps_to_run = [args.step] if args.step else [1, 2, 3, 4, 5]

    for step in steps_to_run:
        print(f"\n{'='*70}\nLAUNCHING STEP {step}\n{'='*70}")
        if step == 1:
            from scripts.run_01_exploration import main as run_1
            run_1()
        elif step == 2:
            from scripts.run_02_crf import main as run_2
            run_2()
        elif step == 3:
            from scripts.run_03_mbert import main as run_3
            run_3()
        elif step == 4:
            from scripts.run_04_evaluation import main as run_4
            run_4()
        elif step == 5:
            from scripts.run_05_error_analysis import main as run_5
            run_5()

if __name__ == "__main__":
    main()
