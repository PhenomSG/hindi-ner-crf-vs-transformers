# Hindi Named Entity Recognition: Traditional vs Transformer-Based Approaches

> **Develop a Named Entity Recognition (NER) system for Hindi text that identifies and classifies named entities such as persons, locations, and organizations. The project compares traditional NER (CRF) with a Transformer-based model (mBERT) and analyzes their performance and language-specific errors on Hindi text.**

## Project Structure

```
ner_using_transformers/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_utils.py           # Dataset loading & preprocessing
│   ├── feature_extraction.py   # CRF feature engineering
│   ├── crf_model.py            # CRF training & inference
│   ├── transformer_ner.py      # mBERT fine-tuning pipeline
│   ├── evaluation.py           # seqeval metrics & confusion matrix
│   └── error_analysis.py       # Error categorization & examples
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_crf_baseline.ipynb
│   ├── 03_mbert_finetuning.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_error_analysis.ipynb
├── configs/
│   └── config.yaml
├── results/
│   ├── figures/
│   ├── tables/
│   └── predictions/
└── models/
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## How to Run

You can run each stage directly using standalone Python scripts or open the corresponding Jupyter notebooks:

### Option 1: Fast CLI Runner (Recommended)

```bash
# Step 1: Explore dataset and generate statistics / plots
python scripts/run_01_exploration.py

# Step 2: Train and evaluate traditional CRF baseline (fast on CPU)
python scripts/run_02_crf.py

# Step 3: Fine-tune mBERT transformer model
python scripts/run_03_mbert.py

# Step 4: Compare CRF vs mBERT performance & generate plots
python scripts/run_04_evaluation.py

# Step 5: Run Hindi qualitative error analysis & case studies
python scripts/run_05_error_analysis.py
```

Or run all steps with:
```bash
python scripts/run_all.py --all
```

### Option 2: Jupyter Notebooks

If you prefer running interactively in Jupyter / VS Code:
1. Install ipykernel in your venv: `uv pip install ipykernel`
2. Open [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) and select the `.venv` kernel.

## Models

| Model | Type | Description |
|-------|------|-------------|
| CRF | Traditional | Feature-based Conditional Random Field |
| mBERT | Transformer | `bert-base-multilingual-cased` fine-tuned for Hindi NER |

## Dataset

**HiNER** — A Hindi Named Entity Recognition dataset with PER, LOC, ORG entity annotations.

## Evaluation

- Entity-level Precision, Recall, F1 (via `seqeval`)
- Per-entity-type breakdown (PER / LOC / ORG)
- Qualitative error analysis on Hindi-specific patterns
