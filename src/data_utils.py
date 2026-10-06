"""
Data utilities for Hindi NER project.

Handles loading, preprocessing, and formatting the HiNER dataset
for both CRF and Transformer pipelines.
"""

import re
import yaml
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────
# Label mappings
# ──────────────────────────────────────────────

# Standard BIO labels for 3-entity NER (aligned with HiNER dataset)
LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────

def load_hiner_dataset() -> dict:
    """
    Load the HiNER-collapsed dataset from HuggingFace.
    Loads raw JSON data files directly to remain compatible with modern `datasets`
    versions where custom Python loading scripts are no longer supported.
    Falls back to WikiANN Hindi if HiNER is not reachable.
    
    Returns:
        Dictionary with 'train', 'validation', 'test' splits.
    """
    try:
        print("Loading HiNER-collapsed dataset from HuggingFace JSON data files...")
        base_url = "https://huggingface.co/datasets/cfilt/HiNER-collapsed/resolve/main/data"
        data_files = {
            "train": f"{base_url}/train.json",
            "validation": f"{base_url}/validation.json",
            "test": f"{base_url}/test.json",
        }
        dataset = load_dataset("json", data_files=data_files)
        print("✓ HiNER loaded successfully via JSON data files.")
        print(f"  Splits: {list(dataset.keys())}")
        for split in dataset:
            print(f"  {split}: {len(dataset[split])} examples")
        return dataset
    except Exception as e:
        print(f"✗ Direct JSON load failed: {e}")
        print("Falling back to WikiANN Hindi...")
        return load_wikiann_hindi()


def load_wikiann_hindi() -> dict:
    """
    Load WikiANN Hindi dataset as fallback.
    Remaps WikiANN label IDs to match our standard LABEL_LIST:
    WikiANN: 0: O, 1: B-PER, 2: I-PER, 3: B-ORG, 4: I-ORG, 5: B-LOC, 6: I-LOC
    Target:  0: O, 1: B-PER, 2: I-PER, 3: B-LOC, 4: I-LOC, 5: B-ORG, 6: I-ORG
    """
    dataset = load_dataset("wikiann", "hi")
    print("✓ WikiANN Hindi loaded successfully.")
    
    # Remap IDs to match HiNER ordering
    wikiann_to_hiner = {0: 0, 1: 1, 2: 2, 3: 5, 4: 6, 5: 3, 6: 4}
    
    def remap_tags(example):
        example["ner_tags"] = [wikiann_to_hiner.get(t, 0) for t in example["ner_tags"]]
        return example
        
    dataset = dataset.map(remap_tags)
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} examples")
    return dataset


# ──────────────────────────────────────────────
# Preprocessing utilities
# ──────────────────────────────────────────────

def is_devanagari(char: str) -> bool:
    """Check if a character is in the Devanagari Unicode block."""
    return '\u0900' <= char <= '\u097F'


def is_devanagari_word(word: str) -> bool:
    """Check if a word is primarily Devanagari script."""
    if not word:
        return False
    devanagari_count = sum(1 for c in word if is_devanagari(c))
    return devanagari_count > len(word) / 2


def is_latin_word(word: str) -> bool:
    """Check if a word is primarily Latin/English script."""
    if not word:
        return False
    latin_count = sum(1 for c in word if c.isascii() and c.isalpha())
    return latin_count > len(word) / 2


def get_word_shape(word: str) -> str:
    """
    Get the shape of a word for CRF features.
    
    Examples:
        "Hello" -> "Xxxxx"
        "123"   -> "ddd"
        "मोदी"  -> "हहहह" (Devanagari placeholder)
    """
    shape = []
    for c in word:
        if is_devanagari(c):
            shape.append('ह')  # Devanagari placeholder
        elif c.isupper():
            shape.append('X')
        elif c.islower():
            shape.append('x')
        elif c.isdigit():
            shape.append('d')
        else:
            shape.append(c)
    return ''.join(shape)


def get_script_type(word: str) -> str:
    """Determine the script type of a word."""
    if is_devanagari_word(word):
        return "devanagari"
    elif is_latin_word(word):
        return "latin"
    elif word.isdigit():
        return "numeric"
    else:
        return "other"


# ──────────────────────────────────────────────
# Data formatting for CRF
# ──────────────────────────────────────────────

def format_for_crf(dataset_split) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Convert a HuggingFace dataset split into the format
    expected by sklearn-crfsuite: lists of (tokens, labels).
    
    Returns:
        sentences: List of lists of tokens
        labels: List of lists of BIO label strings
    """
    sentences = []
    labels = []
    
    for example in dataset_split:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        
        # Convert numeric tag IDs to string labels
        tag_strings = [ID_TO_LABEL.get(tag, "O") for tag in ner_tags]
        
        sentences.append(tokens)
        labels.append(tag_strings)
    
    return sentences, labels


# ──────────────────────────────────────────────
# Data formatting for mBERT (Transformer)
# ──────────────────────────────────────────────

def tokenize_and_align_labels(examples, tokenizer, max_length: int = 128):
    """
    Tokenize sentences and align NER labels to subword tokens.
    
    Transformers use subword tokenization, so a single word may be
    split into multiple subwords. We assign the real label to the
    FIRST subword and -100 (ignored in loss) to the rest.
    
    Example:
        Word:     "कोहली"
        Subwords: ["को", "##हल", "##ी"]
        Labels:   [B-PER, -100,   -100]
    
    Args:
        examples: Batch from HuggingFace dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized inputs with aligned label IDs
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        padding="max_length",
    )
    
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a new word → real label
                label_ids.append(labels[word_idx])
            else:
                # Continuation subword → ignore in loss
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        all_labels.append(label_ids)
    
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# ──────────────────────────────────────────────
# Dataset statistics
# ──────────────────────────────────────────────

def compute_dataset_statistics(dataset_split) -> dict:
    """
    Compute comprehensive statistics for a dataset split.
    
    Returns dict with:
        - num_sentences
        - num_tokens
        - avg_sentence_length
        - entity_distribution (count per label)
        - entity_counts (PER, LOC, ORG counts by B- tags)
        - script_distribution
    """
    num_sentences = len(dataset_split)
    total_tokens = 0
    label_counts = {}
    entity_counts = {"PER": 0, "LOC": 0, "ORG": 0}
    script_counts = {"devanagari": 0, "latin": 0, "numeric": 0, "other": 0}
    sentence_lengths = []
    
    for example in dataset_split:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        
        total_tokens += len(tokens)
        sentence_lengths.append(len(tokens))
        
        for token, tag_id in zip(tokens, ner_tags):
            # Label distribution
            label = ID_TO_LABEL.get(tag_id, "O")
            label_counts[label] = label_counts.get(label, 0) + 1
            
            # Entity counts (only B- tags to count unique entities)
            if label.startswith("B-"):
                entity_type = label[2:]
                if entity_type in entity_counts:
                    entity_counts[entity_type] += 1
            
            # Script distribution
            script = get_script_type(token)
            script_counts[script] += 1
    
    return {
        "num_sentences": num_sentences,
        "num_tokens": total_tokens,
        "avg_sentence_length": round(np.mean(sentence_lengths), 2),
        "max_sentence_length": max(sentence_lengths),
        "min_sentence_length": min(sentence_lengths),
        "label_distribution": dict(sorted(label_counts.items())),
        "entity_counts": entity_counts,
        "script_distribution": script_counts,
    }


def print_statistics(stats: dict, split_name: str = ""):
    """Pretty-print dataset statistics."""
    header = f" {split_name} Statistics " if split_name else " Dataset Statistics "
    print(f"\n{'='*50}")
    print(f"{header:=^50}")
    print(f"{'='*50}")
    print(f"  Sentences:          {stats['num_sentences']:,}")
    print(f"  Tokens:             {stats['num_tokens']:,}")
    print(f"  Avg sentence len:   {stats['avg_sentence_length']}")
    print(f"  Min sentence len:   {stats['min_sentence_length']}")
    print(f"  Max sentence len:   {stats['max_sentence_length']}")
    
    print(f"\n  Entity counts (unique entities):")
    for entity, count in stats['entity_counts'].items():
        print(f"    {entity:12s}: {count:,}")
    
    print(f"\n  Label distribution (all tokens):")
    for label, count in stats['label_distribution'].items():
        pct = count / stats['num_tokens'] * 100
        print(f"    {label:12s}: {count:>8,}  ({pct:5.1f}%)")
    
    print(f"\n  Script distribution:")
    for script, count in stats['script_distribution'].items():
        pct = count / stats['num_tokens'] * 100
        print(f"    {script:12s}: {count:>8,}  ({pct:5.1f}%)")
    print()
