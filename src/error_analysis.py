"""
Error analysis for Hindi NER.

Categorizes and analyzes errors made by CRF and mBERT,
with focus on Hindi-specific error patterns.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

from src.data_utils import is_devanagari_word, is_latin_word, get_script_type
from src.evaluation import extract_entities


# ──────────────────────────────────────────────
# Error types
# ──────────────────────────────────────────────

class ErrorType:
    """Error categories for Hindi NER."""
    BOUNDARY = "Boundary Error"
    TYPE_CONFUSION = "Type Confusion"
    MISSED_ENTITY = "Missed Entity (FN)"
    SPURIOUS_ENTITY = "Spurious Entity (FP)"
    MULTI_WORD = "Multi-word Entity Error"
    SCRIPT_MIXING = "Script/Code-mixing Error"
    RARE_NAME = "Rare/Unseen Name Error"


# ──────────────────────────────────────────────
# Error extraction
# ──────────────────────────────────────────────

def find_errors(
    tokens: List[str],
    true_labels: List[str],
    pred_labels: List[str],
) -> List[Dict]:
    """
    Find and categorize all errors in a single sentence.
    
    Args:
        tokens: Original word tokens
        true_labels: Ground truth BIO labels
        pred_labels: Predicted BIO labels
        
    Returns:
        List of error dictionaries with error details
    """
    errors = []
    
    true_entities = extract_entities(tokens, true_labels)
    pred_entities = extract_entities(tokens, pred_labels)
    
    # Track which entities have been matched
    matched_true = set()
    matched_pred = set()
    
    # Check each predicted entity against true entities
    for pi, (pred_text, pred_type, pred_start, pred_end) in enumerate(pred_entities):
        best_match = None
        best_overlap = 0
        
        for ti, (true_text, true_type, true_start, true_end) in enumerate(true_entities):
            # Calculate overlap
            overlap_start = max(pred_start, true_start)
            overlap_end = min(pred_end, true_end)
            overlap = max(0, overlap_end - overlap_start + 1)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = (ti, true_text, true_type, true_start, true_end)
        
        if best_match is not None:
            ti, true_text, true_type, true_start, true_end = best_match
            matched_true.add(ti)
            matched_pred.add(pi)
            
            # Exact match → no error
            if (pred_start == true_start and pred_end == true_end and 
                pred_type == true_type):
                continue
            
            # Type confusion: same span, different type
            if pred_start == true_start and pred_end == true_end:
                error_type = ErrorType.TYPE_CONFUSION
            # Boundary error: overlapping but different boundaries
            else:
                error_type = ErrorType.BOUNDARY
            
            # Check if it's a multi-word entity error
            true_length = true_end - true_start + 1
            pred_length = pred_end - pred_start + 1
            if true_length > 1 or pred_length > 1:
                if pred_start != true_start or pred_end != true_end:
                    error_type = ErrorType.MULTI_WORD
            
            # Check for script mixing
            entity_tokens = tokens[true_start:true_end+1]
            scripts = set(get_script_type(t) for t in entity_tokens)
            if len(scripts) > 1 or "latin" in scripts:
                error_type = ErrorType.SCRIPT_MIXING
            
            errors.append({
                "error_type": error_type,
                "true_entity": true_text,
                "true_type": true_type,
                "true_span": (true_start, true_end),
                "pred_entity": pred_text,
                "pred_type": pred_type,
                "pred_span": (pred_start, pred_end),
                "context": " ".join(tokens),
                "tokens": tokens,
                "true_labels": true_labels,
                "pred_labels": pred_labels,
            })
        else:
            # Spurious entity (false positive)
            errors.append({
                "error_type": ErrorType.SPURIOUS_ENTITY,
                "true_entity": None,
                "true_type": None,
                "true_span": None,
                "pred_entity": pred_text,
                "pred_type": pred_type,
                "pred_span": (pred_start, pred_end),
                "context": " ".join(tokens),
                "tokens": tokens,
                "true_labels": true_labels,
                "pred_labels": pred_labels,
            })
            matched_pred.add(pi)
    
    # Missed entities (false negatives)
    for ti, (true_text, true_type, true_start, true_end) in enumerate(true_entities):
        if ti not in matched_true:
            errors.append({
                "error_type": ErrorType.MISSED_ENTITY,
                "true_entity": true_text,
                "true_type": true_type,
                "true_span": (true_start, true_end),
                "pred_entity": None,
                "pred_type": None,
                "pred_span": None,
                "context": " ".join(tokens),
                "tokens": tokens,
                "true_labels": true_labels,
                "pred_labels": pred_labels,
            })
    
    return errors


def analyze_all_errors(predictions: List[Dict]) -> List[Dict]:
    """
    Run error analysis on all predictions.
    
    Args:
        predictions: List of dicts with 'tokens', 'true_labels', 'pred_labels'
        
    Returns:
        List of all errors found across all sentences
    """
    all_errors = []
    
    for example in predictions:
        errors = find_errors(
            example["tokens"],
            example["true_labels"],
            example["pred_labels"],
        )
        all_errors.extend(errors)
    
    return all_errors


# ──────────────────────────────────────────────
# Error statistics
# ──────────────────────────────────────────────

def error_statistics(errors: List[Dict]) -> Dict:
    """
    Compute summary statistics over all errors.
    
    Returns:
        Dictionary with error counts, breakdowns, and examples
    """
    total = len(errors)
    
    # Count by error type
    type_counts = Counter(e["error_type"] for e in errors)
    
    # Count by entity type (true entity type involved)
    entity_type_counts = Counter()
    for e in errors:
        if e["true_type"]:
            entity_type_counts[e["true_type"]] += 1
        elif e["pred_type"]:
            entity_type_counts[f"FP-{e['pred_type']}"] += 1
    
    # Type confusion matrix
    confusion_pairs = Counter()
    for e in errors:
        if e["error_type"] == ErrorType.TYPE_CONFUSION:
            pair = f"{e['true_type']} → {e['pred_type']}"
            confusion_pairs[pair] += 1
    
    # Script distribution of errors
    script_errors = Counter()
    for e in errors:
        entity = e.get("true_entity") or e.get("pred_entity") or ""
        words = entity.split()
        scripts = set(get_script_type(w) for w in words if w)
        if "latin" in scripts and "devanagari" in scripts:
            script_errors["mixed"] += 1
        elif "latin" in scripts:
            script_errors["latin"] += 1
        elif "devanagari" in scripts:
            script_errors["devanagari"] += 1
        else:
            script_errors["other"] += 1
    
    return {
        "total_errors": total,
        "error_type_counts": dict(type_counts.most_common()),
        "entity_type_counts": dict(entity_type_counts.most_common()),
        "type_confusion_pairs": dict(confusion_pairs.most_common()),
        "script_distribution": dict(script_errors),
    }


def print_error_summary(stats: Dict, model_name: str = ""):
    """Pretty-print error analysis summary."""
    header = f" Error Analysis — {model_name} " if model_name else " Error Analysis "
    print(f"\n{'='*60}")
    print(f"{header:=^60}")
    print(f"{'='*60}")
    print(f"\n  Total errors: {stats['total_errors']}")
    
    print(f"\n  Errors by type:")
    for error_type, count in stats['error_type_counts'].items():
        pct = count / stats['total_errors'] * 100
        print(f"    {error_type:30s}: {count:5d}  ({pct:5.1f}%)")
    
    print(f"\n  Errors by entity type:")
    for entity_type, count in stats['entity_type_counts'].items():
        print(f"    {entity_type:10s}: {count:5d}")
    
    if stats['type_confusion_pairs']:
        print(f"\n  Type confusion pairs:")
        for pair, count in stats['type_confusion_pairs'].items():
            print(f"    {pair:20s}: {count:5d}")
    
    print(f"\n  Script distribution of errors:")
    for script, count in stats['script_distribution'].items():
        print(f"    {script:12s}: {count:5d}")
    print()


def get_interesting_examples(
    errors: List[Dict], 
    n: int = 10
) -> List[Dict]:
    """
    Select the most interesting/illustrative error examples
    for the report and presentation.
    
    Prioritizes:
    1. Type confusion examples
    2. Multi-word entity errors
    3. Script/code-mixing errors
    4. Boundary errors
    """
    # Group by error type
    by_type = defaultdict(list)
    for e in errors:
        by_type[e["error_type"]].append(e)
    
    # Select from each category
    selected = []
    priority = [
        ErrorType.TYPE_CONFUSION,
        ErrorType.MULTI_WORD,
        ErrorType.SCRIPT_MIXING,
        ErrorType.BOUNDARY,
        ErrorType.MISSED_ENTITY,
        ErrorType.SPURIOUS_ENTITY,
    ]
    
    per_type = max(1, n // len(priority))
    
    for error_type in priority:
        examples = by_type.get(error_type, [])
        selected.extend(examples[:per_type])
    
    return selected[:n]


def format_error_example(error: Dict) -> str:
    """Format a single error example for display."""
    lines = []
    lines.append(f"Error Type: {error['error_type']}")
    lines.append(f"Context:    {error['context']}")
    
    if error['true_entity']:
        lines.append(f"True:       \"{error['true_entity']}\" → {error['true_type']}")
    
    if error['pred_entity']:
        lines.append(f"Predicted:  \"{error['pred_entity']}\" → {error['pred_type']}")
    elif error['true_entity']:
        lines.append(f"Predicted:  (missed)")
    
    # Show token-level alignment
    lines.append("  Token-level:")
    tokens = error['tokens']
    true_labels = error['true_labels']
    pred_labels = error['pred_labels']
    
    for t, tl, pl in zip(tokens, true_labels, pred_labels):
        marker = "  " if tl == pl else "→ "
        lines.append(f"    {marker}{t:20s}  true={tl:8s}  pred={pl:8s}")
    
    return "\n".join(lines)


def save_error_analysis(
    errors: List[Dict],
    stats: Dict,
    filepath: str,
):
    """Save error analysis results to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Serialize errors (convert tuples to lists for JSON)
    serializable_errors = []
    for e in errors:
        se = dict(e)
        if se.get("true_span"):
            se["true_span"] = list(se["true_span"])
        if se.get("pred_span"):
            se["pred_span"] = list(se["pred_span"])
        serializable_errors.append(se)
    
    output = {
        "statistics": stats,
        "errors": serializable_errors[:200],  # Save top 200 errors
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Error analysis saved to {filepath}")
