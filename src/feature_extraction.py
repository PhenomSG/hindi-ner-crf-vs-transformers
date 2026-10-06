"""
Feature extraction for CRF-based Hindi NER.

Extracts handcrafted features from Hindi tokens and their context,
including Devanagari-specific features, word shape, and n-gram features.
"""

import re
from typing import List, Dict
from src.data_utils import is_devanagari_word, is_latin_word, get_word_shape, get_script_type


# ──────────────────────────────────────────────
# Hindi-specific feature helpers
# ──────────────────────────────────────────────

# Common Hindi postpositions (not entities themselves, but useful context)
HINDI_POSTPOSITIONS = {
    "में", "से", "को", "का", "की", "के", "पर", "तक",
    "ने", "और", "है", "हैं", "था", "थी", "थे",
}

# Common Hindi honorifics (often precede person names)
HINDI_HONORIFICS = {
    "श्री", "श्रीमती", "डॉ", "डॉ.", "प्रो", "प्रो.",
    "मि.", "मिस", "सुश्री", "पंडित", "महात्मा",
}

# Common Hindi title words (not entities, but context clues)
HINDI_TITLE_WORDS = {
    "प्रधानमंत्री", "राष्ट्रपति", "मुख्यमंत्री", "मंत्री",
    "अध्यक्ष", "निदेशक", "सचिव", "अधिकारी", "नेता",
}


def word_features(word: str) -> Dict[str, object]:
    """
    Extract features from a single word.
    
    Features include:
        - The word itself (lowercased for CRF)
        - Suffixes and prefixes
        - Script type (Devanagari, Latin, etc.)
        - Word shape
        - Special flags (postposition, honorific, digit, etc.)
    """
    features = {
        "word.lower": word.lower(),
        "word.length": len(word),
        
        # Suffixes (important for Hindi morphology)
        "word.suffix_1": word[-1:] if len(word) >= 1 else "",
        "word.suffix_2": word[-2:] if len(word) >= 2 else "",
        "word.suffix_3": word[-3:] if len(word) >= 3 else "",
        
        # Prefixes
        "word.prefix_1": word[:1] if len(word) >= 1 else "",
        "word.prefix_2": word[:2] if len(word) >= 2 else "",
        "word.prefix_3": word[:3] if len(word) >= 3 else "",
        
        # Script type
        "word.is_devanagari": is_devanagari_word(word),
        "word.is_latin": is_latin_word(word),
        "word.is_digit": word.isdigit(),
        "word.is_punctuation": not word.isalnum() and not is_devanagari_word(word),
        
        # Word shape
        "word.shape": get_word_shape(word),
        "word.script": get_script_type(word),
        
        # Case features (relevant for English words in Hindi text)
        "word.is_title": word.istitle(),
        "word.is_upper": word.isupper(),
        
        # Hindi-specific
        "word.is_postposition": word in HINDI_POSTPOSITIONS,
        "word.is_honorific": word in HINDI_HONORIFICS,
        "word.is_title_word": word in HINDI_TITLE_WORDS,
        
        # Has any digit
        "word.has_digit": bool(re.search(r'\d', word)),
        # Has hyphen (e.g., compound words)
        "word.has_hyphen": '-' in word,
    }
    
    return features


def sentence_to_features(sentence: List[str], context_window: int = 2) -> List[Dict]:
    """
    Extract features for all tokens in a sentence, including context features.
    
    For each token, we extract:
        1. Its own word-level features
        2. Features of surrounding tokens (within context_window)
        3. Beginning/end of sentence flags
    
    Args:
        sentence: List of tokens
        context_window: Number of tokens before/after to include as context
        
    Returns:
        List of feature dictionaries (one per token)
    """
    sentence_features = []
    
    for i in range(len(sentence)):
        features = {}
        
        # Current word features
        current_features = word_features(sentence[i])
        for key, val in current_features.items():
            features[key] = val
        
        # Beginning / end of sentence
        features["BOS"] = (i == 0)
        features["EOS"] = (i == len(sentence) - 1)
        
        # Context features: words before
        for offset in range(1, context_window + 1):
            if i - offset >= 0:
                prev_features = word_features(sentence[i - offset])
                for key, val in prev_features.items():
                    features[f"-{offset}:{key}"] = val
            else:
                features[f"-{offset}:BOS"] = True
        
        # Context features: words after
        for offset in range(1, context_window + 1):
            if i + offset < len(sentence):
                next_features = word_features(sentence[i + offset])
                for key, val in next_features.items():
                    features[f"+{offset}:{key}"] = val
            else:
                features[f"+{offset}:EOS"] = True
        
        # Bigram features (current + next word)
        if i < len(sentence) - 1:
            features["bigram"] = f"{sentence[i]}|{sentence[i+1]}"
        
        # Bigram features (previous + current word)
        if i > 0:
            features["bigram_prev"] = f"{sentence[i-1]}|{sentence[i]}"
        
        sentence_features.append(features)
    
    return sentence_features


def extract_features_for_dataset(
    sentences: List[List[str]], 
    context_window: int = 2
) -> List[List[Dict]]:
    """
    Extract CRF features for an entire dataset.
    
    Args:
        sentences: List of tokenized sentences
        context_window: Context window size
        
    Returns:
        List of feature sequences (one per sentence)
    """
    return [
        sentence_to_features(sent, context_window)
        for sent in sentences
    ]
