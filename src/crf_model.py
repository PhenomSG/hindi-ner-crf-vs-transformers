"""
CRF model for Hindi NER.

Wraps sklearn-crfsuite for training, evaluation, and prediction.
"""

import json
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics
from typing import List, Dict, Tuple, Optional

from src.feature_extraction import extract_features_for_dataset
from src.data_utils import format_for_crf


class HindiCRFModel:
    """
    Conditional Random Field model for Hindi Named Entity Recognition.
    
    This is the 'traditional' baseline in our CRF vs mBERT comparison.
    It relies on handcrafted features (word shape, affixes, script type,
    Hindi-specific indicators) rather than learned representations.
    """
    
    def __init__(
        self,
        algorithm: str = "lbfgs",
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        context_window: int = 2,
    ):
        """
        Initialize the CRF model.
        
        Args:
            algorithm: Training algorithm ('lbfgs', 'l2sgd', 'ap', 'pa', 'arow')
            c1: L1 regularization coefficient
            c2: L2 regularization coefficient
            max_iterations: Maximum training iterations
            context_window: How many tokens before/after to use as context
        """
        self.context_window = context_window
        self.crf = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )
        self._is_trained = False
    
    def _prepare_data(
        self, sentences: List[List[str]], labels: List[List[str]]
    ) -> Tuple[List[List[Dict]], List[List[str]]]:
        """Extract features and return (X, y) for CRF."""
        X = extract_features_for_dataset(sentences, self.context_window)
        return X, labels
    
    def train(
        self, 
        train_sentences: List[List[str]], 
        train_labels: List[List[str]],
        val_sentences: Optional[List[List[str]]] = None,
        val_labels: Optional[List[List[str]]] = None,
    ) -> dict:
        """
        Train the CRF model.
        
        Args:
            train_sentences: Training tokenized sentences
            train_labels: Training BIO label sequences
            val_sentences: Optional validation sentences
            val_labels: Optional validation labels
            
        Returns:
            Dictionary with training info and optional validation metrics
        """
        print("Extracting features for training data...")
        X_train, y_train = self._prepare_data(train_sentences, train_labels)
        
        print(f"Training CRF on {len(X_train)} sentences...")
        self.crf.fit(X_train, y_train)
        self._is_trained = True
        print("✓ CRF training complete.")
        
        results = {
            "num_train_sentences": len(X_train),
            "labels": list(self.crf.classes_),
        }
        
        # Evaluate on validation set if provided
        if val_sentences is not None and val_labels is not None:
            print("Evaluating on validation set...")
            val_metrics = self.evaluate(val_sentences, val_labels)
            results["val_metrics"] = val_metrics
            print(f"  Validation F1: {val_metrics['overall_f1']:.4f}")
        
        return results
    
    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Predict NER labels for a list of sentences.
        
        Args:
            sentences: List of tokenized sentences
            
        Returns:
            List of predicted BIO label sequences
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        X = extract_features_for_dataset(sentences, self.context_window)
        return self.crf.predict(X)
    
    def evaluate(
        self, sentences: List[List[str]], labels: List[List[str]]
    ) -> dict:
        """
        Evaluate the model and return entity-level metrics.
        
        Uses seqeval for proper entity-level evaluation.
        
        Args:
            sentences: Tokenized sentences
            labels: True BIO label sequences
            
        Returns:
            Dictionary with precision, recall, F1 (overall and per-entity)
        """
        from seqeval.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
        
        predictions = self.predict(sentences)
        
        # Overall metrics
        overall_precision = precision_score(labels, predictions)
        overall_recall = recall_score(labels, predictions)
        overall_f1 = f1_score(labels, predictions)
        
        # Detailed report
        report = classification_report(labels, predictions, output_dict=True)
        
        return {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "report": report,
            "report_str": classification_report(labels, predictions),
        }
    
    def get_predictions_with_tokens(
        self, sentences: List[List[str]], labels: List[List[str]]
    ) -> List[Dict]:
        """
        Get predictions alongside tokens and true labels for error analysis.
        
        Returns:
            List of dicts with 'tokens', 'true_labels', 'pred_labels'
        """
        predictions = self.predict(sentences)
        
        results = []
        for tokens, true_labs, pred_labs in zip(sentences, labels, predictions):
            results.append({
                "tokens": tokens,
                "true_labels": true_labs,
                "pred_labels": pred_labs,
            })
        
        return results
    
    def save(self, filepath: str):
        """Save the trained CRF model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "crf": self.crf,
                "context_window": self.context_window,
                "is_trained": self._is_trained,
            }, f)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "HindiCRFModel":
        """Load a trained CRF model from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        model = cls(context_window=data["context_window"])
        model.crf = data["crf"]
        model._is_trained = data["is_trained"]
        print(f"✓ Model loaded from {filepath}")
        return model
    
    def get_top_features(self, n: int = 20) -> dict:
        """
        Get the most important features for each label.
        
        Useful for understanding what the CRF learned and for
        presentation/report.
        
        Returns:
            Dictionary mapping each label to its top-n features
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained yet.")
        
        top_features = {}
        for label in self.crf.classes_:
            if label == "O":
                continue
            
            state_features = self.crf.state_features_
            # Get features for this label, sorted by weight
            label_features = [
                (attr, weight) 
                for (attr, lab), weight in state_features.items() 
                if lab == label
            ]
            label_features.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features[label] = label_features[:n]
        
        return top_features

    def print_top_features(self, top_n: int = 5):
        """Pretty-print top learned state features for each label."""
        top_features = self.get_top_features(n=top_n)
        for label, features in sorted(top_features.items()):
            print(f"\nLabel: {label}")
            for feat, weight in features:
                print(f"  {weight:+0.3f} : {feat}")
