"""
Transformer-based NER for Hindi using mBERT.

Fine-tunes `bert-base-multilingual-cased` for token classification
on Hindi NER data.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_utils import (
    LABEL_LIST,
    LABEL_TO_ID,
    ID_TO_LABEL,
    tokenize_and_align_labels,
)


class HindiTransformerNER:
    """
    mBERT-based NER model for Hindi.
    
    This is the 'Transformer-based' approach in our CRF vs mBERT comparison.
    It uses pretrained multilingual BERT representations fine-tuned on
    Hindi NER data, requiring NO handcrafted features.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        seed: int = 42,
    ):
        """
        Initialize the Transformer NER model.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length for tokenization
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.max_length = max_length
        self.seed = seed
        
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading model: {model_name}")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )
        
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )
        
        self._is_trained = False
        print(f"✓ Model loaded with {len(LABEL_LIST)} labels: {LABEL_LIST}")
    
    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Apply subword tokenization with label alignment to a dataset."""
        return dataset.map(
            lambda examples: tokenize_and_align_labels(
                examples, self.tokenizer, self.max_length
            ),
            batched=True,
            remove_columns=dataset.column_names,
        )
    
    def _compute_metrics(self, eval_preds):
        """
        Compute entity-level metrics for the Trainer.
        
        This function is called during training to evaluate on the
        validation set after each epoch.
        """
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)
        
        # Convert numeric predictions/labels to string labels
        # Only include non-padding tokens (label != -100)
        true_labels = []
        pred_labels = []
        
        for prediction, label in zip(predictions, labels):
            true_seq = []
            pred_seq = []
            for p, l in zip(prediction, label):
                if l != -100:
                    true_seq.append(ID_TO_LABEL[l])
                    pred_seq.append(ID_TO_LABEL[p])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq)
        
        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str = "models/mbert",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 10,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        fp16: bool = True,
    ) -> dict:
        """
        Fine-tune mBERT on Hindi NER data.
        
        Args:
            train_dataset: HuggingFace dataset (train split)
            val_dataset: HuggingFace dataset (validation split)
            output_dir: Where to save model checkpoints
            learning_rate: Learning rate for AdamW
            batch_size: Batch size for training and evaluation
            num_epochs: Maximum number of training epochs
            weight_decay: Weight decay for regularization
            warmup_ratio: Fraction of steps for learning rate warmup
            fp16: Whether to use mixed precision (requires GPU)
            
        Returns:
            Dictionary with training results
        """
        # Tokenize datasets
        print("Tokenizing training data...")
        tokenized_train = self._tokenize_dataset(train_dataset)
        print("Tokenizing validation data...")
        tokenized_val = self._tokenize_dataset(val_dataset)
        
        # Check if GPU is available
        use_fp16 = fp16 and torch.cuda.is_available()
        device_msg = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"Training on: {device_msg}")
        if fp16 and not torch.cuda.is_available():
            print("  (fp16 disabled — no GPU detected)")
        
        # Training arguments
        args_dict = {
            "output_dir": output_dir,
            "save_strategy": "epoch",
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size * 2,
            "num_train_epochs": num_epochs,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "fp16": use_fp16,
            "logging_steps": 100,
            "seed": self.seed,
            "report_to": "none",
        }
        try:
            training_args = TrainingArguments(eval_strategy="epoch", **args_dict)
        except TypeError:
            training_args = TrainingArguments(evaluation_strategy="epoch", **args_dict)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        print(f"\nStarting fine-tuning for up to {num_epochs} epochs...")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size:    {batch_size}")
        print(f"  Warmup ratio:  {warmup_ratio}")
        print(f"  FP16:          {use_fp16}")
        print()
        
        train_result = trainer.train()
        self._is_trained = True
        self.trainer = trainer
        
        # Save the best model
        trainer.save_model(os.path.join(output_dir, "best_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
        print(f"\n✓ Best model saved to {output_dir}/best_model")
        
        # Get training history
        log_history = trainer.state.log_history
        
        return {
            "train_result": train_result,
            "log_history": log_history,
        }
    
    def evaluate(self, test_dataset: Dataset) -> dict:
        """
        Evaluate the model on a test dataset.
        
        Returns:
            Dictionary with precision, recall, F1 (overall and per-entity)
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        print("Tokenizing test data...")
        tokenized_test = self._tokenize_dataset(test_dataset)
        
        # Get predictions
        predictions_output = self.trainer.predict(tokenized_test)
        predictions = np.argmax(predictions_output.predictions, axis=2)
        labels = predictions_output.label_ids
        
        # Convert to string labels
        true_labels, pred_labels = self._decode_predictions(predictions, labels)
        
        # Compute metrics
        overall_precision = precision_score(true_labels, pred_labels)
        overall_recall = recall_score(true_labels, pred_labels)
        overall_f1 = f1_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        return {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "report": report,
            "report_str": classification_report(true_labels, pred_labels),
        }
    
    def get_predictions_with_tokens(self, test_dataset: Dataset) -> List[Dict]:
        """
        Get predictions alongside original tokens and true labels.
        
        This reconstructs the word-level predictions from subword-level
        predictions, which is essential for error analysis.
        
        Returns:
            List of dicts with 'tokens', 'true_labels', 'pred_labels'
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        print("Tokenizing test data...")
        tokenized_test = self._tokenize_dataset(test_dataset)
        
        # Get predictions
        predictions_output = self.trainer.predict(tokenized_test)
        predictions = np.argmax(predictions_output.predictions, axis=2)
        labels = predictions_output.label_ids
        
        results = []
        for i in range(len(test_dataset)):
            tokens = test_dataset[i]["tokens"]
            true_tags = test_dataset[i]["ner_tags"]
            true_labels = [ID_TO_LABEL[t] for t in true_tags]
            
            # Get predicted labels for first subword of each word
            pred_seq = predictions[i]
            label_seq = labels[i]
            pred_labels_for_sentence = []
            
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    pred_labels_for_sentence.append(ID_TO_LABEL[p])
            
            # Truncate to match original token count
            pred_labels_for_sentence = pred_labels_for_sentence[:len(tokens)]
            
            # Pad if needed (shouldn't happen, but safety)
            while len(pred_labels_for_sentence) < len(tokens):
                pred_labels_for_sentence.append("O")
            
            results.append({
                "tokens": tokens,
                "true_labels": true_labels,
                "pred_labels": pred_labels_for_sentence,
            })
        
        return results
    
    def predict_sentence(self, sentence: str) -> List[Dict[str, str]]:
        """
        Predict NER tags for a raw Hindi sentence (for demo purposes).
        
        Args:
            sentence: Raw Hindi text string
            
        Returns:
            List of dicts with 'token' and 'label'
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained yet.")
        
        # Tokenize
        tokens = sentence.split()
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)[0]
        word_ids = inputs.get("word_ids", None)
        
        # If word_ids not available from inputs, reconstruct
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = tokenized.word_ids()
        
        # Map predictions back to words
        results = []
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                pred_label = ID_TO_LABEL[predictions[idx].item()]
                results.append({
                    "token": tokens[word_idx],
                    "label": pred_label,
                })
            previous_word_idx = word_idx
        
        return results
    
    def _decode_predictions(self, predictions, labels):
        """Convert numeric predictions and labels to string labels."""
        true_labels = []
        pred_labels = []
        
        for prediction, label in zip(predictions, labels):
            true_seq = []
            pred_seq = []
            for p, l in zip(prediction, label):
                if l != -100:
                    true_seq.append(ID_TO_LABEL[l])
                    pred_seq.append(ID_TO_LABEL[p])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq)
        
        return true_labels, pred_labels
    
    @classmethod
    def load_trained(cls, model_path: str, max_length: int = 128) -> "HindiTransformerNER":
        """
        Load a previously saved fine-tuned model.
        
        Args:
            model_path: Path to the saved model directory
            max_length: Maximum sequence length
            
        Returns:
            HindiTransformerNER instance with loaded weights
        """
        instance = cls.__new__(cls)
        instance.max_length = max_length
        instance.seed = 42
        instance.model_name = model_path
        
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.model = AutoModelForTokenClassification.from_pretrained(model_path)
        instance.data_collator = DataCollatorForTokenClassification(
            tokenizer=instance.tokenizer
        )
        instance._is_trained = True
        instance.trainer = None  # No trainer for loaded models
        
        print(f"✓ Model loaded from {model_path}")
        return instance
