#!/usr/bin/env python3
"""
Query Classification using SetFit

Trains a SetFit model to classify user queries into 6 categories that map
directly to the graph schema domains. Each category determines which
subgraph schema is fed to the Text2Cypher LLM.

Categories
----------
not_related          → no DB interaction at all
fund_basic           → Fund, Provider, Trust, ShareClass properties,
                       FinancialHighlight, TrailingPerformance, returns
fund_portfolio       → Portfolio, Holding, sector/geo allocations
fund_profile         → vector search on Strategy, Risk, Objective,
                       PerformanceCommentary chunks
company_filing       → Filing10K, RiskFactor, BusinessInfo,
                       ManagementDiscussion, LegalProceeding, Financials
company_people       → Person, InsiderTransaction, CEO, CompensationPackage
"""

import json
import argparse
import torch
from pathlib import Path
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.metrics import classification_report, multilabel_confusion_matrix, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from enum import Enum
from typing import Dict, List, Optional


# ─── Canonical label enum ────────────────────────────────────────────────────
class QueryCategory(str, Enum):
    NOT_RELATED = "not_related"
    FUND_BASIC = "fund_basic"  # includes performance, returns, financial highlights
    FUND_PORTFOLIO = "fund_portfolio"
    FUND_PROFILE = "fund_profile"
    COMPANY_FILING = "company_filing"
    COMPANY_PEOPLE = "company_people"


# Ordered list used by SetFit (index ↔ label mapping)
LABELS: List[str] = [c.value for c in QueryCategory]

# Per-label sigmoid thresholds (lower = more sensitive to that label)
# fund_portfolio is lowered because it frequently co-occurs with fund_basic and
# the model's sigmoid for it sits around 0.35-0.45 for mixed questions.
LABEL_THRESHOLDS: Dict[str, float] = {
    "not_related":    0.5,
    "fund_basic":     0.5,
    "fund_portfolio": 0.3,
    "fund_profile":   0.5,
    "company_filing": 0.5,
    "company_people": 0.5,
}


class QueryClassifier:
    """
    SetFit-based query classifier for routing queries to appropriate handlers.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the query classifier.

        Args:
            model_path: Path to saved SetFit model. If None, uses default base model.
        """
        if model_path and Path(model_path).exists():
            self.model = SetFitModel.from_pretrained(model_path, device="cpu")
            print(f"✓ Loaded model from {model_path} on CPU")
        else:
            # Use a small, efficient sentence transformer as base
            self.model = SetFitModel.from_pretrained(
                "sentence-transformers/paraphrase-mpnet-base-v2",
                labels=LABELS,
                device="cpu"
            )
            print("✓ Initialized base model on CPU")

    def predict(self, query: str) -> dict:
        """
        Classify a single query (multi-label).

        Returns:
            dict with 'labels' (active labels list), 'per_label_confidence', 'top_label'
        """
        probs = self.model.predict_proba([query])[0]  # shape: (n_labels,)
        if hasattr(probs, "numpy"):
            probs = probs.numpy()

        label_probs = {label: float(probs[i]) for i, label in enumerate(LABELS)}
        active_labels = [
            label for label, p in label_probs.items()
            if p >= LABEL_THRESHOLDS.get(label, 0.5)
        ]

        # Fallback: if nothing crosses its threshold, take the highest
        if not active_labels:
            top_label = max(label_probs, key=label_probs.get)
            active_labels = [top_label]

        # Sort by probability descending
        active_labels.sort(key=lambda l: label_probs[l], reverse=True)

        return {
            "labels": active_labels,                    # list of active labels
            "top_label": active_labels[0],              # highest confidence label
            "confidence": label_probs[active_labels[0]],
            "per_label_confidence": label_probs,        # full probability dict
        }

    def predict_batch(self, queries: list) -> list:
        """
        Classify multiple queries (multi-label).

        Returns:
            List of prediction dicts (same format as predict())
        """
        probs_batch = self.model.predict_proba(queries)  # shape: (n_queries, n_labels)
        if hasattr(probs_batch, "numpy"):
            probs_batch = probs_batch.numpy()

        results = []
        for probs in probs_batch:
            label_probs = {label: float(probs[i]) for i, label in enumerate(LABELS)}
            active_labels = [
                label for label, p in label_probs.items()
                if p >= LABEL_THRESHOLDS.get(label, 0.5)
            ]
            if not active_labels:
                active_labels = [max(label_probs, key=label_probs.get)]
            active_labels.sort(key=lambda l: label_probs[l], reverse=True)

            results.append({
                "labels": active_labels,
                "top_label": active_labels[0],
                "confidence": label_probs[active_labels[0]],
                "per_label_confidence": label_probs,
            })
        return results



# ─── Data loading ─────────────────────────────────────────────────────────────

def _get_labels(item: dict) -> List[str]:
    """Resolve 'labels' (list) or legacy 'label' (str) field."""
    if "labels" in item:
        return item["labels"]
    if "label" in item:
        return [item["label"]]
    return []


def load_data(data_path: str) -> tuple:
    """
    Load training and test data from training_data.json (train/test split dict).

    Returns:
        (train_dataset, test_dataset, mlb)
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    mlb = MultiLabelBinarizer(classes=LABELS)
    mlb.fit([LABELS])

    def make_dataset(items):
        texts = [item["text"] for item in items]
        labels_list = [_get_labels(item) for item in items]
        label_matrix = mlb.transform(labels_list).tolist()
        return Dataset.from_dict({"text": texts, "label": label_matrix})

    train_dataset = make_dataset(data["train"])
    test_dataset  = make_dataset(data["test"])

    print(f"✓ Loaded {len(train_dataset)} training examples")
    print(f"✓ Loaded {len(test_dataset)} test examples (from training_data.json)")

    return train_dataset, test_dataset, mlb


def load_standalone_test_set(test_path: str, mlb: MultiLabelBinarizer) -> Dataset:
    """
    Load a standalone flat JSON list (classification_test_set.json).

    Each entry must have 'text' and either 'labels' (list) or 'label' (str).

    Returns:
        Dataset with 'text' and binary 'label' columns.
    """
    with open(test_path, "r") as f:
        items = json.load(f)

    texts = [item["text"] for item in items]
    labels_list = [_get_labels(item) for item in items]
    label_matrix = mlb.transform(labels_list).tolist()
    dataset = Dataset.from_dict({"text": texts, "label": label_matrix})

    print(f"✓ Loaded {len(dataset)} test examples (from {Path(test_path).name})")
    return dataset


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(
    train_dataset: Dataset,
    output_dir: str = "models/query_classifier",
    num_epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5
) -> SetFitModel:
    """
    Train a multi-label SetFit model.

    The model uses one-vs-rest strategy: one binary classifier per label,
    each outputting an independent sigmoid probability.
    """
    print("\n" + "=" * 60)
    print("🚀 Starting SetFit Multi-Label Training")
    print("=" * 60)
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Labels ({len(LABELS)}): {LABELS}")

    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        labels=LABELS,
        multi_target_strategy="one-vs-rest",
    )

    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        body_learning_rate=learning_rate,
        warmup_proportion=0.1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    print("\n📚 Training model...")
    trainer.train()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    print(f"\n✓ Model saved to {output_path}")

    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model: SetFitModel, test_dataset: Dataset, mlb: MultiLabelBinarizer):
    """
    Evaluate the multi-label trained model on test dataset.
    """
    print("\n" + "=" * 60)
    print("📊 Evaluating Multi-Label Model")
    print("=" * 60)

    texts = test_dataset["text"]
    true_binary = np.array(test_dataset["label"])  # shape: (n, n_labels)

    # Convert Tensor → numpy if needed
    probs = model.predict_proba(texts)
    if hasattr(probs, "numpy"):
        probs = probs.numpy()

    # Apply per-label thresholds
    threshold_vec = np.array([LABEL_THRESHOLDS.get(l, 0.5) for l in LABELS])
    pred_binary = (probs >= threshold_vec).astype(int)

    # Hamming loss (fraction of labels wrong per sample)
    hl = hamming_loss(true_binary, pred_binary)
    print(f"\n📉 Hamming Loss: {hl:.4f} (lower is better, 0 = perfect)")

    # Per-label report
    print("\n📈 Per-Label Classification Report:")
    print(classification_report(
        true_binary, pred_binary,
        target_names=LABELS, digits=3, zero_division=0
    ))

    # Exact match accuracy
    exact = np.mean(np.all(true_binary == pred_binary, axis=1))
    print(f"✅ Exact Match Accuracy: {exact:.2%}")

    # Show failures only (up to 10)
    print("\n🔍 Misclassified examples:")
    shown = 0
    for i in range(len(texts)):
        if not np.array_equal(true_binary[i], pred_binary[i]):
            true_lbls = mlb.inverse_transform(true_binary[i:i+1])[0]
            pred_lbls = mlb.inverse_transform(pred_binary[i:i+1])[0]
            print(f"\nQuery:     '{texts[i]}'")
            print(f"Expected:  {list(true_lbls)}")
            print(f"Predicted: {list(pred_lbls)}")
            shown += 1
            if shown >= 10:
                break
    if shown == 0:
        print("  ✨ All examples correct!")


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main():
    """Main multi-label training pipeline."""
    parser = argparse.ArgumentParser(description="SetFit multi-label query classifier")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the already-saved model.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to a standalone test set JSON (flat list). Defaults to training_data.json test split.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_path = script_dir / "training_data.json"
    model_dir = script_dir / "models" / "query_classifier"

    train_dataset, test_dataset, mlb = load_data(str(data_path))

    # Override test set if --test-file provided
    if args.test_file:
        test_dataset = load_standalone_test_set(args.test_file, mlb)
    if args.eval_only:
        if not model_dir.exists():
            print(f"❌ No saved model found at {model_dir}. Run without --eval-only first.")
            return
        print(f"\n⏩ Skipping training — loading saved model from {model_dir}")
        model = SetFitModel.from_pretrained(str(model_dir), device="cpu")
    else:
        model = train_model(train_dataset, str(model_dir))

    evaluate_model(model, test_dataset, mlb)

    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)
    print(f"\nModel at: {model_dir}")


if __name__ == "__main__":
    main()