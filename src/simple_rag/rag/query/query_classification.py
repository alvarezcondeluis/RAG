#!/usr/bin/env python3
"""
Query Classification using SetFit

Trains a SetFit model to classify user queries into 9 categories that map
directly to the graph schema domains. Each category determines which
subgraph schema is fed to the Text2Cypher LLM.

Categories
----------
not_related          → no DB interaction at all
fund_basic           → Fund, Provider, Trust, ShareClass properties
fund_performance     → TrailingPerformance, FinancialHighlight
fund_portfolio       → Portfolio, Holding, sector/geo allocations
fund_profile         → vector search on Strategy, Risk, Objective,
                       PerformanceCommentary chunks
company_filing       → Filing10K, RiskFactor, BusinessInfo,
                       ManagementDiscussion, LegalProceeding, Financials
company_people       → Person, InsiderTransaction, CEO, CompensationPackage
hybrid_graph_vector  → needs both cypher traversal + vector search
cross_entity         → spans Fund + Company + Person in one query
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from enum import Enum
from typing import Dict, List, Optional 


# ─── Canonical label enum ────────────────────────────────────────────────────
class QueryCategory(str, Enum):
    NOT_RELATED = "not_related"
    FUND_BASIC = "fund_basic"
    FUND_PERFORMANCE = "fund_performance"
    FUND_PORTFOLIO = "fund_portfolio"
    FUND_PROFILE = "fund_profile"
    COMPANY_FILING = "company_filing"
    COMPANY_PEOPLE = "company_people"
    HYBRID_GRAPH_VECTOR = "hybrid_graph_vector"
    CROSS_ENTITY = "cross_entity"


# Ordered list used by SetFit (index ↔ label mapping)
LABELS: List[str] = [c.value for c in QueryCategory]


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
        Classify a single query.

        Args:
            query: User query string

        Returns:
            dict with 'label', 'confidence', and 'probabilities'
        """
        prediction = self.model.predict([query])[0]
        probs = self.model.predict_proba([query])[0]
        confidence = float(max(probs))

        # Sort probabilities to get top 2
        label_probs = [(label, float(probs[i])) for i, label in enumerate(LABELS)]
        label_probs.sort(key=lambda x: x[1], reverse=True)
        top_2 = label_probs[:2]

        return {
            "label": prediction,
            "confidence": confidence,
            "top_2": top_2,
            "probabilities": {label: float(probs[i]) for i, label in enumerate(LABELS)},
        }

    def predict_batch(self, queries: list) -> list:
        """
        Classify multiple queries.

        Args:
            queries: List of query strings

        Returns:
            List of prediction dicts
        """
        predictions = self.model.predict(queries)
        probs = self.model.predict_proba(queries)

        results = []
        for pred, prob in zip(predictions, probs):
            # Sort probabilities to get top 2
            label_probs = [(label, float(prob[i])) for i, label in enumerate(LABELS)]
            label_probs.sort(key=lambda x: x[1], reverse=True)
            top_2 = label_probs[:2]

            results.append(
                {
                    "label": pred,
                    "confidence": float(max(prob)),
                    "top_2": top_2,
                    "probabilities": {label: float(prob[i]) for i, label in enumerate(LABELS)},
                }
            )
        return results


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(data_path: str) -> tuple:
    """
    Load training and test data from JSON file.

    Args:
        data_path: Path to training_data.json

    Returns:
        (train_dataset, test_dataset)
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_dict(
        {
            "text": [item["text"] for item in data["train"]],
            "label": [item["label"] for item in data["train"]],
        }
    )

    test_dataset = Dataset.from_dict(
        {
            "text": [item["text"] for item in data["test"]],
            "label": [item["label"] for item in data["test"]],
        }
    )

    print(f"✓ Loaded {len(train_dataset)} training examples")
    print(f"✓ Loaded {len(test_dataset)} test examples")

    return train_dataset, test_dataset


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(
    train_dataset: Dataset, output_dir: str = "models/query_classifier"
):
    """
    Train SetFit model on the training dataset.

    Args:
        train_dataset: Hugging Face Dataset with 'text' and 'label' columns
        output_dir: Directory to save the trained model

    Returns:
        Trained SetFitModel
    """
    print("\n" + "=" * 60)
    print("🚀 Starting SetFit Training")
    print("=" * 60)

    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        labels=LABELS,
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        body_learning_rate=2e-5,
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

def evaluate_model(model: SetFitModel, test_dataset: Dataset):
    """
    Evaluate the trained model on test dataset.

    Args:
        model: Trained SetFitModel
        test_dataset: Test dataset
    """
    print("\n" + "=" * 60)
    print("📊 Evaluating Model")
    print("=" * 60)

    texts = test_dataset["text"]
    true_labels = test_dataset["label"]
    predictions = model.predict(texts)

    print("\n📈 Classification Report:")
    print(classification_report(true_labels, predictions, labels=LABELS, digits=3))

    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=LABELS)

    # Header
    short = [l[:10] for l in LABELS]
    print("\n" + " " * 22 + "Predicted")
    print(" " * 15 + "  ".join([f"{s:>10}" for s in short]))
    for i, label in enumerate(LABELS):
        row = "  ".join([f"{cm[i][j]:>10}" for j in range(len(LABELS))])
        print(f"{label[:14]:>14} | {row}")

    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"\n✅ Overall Accuracy: {accuracy:.2%}")

    # Show some example predictions
    print("\n🔍 Example Predictions:")
    for i in range(min(5, len(texts))):
        probs = model.predict_proba([texts[i]])[0]
        print(f"\nQuery: '{texts[i]}'")
        print(f"True: {true_labels[i]} | Predicted: {predictions[i]}")
        print(f"Confidence: {max(probs):.2%}")


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main():
    """Main training pipeline."""
    script_dir = Path(__file__).parent
    data_path = script_dir / "training_data.json"
    model_dir = script_dir / "models" / "query_classifier"

    train_dataset, test_dataset = load_data(str(data_path))
    model = train_model(train_dataset, str(model_dir))
    evaluate_model(model, test_dataset)

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {model_dir}")
    print("\nTo use the model:")
    print("  from query_classification import QueryClassifier")
    print(f"  classifier = QueryClassifier('{model_dir}')")
    print("  result = classifier.predict('What is the expense ratio of VTI?')")
    print("  print(result)")


if __name__ == "__main__":
    main()