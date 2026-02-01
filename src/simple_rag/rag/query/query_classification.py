#!/usr/bin/env python3
"""
Query Classification using SetFit

Trains a SetFit model to classify user queries into 4 categories:
- 'not related': Queries unrelated to fund management
- 'simple query': Direct property lookups, single-entity queries
- 'complex query': Multi-entity queries with joins, aggregations, filtering
- 'vector search': Semantic queries requiring embedding search (strategy, risk, objectives)
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


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
            self.model = SetFitModel.from_pretrained(model_path)
            print(f"✓ Loaded model from {model_path}")
        else:
            # Use a small, efficient sentence transformer as base
            self.model = SetFitModel.from_pretrained(
                "sentence-transformers/paraphrase-mpnet-base-v2",
                labels=["not related", "simple query", "complex query", "vector search"]
            )
            print("✓ Initialized base model")
    
    def predict(self, query: str) -> dict:
        """
        Classify a single query.
        
        Args:
            query: User query string
            
        Returns:
            dict with 'label' and 'confidence'
        """
        prediction = self.model.predict([query])[0]
        probs = self.model.predict_proba([query])[0]
        confidence = float(max(probs))
        
        return {
            "label": prediction,
            "confidence": confidence,
            "probabilities": {
                "not related": float(probs[0]),
                "simple query": float(probs[1]),
                "complex query": float(probs[2]),
                "vector search": float(probs[3])
            }
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
            results.append({
                "label": pred,
                "confidence": float(max(prob)),
                "probabilities": {
                    "not related": float(prob[0]),
                    "simple query": float(prob[1]),
                    "complex query": float(prob[2]),
                    "vector search": float(prob[3])
                }
            })
        return results


def load_data(data_path: str) -> tuple:
    """
    Load training and test data from JSON file.
    
    Args:
        data_path: Path to training_data.json
        
    Returns:
        (train_dataset, test_dataset)
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_dict({
        "text": [item["text"] for item in data["train"]],
        "label": [item["label"] for item in data["train"]]
    })
    
    test_dataset = Dataset.from_dict({
        "text": [item["text"] for item in data["test"]],
        "label": [item["label"] for item in data["test"]]
    })
    
    print(f"✓ Loaded {len(train_dataset)} training examples")
    print(f"✓ Loaded {len(test_dataset)} test examples")
    
    return train_dataset, test_dataset


def train_model(train_dataset: Dataset, output_dir: str = "models/query_classifier"):
    """
    Train SetFit model on the training dataset.
    
    Args:
        train_dataset: Hugging Face Dataset with 'text' and 'label' columns
        output_dir: Directory to save the trained model
        
    Returns:
        Trained SetFitModel
    """
    print("\n" + "="*60)
    print("🚀 Starting SetFit Training")
    print("="*60)
    
    # Initialize model
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        labels=["not related", "simple query", "complex query", "vector search"]
    )
    
    # Training arguments
    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,  # SetFit typically needs only 1 epoch
        body_learning_rate=2e-5,
        warmup_proportion=0.1,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    
    # Train
    print("\n📚 Training model...")
    trainer.train()
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    print(f"\n✓ Model saved to {output_path}")
    
    return model


def evaluate_model(model: SetFitModel, test_dataset: Dataset):
    """
    Evaluate the trained model on test dataset.
    
    Args:
        model: Trained SetFitModel
        test_dataset: Test dataset
    """
    print("\n" + "="*60)
    print("📊 Evaluating Model")
    print("="*60)
    
    # Get predictions
    texts = test_dataset["text"]
    true_labels = test_dataset["label"]
    predictions = model.predict(texts)
    
    # Calculate metrics
    print("\n📈 Classification Report:")
    print(classification_report(true_labels, predictions, digits=3))
    
    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    labels = ["not related", "simple query", "complex query", "vector search"]
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    
    print("\n" + " "*20 + "Predicted")
    print(" "*15 + "  ".join([f"{l[:8]:>8}" for l in labels]))
    for i, label in enumerate(labels):
        print(f"{label[:12]:>12} | " + "  ".join([f"{cm[i][j]:>8}" for j in range(len(labels))]))
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"\n✅ Overall Accuracy: {accuracy:.2%}")
    
    # Show some example predictions
    print("\n🔍 Example Predictions:")
    for i in range(min(5, len(texts))):
        probs = model.predict_proba([texts[i]])[0]
        print(f"\nQuery: '{texts[i]}'")
        print(f"True: {true_labels[i]} | Predicted: {predictions[i]}")
        print(f"Confidence: {max(probs):.2%}")


def main():
    """
    Main training pipeline.
    """
    # Paths
    script_dir = Path(__file__).parent
    data_path = script_dir / "training_data.json"
    model_dir = script_dir / "models" / "query_classifier"
    
    # Load data
    train_dataset, test_dataset = load_data(str(data_path))
    
    # Train model
    model = train_model(train_dataset, str(model_dir))
    
    # Evaluate model
    evaluate_model(model, test_dataset)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_dir}")
    print("\nTo use the model:")
    print("  from query_classification import QueryClassifier")
    print(f"  classifier = QueryClassifier('{model_dir}')")
    print("  result = classifier.predict('What is the expense ratio of VTI?')")
    print("  print(result)")


if __name__ == "__main__":
    main()