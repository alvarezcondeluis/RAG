#!/usr/bin/env python3
"""
Query Classification Runner

Loads the trained SetFit model and classifies queries interactively
or evaluates against the held-out test set.

Usage:
    # Interactive mode (type queries, get classifications)
    uv run python src/simple_rag/rag/query/classify.py

    # Evaluate on test set
    uv run python src/simple_rag/rag/query/classify.py --test

    # Classify a single query from CLI
    uv run python src/simple_rag/rag/query/classify.py --query "What is the expense ratio of VTI?"
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

from query_classification import QueryClassifier, LABELS


# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / "models" / "query_classifier"
TEST_SET_PATH = SCRIPT_DIR / "classification_test_set.json"


def load_test_set() -> list:
    """Load the classification test set."""
    with open(TEST_SET_PATH, "r") as f:
        return json.load(f)


def classify_single(classifier: QueryClassifier, query: str):
    """Classify and pretty-print a single query."""
    result = classifier.predict(query)
    label = result["label"]
    conf = result["confidence"]
    probs = result["probabilities"]

    print(f"\n  Query:      {query}")
    print(f"  Category:   {label}")
    print(f"  Confidence: {conf:.2%}")
    print(f"  ─── probabilities ───")
    for lbl in LABELS:
        bar = "█" * int(probs[lbl] * 40)
        print(f"    {lbl:25} {probs[lbl]:6.2%}  {bar}")


def run_test_set(classifier: QueryClassifier):
    """Evaluate on the held-out classification test set."""
    test_data = load_test_set()

    print(f"\n{'=' * 70}")
    print(f"  Evaluating on {len(test_data)} test queries")
    print(f"{'=' * 70}\n")

    correct = 0
    total = len(test_data)
    per_class = {l: {"tp": 0, "total": 0, "fp": 0} for l in LABELS}
    errors = []

    for item in test_data:
        query = item["text"]
        expected = item["label"]
        result = classifier.predict(query)
        predicted = result["label"]

        per_class[expected]["total"] += 1

        if predicted == expected:
            correct += 1
            per_class[expected]["tp"] += 1
        else:
            per_class[predicted]["fp"] += 1
            errors.append({
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "confidence": result["confidence"],
            })

    # Summary
    accuracy = correct / total if total else 0
    print(f"  Overall Accuracy: {accuracy:.2%}  ({correct}/{total})\n")

    # Per-class table
    print(f"  {'Category':25} {'Prec':>6} {'Recall':>6} {'Support':>7}")
    print(f"  {'─' * 50}")
    for lbl in LABELS:
        tp = per_class[lbl]["tp"]
        tot = per_class[lbl]["total"]
        fp = per_class[lbl]["fp"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / tot if tot > 0 else 0
        print(f"  {lbl:25} {precision:6.1%} {recall:6.1%} {tot:7}")

    # Show errors
    if errors:
        print(f"\n  ❌ Misclassified ({len(errors)}):")
        for err in errors:
            print(f"    • \"{err['query']}\"")
            print(f"      expected={err['expected']}  predicted={err['predicted']}  conf={err['confidence']:.2%}")
    else:
        print(f"\n  ✅ No misclassifications!")


def run_interactive(classifier: QueryClassifier):
    """Interactive REPL: type a query, see the classification."""
    print(f"\n{'=' * 70}")
    print("  Query Classifier — Interactive Mode")
    print("  Type a query and press Enter. Type 'quit' to exit.")
    print(f"{'=' * 70}")

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        classify_single(classifier, query)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classify queries using the trained SetFit model.")
    parser.add_argument("--test", action="store_true", help="Evaluate on the classification test set")
    parser.add_argument("--query", type=str, default=None, help="Classify a single query")
    args = parser.parse_args()

    # Load model
    if not MODEL_DIR.exists():
        print(f"❌ Trained model not found at {MODEL_DIR}")
        print("   Run query_classification.py first to train the model.")
        sys.exit(1)

    print(f"Loading model from {MODEL_DIR} ...")
    classifier = QueryClassifier(str(MODEL_DIR))

    if args.query:
        classify_single(classifier, args.query)
    elif args.test:
        run_test_set(classifier)
    else:
        run_interactive(classifier)


if __name__ == "__main__":
    main()
