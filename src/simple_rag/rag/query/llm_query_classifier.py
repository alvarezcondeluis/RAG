#!/usr/bin/env python3
"""
LLM-based Query Classification using Ollama Llama 3.2 3B

Alternative to SetFit that uses a lightweight LLM for zero-shot classification.
No training required - uses prompt engineering with Ollama's Llama 3.2 3B model.

Categories match the SetFit classifier (see query_classification.py):
    not_related, fund_basic, fund_performance, fund_portfolio, fund_profile,
    company_filing, company_people, hybrid_graph_vector, cross_entity
"""

import json
import re
from typing import Dict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .query_classification import LABELS, QueryCategory


class LLMQueryClassifier:
    """
    LLM-based query classifier using Ollama Llama 3.2 3B.
    Uses prompt engineering for zero-shot classification.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        api_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        """
        Initialize the LLM-based query classifier.

        Args:
            model_name: Ollama model name (default: llama3.2:3b)
            api_url: Ollama API URL
            temperature: Sampling temperature (0 = deterministic)
        """
        self.model_name = model_name
        self.labels = LABELS

        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=model_name,
            base_url=api_url,
            temperature=temperature,
            num_predict=50,
        )

        # Classification prompt template
        self.prompt_template = """You are a query classification expert for a financial database that stores mutual-fund and company data in a graph.

Classify the following user query into EXACTLY ONE of these categories:

1. "not_related"          – Queries unrelated to fund management, finance, or investments.
   Examples: "What's the weather?", "How to cook pasta?", "Tell me a joke"

2. "fund_basic"           – Direct property lookups on Fund, Provider, Trust, or ShareClass nodes.
   Examples: "What is the expense ratio of VTI?", "Which provider manages VOO?", "List all share classes for Vanguard"

3. "fund_performance"     – Queries about TrailingPerformance or FinancialHighlight nodes (returns, NAV, turnover by year).
   Examples: "What was VTI's total return in 2022?", "Show 5-year trailing returns for VOO", "Average turnover across all years for BND"

4. "fund_portfolio"       – Queries about Portfolio, Holding, sector allocations, or geographic allocations.
   Examples: "Top 10 holdings of VTI", "What sectors does VOO invest in?", "Geographic breakdown of VXUS"

5. "fund_profile"         – Semantic / conceptual queries requiring vector search on Strategy, Risk, Objective, or PerformanceCommentary chunks.
   Examples: "Find funds with conservative strategy", "Which funds focus on growth?", "Funds with low-risk approach"

6. "company_filing"       – Queries about Filing10K sections: RiskFactor, BusinessInfo, ManagementDiscussion, LegalProceeding, or Financials.
   Examples: "Show risk factors for Apple", "Revenue breakdown from AAPL 10-K", "Legal proceedings for MSFT"

7. "company_people"       – Queries about Person, InsiderTransaction, CEO, or CompensationPackage.
   Examples: "Who is the CEO of Apple?", "Insider transactions for TSLA", "CEO compensation at Microsoft"

8. "hybrid_graph_vector"  – Queries that need BOTH a Cypher graph traversal AND a vector similarity search.
   Examples: "Find low-risk funds managed by Vanguard", "Show conservative strategy funds with expense ratio below 0.1%"

9. "cross_entity"         – Queries that span Fund + Company + Person entities in a single question.
   Examples: "Which funds hold Apple stock and who manages them?", "Show Vanguard funds that invest in companies with recent insider sales"

User Query: "{query}"

Classification (respond with ONLY the category name, nothing else):"""

        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=self.prompt_template,
        )

        # Create chain
        self.chain = self.prompt | self.llm | StrOutputParser()

        print(f"✓ LLM Query Classifier initialized with {model_name}")

    def predict(self, query: str) -> Dict[str, any]:
        """
        Classify a single query.

        Args:
            query: User query string

        Returns:
            dict with 'label' and 'confidence'
        """
        try:
            response = self.chain.invoke({"query": query}).strip().lower()
            label = self._extract_label(response)

            # Since LLM doesn't provide probabilities, we use a simple heuristic
            confidence = 0.95 if label in response else 0.75

            return {
                "label": label,
                "confidence": confidence,
                "raw_response": response,
            }

        except Exception as e:
            print(f"Error classifying query: {e}")
            return {
                "label": QueryCategory.FUND_BASIC.value,  # Default fallback
                "confidence": 0.5,
                "error": str(e),
            }

    def predict_batch(self, queries: List[str]) -> List[Dict[str, any]]:
        """
        Classify multiple queries.

        Args:
            queries: List of query strings

        Returns:
            List of prediction dicts
        """
        results = []
        for query in queries:
            results.append(self.predict(query))
        return results

    def _extract_label(self, response: str) -> str:
        """
        Extract the classification label from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Extracted label (one of the 9 categories)
        """
        response_lower = response.lower().strip()

        # Direct match
        for label in self.labels:
            if label == response_lower:
                return label

        # Fuzzy match — check if label substring is in the response
        for label in self.labels:
            if label in response_lower:
                return label

        # Pattern matching for common LLM paraphrases
        _FALLBACK_MAP = {
            "not_related": ["unrelated", "irrelevant", "off-topic", "not related"],
            "fund_basic": ["simple", "basic", "lookup", "fund property"],
            "fund_performance": ["performance", "return", "trailing", "financial highlight"],
            "fund_portfolio": ["portfolio", "holding", "sector", "geographic", "allocation"],
            "fund_profile": ["vector", "semantic", "embedding", "strategy", "risk profile", "conceptual"],
            "company_filing": ["filing", "10-k", "10k", "risk factor", "legal"],
            "company_people": ["ceo", "insider", "compensation", "executive"],
            "hybrid_graph_vector": ["hybrid", "graph and vector", "both"],
            "cross_entity": ["cross", "span", "fund and company", "multiple entities"],
        }

        for label, keywords in _FALLBACK_MAP.items():
            if any(kw in response_lower for kw in keywords):
                return label

        # Ultimate default
        print(
            f"Warning: Could not extract label from response: '{response}'. "
            f"Defaulting to '{QueryCategory.FUND_BASIC.value}'"
        )
        return QueryCategory.FUND_BASIC.value


# ─── Evaluation helper ──────────────────────────────────────────────────────

def evaluate_on_test_set(classifier: LLMQueryClassifier, test_data_path: str):
    """
    Evaluate the LLM classifier on the test set.

    Args:
        classifier: LLMQueryClassifier instance
        test_data_path: Path to training_data.json
    """
    print("\n" + "=" * 60)
    print("📊 Evaluating LLM Query Classifier")
    print("=" * 60)

    with open(test_data_path, "r") as f:
        data = json.load(f)

    test_examples = data.get("test", [])

    if not test_examples:
        print("No test data found!")
        return

    print(f"\n✓ Loaded {len(test_examples)} test examples")

    correct = 0
    total = len(test_examples)
    results_by_label = {label: {"correct": 0, "total": 0} for label in classifier.labels}

    print("\n🔍 Running predictions...")
    for i, example in enumerate(test_examples, 1):
        query = example["text"]
        true_label = example["label"]

        result = classifier.predict(query)
        predicted_label = result["label"]

        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
            results_by_label[true_label]["correct"] += 1
        results_by_label[true_label]["total"] += 1

        if i % 10 == 0:
            print(f"  Processed {i}/{total} examples...")

    accuracy = (correct / total) * 100

    print("\n" + "=" * 60)
    print("📈 Results")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")

    print("\nPer-Class Accuracy:")
    for label in classifier.labels:
        stats = results_by_label[label]
        if stats["total"] > 0:
            class_acc = (stats["correct"] / stats["total"]) * 100
            print(f"  {label:25} {class_acc:6.2f}% ({stats['correct']}/{stats['total']})")

    print("\n🔍 Example Predictions:")
    for i in range(min(5, len(test_examples))):
        example = test_examples[i]
        result = classifier.predict(example["text"])

        status = "✅" if result["label"] == example["label"] else "❌"
        print(f"\n{status} Query: '{example['text']}'")
        print(
            f"   True: {example['label']} | Predicted: {result['label']} "
            f"(confidence: {result['confidence']:.2f})"
        )


def main():
    """Main evaluation script."""
    from pathlib import Path

    print("\n" + "=" * 60)
    print("🚀 LLM Query Classifier - Ollama Llama 3.2 3B")
    print("=" * 60)

    classifier = LLMQueryClassifier(model_name="llama3.2:3b", temperature=0.0)

    print("\n📝 Testing on sample queries:")

    test_queries = [
        "What's the weather like today?",
        "What is the expense ratio of VTI?",
        "What was VTI's total return in 2022?",
        "Top 10 holdings of VOO",
        "Find funds with conservative investment strategy",
        "Show risk factors from Apple's 10-K",
        "Who is the CEO of Microsoft?",
        "Find low-risk funds managed by Vanguard with expense ratio below 0.1%",
        "Which funds hold Apple stock and who manages them?",
    ]

    for query in test_queries:
        result = classifier.predict(query)
        print(f"\nQuery: '{query}'")
        print(f"  → {result['label']} (confidence: {result['confidence']:.2f})")

    # Evaluate on full test set if available
    script_dir = Path(__file__).parent
    test_data_path = script_dir / "training_data.json"

    if test_data_path.exists():
        evaluate_on_test_set(classifier, str(test_data_path))
    else:
        print(f"\n⚠️  Test data not found at {test_data_path}")
        print("   Skipping full evaluation.")

    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
