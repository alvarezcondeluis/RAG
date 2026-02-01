#!/usr/bin/env python3
"""
LLM-based Query Classification using Ollama Llama 3.2 3B

Alternative to SetFit that uses a lightweight LLM for zero-shot classification.
No training required - uses prompt engineering with Ollama's Llama 3.2 3B model.
"""

import json
import re
from typing import Dict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LLMQueryClassifier:
    """
    LLM-based query classifier using Ollama Llama 3.2 3B.
    Uses prompt engineering for zero-shot classification.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        api_url: str = "http://localhost:11434",
        temperature: float = 0.0
    ):
        """
        Initialize the LLM-based query classifier.
        
        Args:
            model_name: Ollama model name (default: llama3.2:3b)
            api_url: Ollama API URL
            temperature: Sampling temperature (0 = deterministic)
        """
        self.model_name = model_name
        self.labels = ["not related", "simple query", "complex query", "vector search"]
        
        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=model_name,
            base_url=api_url,
            temperature=temperature,
            num_predict=50  # Short output for classification
        )
        
        # Classification prompt template
        self.prompt_template = """You are a query classification expert for a fund management database system.

Classify the following user query into ONE of these categories:

1. "not related" - Queries unrelated to fund management, finance, or investments
   Examples: "What's the weather?", "How to cook pasta?", "Tell me a joke"

2. "simple query" - Direct property lookups or single-entity queries
   Examples: "What is the expense ratio of VTI?", "Show me the ticker for VOO", "Get net assets of BND"

3. "complex query" - Multi-entity queries with joins, aggregations, filtering, or sorting
   Examples: "Show all funds with expense ratio < 0.1%", "Top 5 holdings in VTI", "List funds managed by Vanguard"

4. "vector search" - Semantic/conceptual queries requiring embedding search (strategy, risk, objectives, performance commentary)
   Examples: "Find funds with conservative strategy", "What funds focus on growth?", "Show funds with low risk"

User Query: "{query}"

Classification (respond with ONLY the category name, nothing else):"""
        
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=self.prompt_template
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
            # Get prediction from LLM
            response = self.chain.invoke({"query": query}).strip().lower()
            
            # Clean and extract label
            label = self._extract_label(response)
            
            # Since LLM doesn't provide probabilities, we use a simple heuristic
            # If the response matches exactly, high confidence; otherwise lower
            confidence = 0.95 if label in response else 0.75
            
            return {
                "label": label,
                "confidence": confidence,
                "raw_response": response
            }
        
        except Exception as e:
            print(f"Error classifying query: {e}")
            return {
                "label": "simple query",  # Default fallback
                "confidence": 0.5,
                "error": str(e)
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
            Extracted label (one of the 4 categories)
        """
        response_lower = response.lower().strip()
        
        # Direct match
        for label in self.labels:
            if label == response_lower:
                return label
        
        # Fuzzy match - check if label is in response
        for label in self.labels:
            if label in response_lower:
                return label
        
        # Pattern matching for common variations
        if any(word in response_lower for word in ["unrelated", "irrelevant", "off-topic"]):
            return "not related"
        elif any(word in response_lower for word in ["simple", "basic", "direct", "lookup"]):
            return "simple query"
        elif any(word in response_lower for word in ["complex", "advanced", "multi", "aggregate"]):
            return "complex query"
        elif any(word in response_lower for word in ["vector", "semantic", "embedding", "conceptual"]):
            return "vector search"
        
        # Default fallback
        print(f"Warning: Could not extract label from response: '{response}'. Defaulting to 'simple query'")
        return "simple query"


def evaluate_on_test_set(classifier: LLMQueryClassifier, test_data_path: str):
    """
    Evaluate the LLM classifier on the test set.
    
    Args:
        classifier: LLMQueryClassifier instance
        test_data_path: Path to training_data.json
    """
    print("\n" + "="*60)
    print("📊 Evaluating LLM Query Classifier")
    print("="*60)
    
    # Load test data
    with open(test_data_path, 'r') as f:
        data = json.load(f)
    
    test_examples = data.get("test", [])
    
    if not test_examples:
        print("No test data found!")
        return
    
    print(f"\n✓ Loaded {len(test_examples)} test examples")
    
    # Evaluate
    correct = 0
    total = len(test_examples)
    results_by_label = {label: {"correct": 0, "total": 0} for label in classifier.labels}
    
    print("\n🔍 Running predictions...")
    for i, example in enumerate(test_examples, 1):
        query = example["text"]
        true_label = example["label"]
        
        # Predict
        result = classifier.predict(query)
        predicted_label = result["label"]
        
        # Track results
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
            results_by_label[true_label]["correct"] += 1
        results_by_label[true_label]["total"] += 1
        
        # Show progress
        if i % 10 == 0:
            print(f"  Processed {i}/{total} examples...")
    
    # Calculate metrics
    accuracy = (correct / total) * 100
    
    print("\n" + "="*60)
    print("📈 Results")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print("\nPer-Class Accuracy:")
    for label in classifier.labels:
        stats = results_by_label[label]
        if stats["total"] > 0:
            class_acc = (stats["correct"] / stats["total"]) * 100
            print(f"  {label:15} {class_acc:6.2f}% ({stats['correct']}/{stats['total']})")
    
    # Show some example predictions
    print("\n🔍 Example Predictions:")
    for i in range(min(5, len(test_examples))):
        example = test_examples[i]
        result = classifier.predict(example["text"])
        
        status = "✅" if result["label"] == example["label"] else "❌"
        print(f"\n{status} Query: '{example['text']}'")
        print(f"   True: {example['label']} | Predicted: {result['label']} (confidence: {result['confidence']:.2f})")


def main():
    """
    Main evaluation script.
    """
    from pathlib import Path
    
    print("\n" + "="*60)
    print("🚀 LLM Query Classifier - Ollama Llama 3.2 3B")
    print("="*60)
    
    # Initialize classifier
    classifier = LLMQueryClassifier(
        model_name="llama3.2:3b",
        temperature=0.0
    )
    
    # Test on a few examples
    print("\n📝 Testing on sample queries:")
    
    test_queries = [
        "What's the weather like today?",
        "What is the expense ratio of VTI?",
        "Show me all funds with expense ratio less than 0.1%",
        "Find funds with conservative investment strategy"
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
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
