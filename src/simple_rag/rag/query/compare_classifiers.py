#!/usr/bin/env python3
"""
Compare SetFit vs LLM-based Query Classifiers

Compares the performance and characteristics of both classification approaches:
- SetFit: Fast, requires training, no LLM needed at inference
- LLM (Llama 3.2 3B): Zero-shot, no training, requires Ollama running
"""

import time
from pathlib import Path
from .llm_query_classifier import LLMQueryClassifier


def compare_classifiers():
    """
    Compare both classifiers on sample queries.
    """
    print("\n" + "=" * 80)
    print("🔬 Comparing Query Classifiers: SetFit vs LLM (Llama 3.2 3B)")
    print("=" * 80)

    # Initialize LLM classifier (always available)
    print("\n📦 Initializing LLM Classifier...")
    llm_classifier = LLMQueryClassifier(model_name="llama3.2:3b")

    # Try to initialize SetFit classifier (may not be available)
    setfit_classifier = None
    try:
        from .query_classification import QueryClassifier
        script_dir = Path(__file__).parent
        model_path = script_dir / "models" / "query_classifier"

        if model_path.exists():
            print("📦 Initializing SetFit Classifier...")
            setfit_classifier = QueryClassifier(str(model_path))
        else:
            print("⚠️  SetFit model not found. Train it first with: python query_classification.py")
    except ImportError:
        print("⚠️  SetFit not installed. Install with: poetry add setfit")

    # Test queries — updated for the 9-category taxonomy
    test_queries = [
        ("What's the weather like today?", "not_related"),
        ("How to cook pasta?", "not_related"),
        ("What is the current price of gold", "not_related"),
        ("What is the expense ratio of VTI?", "fund_basic"),
        ("Show me the ticker for VOO", "fund_basic"),
        ("Get the net assets of BND", "fund_basic"),
        ("Which provider manages VOO?", "fund_basic"),
        ("What was VTI's total return in 2022?", "fund_performance"),
        ("Show 5-year trailing returns for VOO", "fund_performance"),
        ("Average turnover across all years for BND", "fund_performance"),
        ("List the top 5 holdings in VTI by weight", "fund_portfolio"),
        ("What sectors does VOO invest in?", "fund_portfolio"),
        ("Geographic breakdown of VXUS", "fund_portfolio"),
        ("Find funds with conservative investment strategy", "fund_profile"),
        ("What funds focus on growth investing?", "fund_profile"),
        ("Show me funds with low risk profile", "fund_profile"),
        ("Show risk factors for Apple from 10-K", "company_filing"),
        ("Revenue breakdown from AAPL 10-K", "company_filing"),
        ("Who is the CEO of Apple?", "company_people"),
        ("Insider transactions for TSLA", "company_people"),
        ("Find low-risk funds managed by Vanguard", "hybrid_graph_vector"),
        ("Which funds hold Apple stock and who manages them?", "cross_entity"),
    ]

    print(f"\n📊 Testing on {len(test_queries)} sample queries")
    print("=" * 80)

    # Track metrics
    llm_correct = 0
    setfit_correct = 0
    llm_times = []
    setfit_times = []

    for i, (query, true_label) in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Query: '{query}'")
        print(f"True Label: {true_label}")

        # LLM prediction
        start = time.time()
        llm_result = llm_classifier.predict(query)
        llm_time = (time.time() - start) * 1000
        llm_times.append(llm_time)

        llm_match = "✅" if llm_result["label"] == true_label else "❌"
        if llm_result["label"] == true_label:
            llm_correct += 1

        print(f"  LLM:    {llm_match} {llm_result['label']:25} (confidence: {llm_result['confidence']:.2f}, time: {llm_time:.0f}ms)")

        # SetFit prediction (if available)
        if setfit_classifier:
            start = time.time()
            setfit_result = setfit_classifier.predict(query)
            setfit_time = (time.time() - start) * 1000
            setfit_times.append(setfit_time)

            setfit_match = "✅" if setfit_result["label"] == true_label else "❌"
            if setfit_result["label"] == true_label:
                setfit_correct += 1

            print(f"  SetFit: {setfit_match} {setfit_result['label']:25} (confidence: {setfit_result['confidence']:.2f}, time: {setfit_time:.0f}ms)")

    # Summary
    print("\n" + "=" * 80)
    print("📈 Summary")
    print("=" * 80)

    total = len(test_queries)

    print(f"\n🤖 LLM Classifier (Llama 3.2 3B):")
    print(f"  Accuracy:     {(llm_correct / total) * 100:.1f}% ({llm_correct}/{total})")
    print(f"  Avg Time:     {sum(llm_times) / len(llm_times):.0f}ms")
    print(f"  Total Time:   {sum(llm_times):.0f}ms")

    if setfit_classifier:
        print(f"\n🎯 SetFit Classifier:")
        print(f"  Accuracy:     {(setfit_correct / total) * 100:.1f}% ({setfit_correct}/{total})")
        print(f"  Avg Time:     {sum(setfit_times) / len(setfit_times):.0f}ms")
        print(f"  Total Time:   {sum(setfit_times):.0f}ms")

        speedup = sum(llm_times) / sum(setfit_times)
        print(f"\n⚡ SetFit is {speedup:.1f}x faster than LLM")

    # Recommendations
    print("\n" + "=" * 80)
    print("💡 Recommendations")
    print("=" * 80)

    print("\n🤖 Use LLM Classifier when:")
    print("  • You don't want to train a model")
    print("  • You need zero-shot classification")
    print("  • You want interpretable responses")
    print("  • Latency is not critical (100-500ms per query)")
    print("  • You already have Ollama running")

    print("\n🎯 Use SetFit Classifier when:")
    print("  • You want maximum speed (10-50ms per query)")
    print("  • You can afford one-time training")
    print("  • You want consistent, deterministic results")
    print("  • You need to classify many queries in batch")
    print("  • You want to avoid LLM dependencies at inference")

    print("\n🔄 Hybrid Approach:")
    print("  • Use SetFit for production (fast, reliable)")
    print("  • Use LLM for development/testing (no training needed)")
    print("  • Use LLM as fallback when SetFit is uncertain")


def main():
    compare_classifiers()


if __name__ == "__main__":
    main()
