#!/usr/bin/env python3
"""
Test RAGAS evaluation with LOCAL models only (no OpenAI API needed)
"""

import sys
sys.path.append('src')

from simple_rag.evaluation.ragas import RAGEvaluator

# Path to your JSON with pre-generated answers
JSON_PATH = "src/simple_rag/evaluation/pair_answers_rag_rerank_gemini.json"

print("="*60)
print("RAGAS EVALUATION - LOCAL MODELS ONLY")
print("="*60)
print()

# Initialize evaluator with LOCAL models
# Option 1: Ollama LLM + Ollama Embeddings (fastest, all local)
print("Configuration:")
print("  - Judge LLM: llama3.1:8b (Ollama)")
print("  - Embeddings: nomic-embed-text (Ollama)")
print()

try:
    # First, let's check the JSON structure
    import json
    print("Checking JSON structure...")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            sample = data[0]
        else:
            sample = data['questions'][0] if 'questions' in data else data
        
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  contexts type: {type(sample.get('contexts'))}")
        if 'contexts' in sample and isinstance(sample['contexts'], list):
            print(f"  Number of contexts: {len(sample['contexts'])}")
            print(f"  First context type: {type(sample['contexts'][0])}")
            print(f"  First context preview: {sample['contexts'][0][:100]}...")
    
    print("\nInitializing evaluator...")
    evaluator = RAGEvaluator(
        collection_name="simple_rag",
        judge_llm_model="llama3.1:8b",
        judge_llm_provider="ollama",
        ragas_embedding_provider="ollama",  # Use Ollama embeddings
        ollama_embedding_model="nomic-embed-text"
    )
    
    print("✓ Evaluator initialized successfully")
    print(f"  Judge LLM type: {type(evaluator.judge_llm)}")
    print(f"  Embeddings type: {type(evaluator.ragas_embeddings)}")
    
    # Quick test with 3 samples first
    print("\nRunning quick test with 3 samples...")
    result = evaluator.evaluate_with_ragas_subset(
        rag_output_json_path=JSON_PATH,
        num_samples=3
    )
    
    print("\n✓ Success! RAGAS is working with local models only.")
    print("\nTo evaluate all questions, use:")
    print("  result = evaluator.evaluate_with_ragas(rag_output_json_path=JSON_PATH)")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
