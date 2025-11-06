# RAGAS Evaluation - Quick Start Guide

## Problem Fixed ✅

**Issue 1**: OpenAI API error - RAGAS was trying to use OpenAI instead of local models
**Solution**: Added `LangchainLLMWrapper` and `LangchainEmbeddingsWrapper` to properly configure RAGAS for local models

**Issue 2**: `NoneType has no attribute 'agenerate_prompt'` error
**Solution**: Removed duplicate `self.judge_llm = None` line that was overwriting the initialized judge LLM

## Quick Start - Local Models Only

### 1. Basic Usage (Ollama LLM + Ollama Embeddings)

```python
from simple_rag.evaluation.ragas import RAGEvaluator

# Initialize with LOCAL models only
evaluator = RAGEvaluator(
    judge_llm_model="llama3.1:8b",
    judge_llm_provider="ollama",
    ragas_embedding_provider="ollama",
    ollama_embedding_model="nomic-embed-text"
)

# Evaluate pre-generated answers (NO OpenAI needed)
result = evaluator.evaluate_with_ragas(
    rag_output_json_path="pair_answers_rag_rerank_gemini.json"
)

# Results
print(f"Faithfulness: {result['faithfulness']:.4f}")
print(f"Answer Correctness: {result['answer_correctness']:.4f}")
print(f"Context Recall: {result['context_recall']:.4f}")
print(f"Context Precision: {result['context_precision']:.4f}")
```

### 2. Quick Test with Subset

```python
# Test with first 10 questions
result = evaluator.evaluate_with_ragas_subset(
    rag_output_json_path="pair_answers_rag_rerank_gemini.json",
    num_samples=10
)
```

### 3. Alternative: HuggingFace Embeddings

```python
evaluator = RAGEvaluator(
    judge_llm_model="llama3.1:8b",
    judge_llm_provider="ollama",
    ragas_embedding_provider="huggingface",  # Use HuggingFace
    embedding_model="BAAI/bge-base-en-v1.5"
)
```

## Test Script

Run the test script to verify everything works:

```bash
cd /home/alvar/CascadeProjects/windsurf-project/RAG
python test_ragas_local.py
```

## Configuration Options

### Judge LLM Options
- **Ollama** (local): `judge_llm_provider="ollama"`, `judge_llm_model="llama3.1:8b"`
- **Gemini** (API): `judge_llm_provider="gemini"`, `judge_llm_model="gemini-1.5-flash"`

### Embedding Options
- **Ollama** (local, fast): `ragas_embedding_provider="ollama"`, `ollama_embedding_model="nomic-embed-text"`
- **HuggingFace** (local): `ragas_embedding_provider="huggingface"`, `embedding_model="BAAI/bge-base-en-v1.5"`

## What Was Fixed

### In `ragas.py`:

1. **Added imports** (lines 21-22):
```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
```

2. **Wrapped models before RAGAS** (in both evaluation methods):
```python
# Wrap LLM and embeddings for RAGAS (required for local models)
ragas_llm = LangchainLLMWrapper(self.judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(self.ragas_embeddings)

# Pass wrapped models to RAGAS
result = evaluate(
    dataset,
    metrics=[...],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)
```

3. **Removed duplicate initialization** (line 143):
```python
# BEFORE (BUG):
self.judge_llm = None  # This was overwriting the initialized LLM!

# AFTER (FIXED):
# Note: self.judge_llm is already initialized above
```

## Requirements

- ✅ Ollama running locally (for Ollama models)
- ✅ JSON file with pre-generated answers
- ✅ NO OpenAI API key needed
- ✅ NO internet connection needed (if using Ollama)

## JSON Format Required

```json
[
  {
    "question": "What is...",
    "answer": "The answer is...",
    "contexts": ["context1", "context2"],
    "ground_truth": "The correct answer..."
  }
]
```

## Troubleshooting

### Error: "OpenAI API key required"
- Make sure you're using the updated `ragas.py` with wrappers
- Verify `judge_llm_provider="ollama"` (not "openai")

### Error: "NoneType has no attribute 'agenerate_prompt'"
- This is fixed by removing the duplicate `self.judge_llm = None` line
- Make sure you're using the latest version of `ragas.py`

### Slow evaluation
- Use `evaluate_with_ragas_subset()` for quick testing
- Ollama embeddings are faster than HuggingFace

### Timeout errors
- Increase timeout: `os.environ["RAGAS_TIMEOUT"] = "1200"`
- Check if Ollama is running: `ollama list`

## Performance Tips

- **Fastest**: Ollama LLM + Ollama embeddings
- **Most accurate**: Gemini LLM + HuggingFace embeddings (requires API key)
- **Balanced**: Ollama LLM + HuggingFace embeddings

## Next Steps

1. Test with 3-10 samples using `evaluate_with_ragas_subset()`
2. If successful, run full evaluation with `evaluate_with_ragas()`
3. Results will show 4 metrics: faithfulness, answer_correctness, context_recall, context_precision
