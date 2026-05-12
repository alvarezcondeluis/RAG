import sys
import argparse
from pathlib import Path

# Setup paths assuming execution from project root
ROOT_DIR = Path(".")
SRC_DIR = Path("./src")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simple_rag.evaluation.benchmark import Text2CypherBenchmark

# ── Embedding model aliases ──────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "nomic":  "nomic-ai/nomic-embed-text-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run the Text2Cypher benchmark.")
parser.add_argument(
    "--backend",
    choices=["openai", "groq", "ollama"],
    default="openai",
    help="LLM backend (default: openai — any OpenAI-compatible server like llama.cpp/LM Studio)",
)
parser.add_argument(
    "--model",
    default="tomasonjo/llama3-text2cypher-demo:8b_4bit",
    help="Text2Cypher model name (Ollama tag, Groq model ID, or OpenAI-compat model string)",
)
parser.add_argument(
    "--host",
    default="localhost",
    help="Hostname for OpenAI-compatible server (default: localhost)",
)
parser.add_argument(
    "--port",
    type=int,
    default=8081,
    help="Port for OpenAI-compatible server (default: 8081)",
)
parser.add_argument(
    "--few-shot-model",
    choices=list(EMBEDDING_MODELS.keys()),
    default="nomic",
    help="Embedding model for few-shot example retrieval (default: nomic)",
)
parser.add_argument(
    "--no-schema-injection",
    action="store_true",
    help="Disable schema slice injection (use full schema for every query)",
)
parser.add_argument(
    "--no-retry",
    action="store_true",
    help="Disable the validator retry loop (generate once, no retry prompts)",
)
parser.add_argument(
    "--complexity",
    nargs="+",
    choices=["easy", "medium", "hard"],
    default=None,
    help="Run only questions of the given complexity level(s)",
)
parser.add_argument(
    "--interactive",
    action="store_true",
    help="Pause after each failed query for manual review",
)
parser.add_argument(
    "--test-set",
    default="./src/simple_rag/evaluation/test_set.json",
    help="Path to the test set JSON file",
)

args = parser.parse_args()
print(f"Current path: {Path.cwd()}")

few_shot_model = EMBEDDING_MODELS[args.few_shot_model]
print(f"Few-shot embedding model: {few_shot_model}")

benchmark = Text2CypherBenchmark(
    test_set_path=args.test_set,
    model_name=args.model,
    backend=args.backend,
    interactive=args.interactive,
    openai_compatible_host=args.host,
    openai_compatible_port=args.port,
    use_schema_injection=not args.no_schema_injection,
    few_shot_embedding_model=few_shot_model,
    retry_module=not args.no_retry,
)

try:
    benchmark.run(complexity_filter=args.complexity)
except KeyboardInterrupt:
    print("\nBenchmark stopped by user.")
    benchmark.cleanup()
