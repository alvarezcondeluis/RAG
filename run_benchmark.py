import sys
from pathlib import Path

# Setup paths assuming execution from project root
ROOT_DIR = Path(".")
SRC_DIR = Path("./src")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simple_rag.evaluation.benchmark import Text2CypherBenchmark

MODEL = "tomasonjo/llama3-text2cypher-demo:8b_4bit"
print(f"Current path: {Path.cwd()}")

# Adjusted path for root execution
PATH = Path("./src/simple_rag/evaluation/test_set.json")
GROQ_MODEL = "llama-3.3-70b-versatile"
backend = "openai"  # llama.cpp server

# Schema mode:
#   True  — classifier picks a focused schema slice for each query (default)
#   False — full DETAILED_SCHEMA is always used (classifier still routes, but no slice injection)
# Regardless of this setting, retries on failed queries always use the full DETAILED_SCHEMA.
USE_SCHEMA_INJECTION = True

if backend == "groq":
    benchmark = Text2CypherBenchmark(PATH, GROQ_MODEL, backend="groq", use_schema_injection=USE_SCHEMA_INJECTION)
    try:
        benchmark.run(complexity_filter=["hard"])
    except KeyboardInterrupt:
        print("\nBenchmark stopped by user.")
        benchmark.cleanup()
else:
    benchmark = Text2CypherBenchmark(
        PATH,
        MODEL,
        backend="openai",
        interactive=False,
        openai_compatible_host="localhost",
        openai_compatible_port=8081,
        use_schema_injection=USE_SCHEMA_INJECTION,
    )
    try:
        benchmark.run()
    except KeyboardInterrupt:
        print("\nBenchmark stopped by user.")
        benchmark.cleanup()
