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

if backend == "groq":
    benchmark = Text2CypherBenchmark(PATH, GROQ_MODEL, backend="groq")
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
        llama_cpp_host="localhost",
        llama_cpp_port=8081,
    )
    try:
        benchmark.run()
    except KeyboardInterrupt:
        print("\nBenchmark stopped by user.")
        benchmark.cleanup()
