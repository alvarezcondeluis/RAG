#!/usr/bin/env python3
"""
Benchmark runner for the full QueryHandler pipeline.
Executes the Text2CypherBenchmark and prints a full report.

Backends
--------
  ollama  – Local Ollama server (auto-started if needed)
  groq    – Groq cloud API  (set GROQ_API_KEY env var)
  openai  – Local llama.cpp server exposing an OpenAI-compatible API
             Start the server first:
               llama-server -m ~/models/qwen3-coder/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
                            --port 8080 --ctx-size 4096 -ngl 99
             Or use the helper script:
               python scripts/start_llama_server.py
"""

import sys
import os

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

from pathlib import Path

# Ensure project root is on sys.path
SRC_DIR = Path(__file__).resolve().parents[2]  # src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simple_rag.evaluation.benchmark import Text2CypherBenchmark

# ─── Configuration ────────────────────────────────────────────────────────────

# Change BACKEND to "ollama", "groq", or "openai"
BACKEND = "openai"

# Ollama / HuggingFace model
OLLAMA_MODEL = "tomasonjo/llama3-text2cypher-demo:8b_4bit"

# Groq model (cloud)
GROQ_MODEL = "llama-3.3-70b-versatile"

# llama.cpp local server settings (only used when BACKEND = "openai")
LLAMA_CPP_HOST = "localhost"
LLAMA_CPP_PORT = 8081
# Model name is passed to the API but llama.cpp ignores it — any string works.
LLAMA_CPP_MODEL = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M"

PATH = SRC_DIR / "simple_rag" / "evaluation" / "test_set.json"

# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Test set : {PATH}",   flush=True)
    print(f"Backend  : {BACKEND}", flush=True)

    try:
        if BACKEND == "groq":
            print(f"Model    : {GROQ_MODEL}", flush=True)
            benchmark = Text2CypherBenchmark(
                str(PATH), GROQ_MODEL, backend="groq"
            )
            benchmark.run(complexity_filter=["hard"])

        elif BACKEND == "openai":
            print(f"Model    : {LLAMA_CPP_MODEL}", flush=True)
            print(f"Server   : http://{LLAMA_CPP_HOST}:{LLAMA_CPP_PORT}", flush=True)
            benchmark = Text2CypherBenchmark(
                str(PATH),
                LLAMA_CPP_MODEL,
                backend="openai",
                llama_cpp_host=LLAMA_CPP_HOST,
                llama_cpp_port=LLAMA_CPP_PORT,
            )
            benchmark.run()

        else:  # ollama (default)
            print(f"Model    : {OLLAMA_MODEL}", flush=True)
            benchmark = Text2CypherBenchmark(
                str(PATH), OLLAMA_MODEL, backend="ollama", interactive=False
            )
            benchmark.run()

    except KeyboardInterrupt:
        print("\nBenchmark stopped by user.")
        benchmark.cleanup()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        benchmark.cleanup()
