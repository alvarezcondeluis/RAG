"""
CLI entrypoint for the deployment module.

Usage:
    uv run python -m simple_rag.rag.deployment
    uv run python -m simple_rag.rag.deployment --config deployment.yaml
    uv run python -m simple_rag.rag.deployment --neo4j-start-mode docker
    uv run python -m simple_rag.rag.deployment --answer-provider groq --answer-model llama-3.3-70b-versatile
    uv run python -m simple_rag.rag.deployment --port 8502 --no-browser
    uv run python -m simple_rag.rag.deployment --model /models/qwen2.5-coder-7b-q8_0.gguf --ctx-size 16384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project src/ is on the path when run as a module from the project root
_src = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import load_dotenv
load_dotenv(_src.parent / ".env")

from simple_rag.rag.deployment.config import load_config
from simple_rag.rag.deployment.orchestrator import deploy


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simple_rag.rag.deployment",
        description="Deploy the full SEC Filings Intelligence RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from deployment.yaml
  uv run python -m simple_rag.rag.deployment

  # Start Neo4j automatically via Docker
  uv run python -m simple_rag.rag.deployment --neo4j-start-mode docker

  # Custom LM Studio model + different answer provider
  uv run python -m simple_rag.rag.deployment \\
      --lm-studio-model "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF" \\
      --answer-provider gemini --answer-model gemini-1.5-flash-latest

  # Different port, skip browser
  uv run python -m simple_rag.rag.deployment --port 8502 --no-browser
""",
    )

    # Config file
    p.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to deployment.yaml (default: ./deployment.yaml)",
    )

    # Neo4j
    neo = p.add_argument_group("Neo4j")
    neo.add_argument(
        "--neo4j-start-mode",
        choices=["docker", "manual"],
        default=None,
        help="How to start Neo4j if not running (overrides deployment.yaml)",
    )

    # llama.cpp server
    lm = p.add_argument_group("llama.cpp server")
    lm.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="Path to the GGUF model file (e.g. /models/qwen2.5-coder-7b-q8_0.gguf)",
    )
    lm.add_argument(
        "--url",
        metavar="URL",
        default=None,
        help="llama-server base URL (default: http://localhost:8080)",
    )
    lm.add_argument(
        "--ctx-size",
        type=int,
        default=None,
        metavar="N",
        help="KV-cache / context window size in tokens (default: 8192)",
    )
    lm.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="F",
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    lm.add_argument(
        "--draft-model",
        metavar="PATH",
        default=None,
        help="Path to draft GGUF for speculative decoding (optional)",
    )

    # Answer LLM
    ans = p.add_argument_group("Answer LLM")
    ans.add_argument("--answer-provider", metavar="NAME", default=None,
                     help="Answer generation provider: groq | openrouter | gemini | ollama")
    ans.add_argument("--answer-model", metavar="MODEL_ID", default=None,
                     help="Answer generation model ID")

    # Streamlit
    st = p.add_argument_group("Streamlit")
    st.add_argument("--port", type=int, default=None, help="Streamlit port (default: 8501)")
    st.add_argument("--no-browser", action="store_true",
                    help="Do not open browser automatically after startup")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    overrides: dict = {}
    if args.neo4j_start_mode:
        overrides["neo4j_start_mode"] = args.neo4j_start_mode
    if args.model:
        overrides["llama_cpp_model"] = args.model
    if args.url:
        overrides["llama_cpp_url"] = args.url
    if args.answer_provider:
        overrides["answer_provider"] = args.answer_provider
    if args.answer_model:
        overrides["answer_model"] = args.answer_model
    if args.port:
        overrides["port"] = args.port

    config = load_config(yaml_path=args.config, overrides=overrides)

    # Apply inference parameter overrides directly on config
    if args.ctx_size is not None:
        config.llama_cpp.ctx_size = args.ctx_size
    if args.temperature is not None:
        config.llama_cpp.temperature = args.temperature
    if args.draft_model:
        config.llama_cpp.draft_model = args.draft_model

    if args.no_browser:
        config.streamlit.open_browser = False

    deploy(config)


if __name__ == "__main__":
    main()
