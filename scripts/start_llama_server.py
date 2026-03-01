#!/usr/bin/env python3
"""
Start the llama.cpp server with the Qwen3-Coder model.

This script launches `llama-server` as a subprocess and keeps it running
until you press Ctrl+C. The server exposes an OpenAI-compatible REST API
at http://localhost:8080/v1 that the RAG pipeline can use directly.

Usage:
    python scripts/start_llama_server.py

Requirements:
    llama.cpp must be installed and `llama-server` must be on your PATH.
    Download from: https://github.com/ggml-org/llama.cpp/releases
    (grab the llama-server binary for your platform)
"""

import subprocess
import sys
import signal
import os
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = Path.home() / "models" / "qwen3-coder" / "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"

PORT          = 8080
CTX_SIZE      = 4096   # context window (tokens)
GPU_LAYERS    = 99     # layers to offload to GPU (-1 = all, 0 = CPU only)
THREADS       = 8      # CPU threads (used for layers not on GPU)
BATCH_SIZE    = 512    # prompt processing batch size
HOST          = "127.0.0.1"

# ─── Build command ────────────────────────────────────────────────────────────

CMD = [
    "llama-server",
    "--model",      str(MODEL_PATH),
    "--host",       HOST,
    "--port",       str(PORT),
    "--ctx-size",   str(CTX_SIZE),
    "-ngl",         str(GPU_LAYERS),
    "--threads",    str(THREADS),
    "--batch-size", str(BATCH_SIZE),
    "--log-disable",          # silence verbose llama.cpp logs
]

# ─── Launch ───────────────────────────────────────────────────────────────────

def main():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("   Download it with:")
        print("   huggingface-cli download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \\")
        print("     --include 'Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf' \\")
        print(f"    --local-dir {MODEL_PATH.parent}")
        sys.exit(1)

    print("=" * 60)
    print("🦙 Starting llama.cpp server")
    print("=" * 60)
    print(f"  Model   : {MODEL_PATH.name}")
    print(f"  Address : http://{HOST}:{PORT}")
    print(f"  Context : {CTX_SIZE} tokens")
    print(f"  GPU layers: {GPU_LAYERS}")
    print(f"\nOpenAI-compatible API available at:")
    print(f"  http://{HOST}:{PORT}/v1/chat/completions")
    print("\nPress Ctrl+C to stop.\n")
    print(" ".join(CMD))
    print()

    proc = subprocess.Popen(CMD, stdout=sys.stdout, stderr=sys.stderr)

    def _shutdown(sig, frame):
        print("\n🛑 Shutting down llama.cpp server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("✓ Server stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    proc.wait()


if __name__ == "__main__":
    main()
