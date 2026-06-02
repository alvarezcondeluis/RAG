"""
llama.cpp server service management — health probe, model presence check, startup, readiness wait.

llama-server exposes an OpenAI-compatible REST API (default: http://localhost:8080).
"""

from __future__ import annotations

import subprocess
import time
import urllib.request
import urllib.error
import json

from simple_rag.rag.deployment.config import LlamaCppConfig, TimeoutConfig
from simple_rag.rag.deployment.health import ServiceStatus


class LlamaCppServiceError(RuntimeError):
    pass


def probe(config: LlamaCppConfig) -> bool:
    """
    Return True if the llama-server API is up (GET /v1/models returns 200).
    Timeout: 3 seconds.
    """
    try:
        req = urllib.request.Request(f"{config.base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def model_loaded(config: LlamaCppConfig) -> bool:
    """
    Return True if `config.model_id` appears in the list of loaded models.
    If model_id is blank, returns True (skip model check).
    """
    if not config.model_id:
        return True
    try:
        req = urllib.request.Request(f"{config.base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            loaded_ids = [m["id"] for m in data.get("data", [])]
            # Check by exact match or substring (llama-server uses filename as model id)
            return any(config.model_id in mid or mid in config.model_id for mid in loaded_ids)
    except Exception:
        return False


def start(config: LlamaCppConfig) -> None:
    """
    Attempt to start llama-server according to start_mode.

    "auto"   → launch llama-server subprocess (requires config.executable)
    "manual" → print instructions and raise so the caller can prompt the user
    """
    if config.start_mode == "manual":
        raise LlamaCppServiceError(
            "llama-server is not reachable and start_mode is 'manual'.\n"
            f"  Please start llama-server on {config.base_url}.\n"
            "  Example:\n"
            f"    llama-server --model {config.model_id or '/path/to/model.gguf'}"
            f" --ctx-size {config.ctx_size}"
            f" --port 8080\n"
            "  Or set llama_cpp.start_mode: auto with llama_cpp.executable in deployment.yaml"
        )

    if config.start_mode == "auto":
        if not config.executable:
            raise LlamaCppServiceError(
                "start_mode is 'auto' but llama_cpp.executable is not set.\n"
                "  Set it to the full path of the llama-server binary in deployment.yaml."
            )
        if not config.model_id:
            raise LlamaCppServiceError(
                "start_mode is 'auto' but llama_cpp.model_id is not set.\n"
                "  Set it to the full path of the GGUF model file in deployment.yaml."
            )

        cmd = [
            config.executable,
            "--model", config.model_id,
            "--ctx-size", str(config.ctx_size),
            "--port", config.base_url.split(":")[-1],  # extract port from base_url
        ]

        # Speculative decoding — only add flags when a draft model is configured
        if config.draft_model:
            cmd += [
                "--model-draft", config.draft_model,
                "--draft-max", str(config.draft_max),
            ]

        print(f"  → Launching llama-server: {' '.join(cmd)}")
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _print_manual_instructions(config: LlamaCppConfig) -> None:
    print("\n" + "─" * 60)
    print("  ACTION REQUIRED — llama-server is not running.")
    print("  Start it with:")
    model_path = config.model_id or "/path/to/model.gguf"
    print(f"    llama-server --model {model_path} \\")
    print(f"                 --ctx-size {config.ctx_size} \\")
    if config.draft_model:
        print(f"                 --model-draft {config.draft_model} \\")
        print(f"                 --draft-max {config.draft_max} \\")
    print(f"                 --port 8080")
    print(f"  Then press Enter to retry...")
    print("─" * 60)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def wait_ready(config: LlamaCppConfig, timeouts: TimeoutConfig, status: ServiceStatus) -> None:
    """
    Poll until the API server is up AND the configured model is loaded.
    In manual mode, print instructions and wait for user confirmation before retrying.
    """
    deadline = time.time() + timeouts.probe_timeout_s
    attempt = 0

    while time.time() < deadline:
        attempt += 1

        if not probe(config):
            if config.start_mode == "manual" and attempt == 1:
                _print_manual_instructions(config)
                deadline = time.time() + timeouts.probe_timeout_s
                attempt = 0
                continue
            status.mark_starting(f"waiting for server... (attempt {attempt})")
            time.sleep(timeouts.probe_interval_s)
            continue

        # Server is up — check model
        if config.model_id and not model_loaded(config):
            status.mark_starting(f"server up, waiting for model '{config.model_id}'...")
            time.sleep(timeouts.probe_interval_s)
            continue

        # All good
        model_msg = f"model '{config.model_id}' loaded" if config.model_id else "server ready"
        status.mark_ready(f"{config.base_url} — {model_msg}")
        return

    status.mark_failed(f"not ready after {timeouts.probe_timeout_s}s")
    raise LlamaCppServiceError(
        f"llama-server did not become ready within {timeouts.probe_timeout_s}s.\n"
        f"  Expected API at: {config.base_url}\n"
        + (f"  Expected model: {config.model_id}\n" if config.model_id else "")
    )


def ensure_ready(config: LlamaCppConfig, timeouts: TimeoutConfig, status: ServiceStatus) -> None:
    """
    Full lifecycle: probe → start if needed → wait until ready.
    """
    status.mark_starting("probing...")

    if probe(config) and model_loaded(config):
        model_msg = f"model '{config.model_id}' loaded" if config.model_id else "server running"
        status.mark_ready(f"already running — {model_msg}")
        return

    if not probe(config):
        status.mark_starting("not running — starting...")
        try:
            start(config)
        except LlamaCppServiceError:
            # Manual mode: start() raises, we fall through to wait_ready which
            # will prompt the user interactively
            pass

    wait_ready(config, timeouts, status)
