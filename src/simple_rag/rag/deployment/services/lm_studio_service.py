"""
LM Studio service management — health probe, model presence check, startup, readiness wait.

LM Studio exposes an OpenAI-compatible REST API (default: http://localhost:1234).
"""

from __future__ import annotations

import subprocess
import time
import urllib.request
import urllib.error
import json

from simple_rag.rag.deployment.config import LMStudioConfig, TimeoutConfig
from simple_rag.rag.deployment.health import ServiceStatus


class LMStudioServiceError(RuntimeError):
    pass


def probe(config: LMStudioConfig) -> bool:
    """
    Return True if the LM Studio API server is up (GET /v1/models returns 200).
    Timeout: 3 seconds.
    """
    try:
        req = urllib.request.Request(f"{config.base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def model_loaded(config: LMStudioConfig) -> bool:
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
            # Check by exact match or substring (LM Studio sometimes appends quantization suffix)
            return any(config.model_id in mid or mid in config.model_id for mid in loaded_ids)
    except Exception:
        return False


def start(config: LMStudioConfig) -> None:
    """
    Attempt to start LM Studio according to start_mode.

    "auto"    → launch LM Studio CLI server (requires config.executable)
    "manual"  → print instructions and raise so the orchestrator can prompt the user
    """
    if config.start_mode == "manual":
        raise LMStudioServiceError(
            "LM Studio API server is not reachable and start_mode is 'manual'.\n"
            f"  Please start LM Studio and enable the Local Server on {config.base_url}.\n"
            "  In LM Studio: Local Server tab → Start Server\n"
            "  Or set lm_studio.start_mode: auto with lm_studio.executable in deployment.yaml"
        )

    if config.start_mode == "auto":
        if not config.executable:
            raise LMStudioServiceError(
                "start_mode is 'auto' but lm_studio.executable is not set.\n"
                "  Set it to the full path of the LM Studio CLI binary in deployment.yaml."
            )
        print(f"  → Launching LM Studio server: {config.executable}")
        subprocess.Popen(
            [config.executable, "server", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _print_manual_instructions(config: LMStudioConfig) -> None:
    print("\n" + "─" * 52)
    print("  ACTION REQUIRED — LM Studio is not running.")
    print(f"  1. Open LM Studio")
    print(f"  2. Go to the 'Local Server' tab")
    print(f"  3. Click 'Start Server' (port should be 1234)")
    if config.model_id:
        print(f"  4. Load model: {config.model_id}")
    print(f"  Then press Enter to retry...")
    print("─" * 52)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def wait_ready(config: LMStudioConfig, timeouts: TimeoutConfig, status: ServiceStatus) -> None:
    """
    Poll until the API server is up AND the configured model is loaded.
    In manual mode, if the server doesn't come up within the timeout,
    print instructions and wait for user confirmation before retrying.
    """
    deadline = time.time() + timeouts.probe_timeout_s
    attempt = 0

    while time.time() < deadline:
        attempt += 1

        if not probe(config):
            if config.start_mode == "manual" and attempt == 1:
                # Give user a chance to start LM Studio manually
                _print_manual_instructions(config)
                deadline = time.time() + timeouts.probe_timeout_s  # reset timer after user ack
                attempt = 0
                continue
            status.mark_starting(f"waiting for API server... (attempt {attempt})")
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
    raise LMStudioServiceError(
        f"LM Studio did not become ready within {timeouts.probe_timeout_s}s.\n"
        f"  Expected API at: {config.base_url}\n"
        + (f"  Expected model: {config.model_id}\n" if config.model_id else "")
    )


def ensure_ready(config: LMStudioConfig, timeouts: TimeoutConfig, status: ServiceStatus) -> None:
    """
    Full lifecycle: probe → start if needed → wait until ready.
    """
    status.mark_starting("probing...")

    if probe(config) and model_loaded(config):
        model_msg = f"model '{config.model_id}' loaded" if config.model_id else "server running"
        status.mark_ready(f"already running — {model_msg}")
        return

    # Not ready — try to start (or instruct)
    if not probe(config):
        status.mark_starting("not running — starting...")
        try:
            start(config)
        except LMStudioServiceError:
            # Manual mode: start() raises, we fall through to wait_ready which
            # will prompt the user interactively
            pass

    wait_ready(config, timeouts, status)
