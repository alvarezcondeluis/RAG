"""
Streamlit service — subprocess launcher and readiness probe.
"""

from __future__ import annotations

import subprocess
import time
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

from simple_rag.rag.deployment.config import StreamlitConfig, TimeoutConfig
from simple_rag.rag.deployment.health import ServiceStatus


class StreamlitServiceError(RuntimeError):
    pass


def probe(port: int) -> bool:
    """Return True if Streamlit's health endpoint responds with 200."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def launch(config: StreamlitConfig, project_root: Path) -> subprocess.Popen:
    """
    Start Streamlit as a subprocess. Returns the Popen handle.
    Output is inherited so logs appear in the terminal.
    """
    app_path = project_root / config.app_path
    if not app_path.exists():
        raise StreamlitServiceError(
            f"Streamlit app not found: {app_path}\n"
            "  Check streamlit.app_path in deployment.yaml."
        )

    cmd = [
        "uv", "run", "streamlit", "run", str(app_path),
        "--server.port", str(config.port),
        "--server.headless", "true",
        "--server.fileWatcherType", "none",  # disable hot-reload in deployment mode
    ]
    print(f"  → Launching Streamlit: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        # Let stdout/stderr flow to the parent terminal so the user sees Streamlit logs
        stdout=None,
        stderr=None,
    )
    return proc


def wait_ready(port: int, timeouts: TimeoutConfig, status: ServiceStatus) -> None:
    """Poll /healthz until Streamlit is accepting connections."""
    deadline = time.time() + timeouts.probe_timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        if probe(port):
            status.mark_ready(f"http://localhost:{port}")
            return
        status.mark_starting(f"waiting for Streamlit... (attempt {attempt})")
        time.sleep(timeouts.probe_interval_s)

    status.mark_failed(f"not ready after {timeouts.probe_timeout_s}s")
    raise StreamlitServiceError(
        f"Streamlit did not start within {timeouts.probe_timeout_s}s on port {port}."
    )


def open_browser(port: int) -> None:
    webbrowser.open(f"http://localhost:{port}")


def ensure_ready(
    config: StreamlitConfig,
    timeouts: TimeoutConfig,
    status: ServiceStatus,
    project_root: Path,
) -> subprocess.Popen:
    """
    Launch Streamlit and wait until it's ready.
    Returns the subprocess handle for the caller to monitor.
    """
    status.mark_starting("launching...")
    proc = launch(config, project_root)
    wait_ready(config.port, timeouts, status)
    return proc
