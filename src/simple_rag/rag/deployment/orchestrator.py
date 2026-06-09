"""
Deployment orchestrator — runs the full boot sequence and blocks until shutdown.

Boot order (strict dependency chain):
    [1] Neo4j         → graph database
    [2] llama.cpp     → text2cypher LLM backend
    [3] Streamlit     → frontend (pipeline is initialised inside the app)

Shutdown (Ctrl+C / SIGTERM):
    Streamlit subprocess → Neo4j Docker container (if we started it)
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from simple_rag.rag.deployment.config import DeploymentConfig
from simple_rag.rag.deployment.health import DeploymentReporter
from simple_rag.rag.deployment.services import neo4j_service, llama_cpp_service, streamlit_service


def deploy(config: DeploymentConfig) -> None:
    """
    Run the full deployment sequence and block until the user presses Ctrl+C.
    """
    reporter = DeploymentReporter()
    streamlit_proc: Optional[subprocess.Popen] = None
    neo4j_was_started = False   # track if we started Docker so we can stop it on shutdown

    # ── 1. Neo4j ──────────────────────────────────────────────────────────────
    neo4j_status = reporter.register("Neo4j")
    reporter.update(neo4j_status)
    try:
        # If it's already running we skip Docker
        was_running = neo4j_service.probe(config.neo4j)
        neo4j_service.ensure_ready(config.neo4j, config.timeouts, neo4j_status, config.project_root)
        neo4j_was_started = (not was_running) and config.neo4j.start_mode == "docker"
        reporter.update(neo4j_status)
    except neo4j_service.Neo4jServiceError as exc:
        neo4j_status.mark_failed(str(exc).splitlines()[0])
        reporter.update(neo4j_status)
        reporter.print_summary(config.streamlit.port)
        sys.exit(1)

    # ── 2. llama.cpp ──────────────────────────────────────────────────────────
    lm_status = reporter.register("llama.cpp")
    reporter.update(lm_status)
    try:
        llama_cpp_service.ensure_ready(config.llama_cpp, config.timeouts, lm_status)
        reporter.update(lm_status)
    except llama_cpp_service.LlamaCppServiceError as exc:
        lm_status.mark_failed(str(exc).splitlines()[0])
        reporter.update(lm_status)
        _shutdown(streamlit_proc, neo4j_was_started, config)
        reporter.print_summary(config.streamlit.port)
        sys.exit(1)

    # ── 3. Streamlit ──────────────────────────────────────────────────────────
    streamlit_status = reporter.register("Streamlit")
    reporter.update(streamlit_status)
    try:
        streamlit_proc = streamlit_service.ensure_ready(
            config.streamlit, config.timeouts, streamlit_status, config.project_root
        )
        reporter.update(streamlit_status)
    except streamlit_service.StreamlitServiceError as exc:
        streamlit_status.mark_failed(str(exc).splitlines()[0])
        reporter.update(streamlit_status)
        _shutdown(streamlit_proc, neo4j_was_started, config)
        reporter.print_summary(config.streamlit.port)
        sys.exit(1)

    # ── All ready ─────────────────────────────────────────────────────────────
    reporter.print_summary(config.streamlit.port)
    if config.streamlit.open_browser:
        # Small delay so the browser opens after the summary is printed
        time.sleep(0.5)
        streamlit_service.open_browser(config.streamlit.port)

    # ── Block until Ctrl+C / SIGTERM ─────────────────────────────────────────
    def _handle_signal(signum, frame):
        print("\n\n  Shutting down...")
        _shutdown(streamlit_proc, neo4j_was_started, config)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print("  Press Ctrl+C to stop the system.\n")
    try:
        # Monitor the Streamlit process — if it dies unexpectedly, exit
        while True:
            if streamlit_proc.poll() is not None:
                print("\n  ⚠  Streamlit process exited unexpectedly.")
                _shutdown(None, neo4j_was_started, config)
                sys.exit(1)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\n  Shutting down...")
        _shutdown(streamlit_proc, neo4j_was_started, config)
        sys.exit(0)


def _shutdown(
    streamlit_proc: Optional[subprocess.Popen],
    neo4j_was_started: bool,
    config: DeploymentConfig,
) -> None:
    """Graceful teardown in reverse boot order."""
    # 1. Stop Streamlit
    if streamlit_proc is not None and streamlit_proc.poll() is None:
        print("  Stopping Streamlit...")
        streamlit_proc.terminate()
        try:
            streamlit_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_proc.kill()

    # 2. Stop Neo4j Docker (only if we started it)
    if neo4j_was_started and config.neo4j.start_mode == "docker":
        compose_dir = config.project_root / config.neo4j.compose_dir
        print(f"  Stopping Neo4j container ({config.neo4j.container_name})...")
        subprocess.run(
            ["docker", "compose", "stop"],
            cwd=compose_dir,
            capture_output=True,
        )

    print("  ✓ System shut down cleanly.")
