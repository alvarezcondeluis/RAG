"""
Neo4j service management — health probe, Docker startup, readiness wait.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from simple_rag.rag.deployment.config import Neo4jConfig, TimeoutConfig
from simple_rag.rag.deployment.health import ServiceStatus


class Neo4jServiceError(RuntimeError):
    pass


def probe(config: Neo4jConfig) -> bool:
    """
    Return True if Neo4j is reachable and accepts the configured credentials.
    Fast — uses a 3-second socket timeout.
    """
    try:
        import neo4j as neo4j_lib
        driver = neo4j_lib.GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
            connection_timeout=3,
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def start(config: Neo4jConfig, project_root: Path) -> None:
    """
    Attempt to start Neo4j according to start_mode.

    "docker"  → docker compose up -d  (in project_root / compose_dir)
    "manual"  → raise with human-readable instructions
    """
    if config.start_mode == "manual":
        raise Neo4jServiceError(
            "Neo4j is not reachable and start_mode is 'manual'.\n"
            f"  Please start Neo4j manually and ensure it listens on {config.uri}.\n"
            "  Docker: cd neo4j && docker compose up -d\n"
            "  Or set neo4j.start_mode: docker in deployment.yaml"
        )

    if config.start_mode == "docker":
        compose_dir = project_root / config.compose_dir
        if not (compose_dir / "docker-compose.yml").exists():
            raise Neo4jServiceError(
                f"docker-compose.yml not found at {compose_dir}.\n"
                "  Check neo4j.compose_dir in deployment.yaml."
            )

        print(f"  → Starting Neo4j container via Docker Compose ({compose_dir}) ...")
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=compose_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Neo4jServiceError(
                f"docker compose up failed:\n{result.stderr.strip()}"
            )
        print(f"  → Container '{config.container_name}' started.")


def wait_ready(config: Neo4jConfig, timeouts: TimeoutConfig, status: ServiceStatus) -> None:
    """
    Poll probe() until Neo4j accepts connections or the timeout is exceeded.
    Updates `status` on each attempt.
    """
    deadline = time.time() + timeouts.probe_timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        if probe(config):
            status.mark_ready(f"connected to {config.uri}")
            return
        status.mark_starting(f"waiting for Neo4j... (attempt {attempt})")
        time.sleep(timeouts.probe_interval_s)

    status.mark_failed(f"not reachable after {timeouts.probe_timeout_s}s")
    raise Neo4jServiceError(
        f"Neo4j did not become ready within {timeouts.probe_timeout_s}s.\n"
        f"  URI: {config.uri}\n"
        "  Check that the container is running: docker ps | grep fund_graph_db"
    )


def ensure_ready(config: Neo4jConfig, timeouts: TimeoutConfig, status: ServiceStatus, project_root: Path) -> None:
    """
    Full lifecycle: probe → start if needed → wait until ready.
    Raises Neo4jServiceError on failure.
    """
    status.mark_starting("probing...")
    if probe(config):
        status.mark_ready(f"already running at {config.uri}")
        return

    # Not reachable — try to start
    status.mark_starting("not running — starting...")
    start(config, project_root)

    # Now wait
    wait_ready(config, timeouts, status)
