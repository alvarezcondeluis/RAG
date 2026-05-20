"""
Health status types and live console reporter for the deployment orchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


ServiceState = Literal["pending", "starting", "ready", "failed", "skipped"]

_STATE_ICON = {
    "pending":  "○",
    "starting": "◌",
    "ready":    "✓",
    "failed":   "✗",
    "skipped":  "–",
}

_STATE_COLOR = {
    "pending":  "\033[90m",   # gray
    "starting": "\033[93m",   # yellow
    "ready":    "\033[92m",   # green
    "failed":   "\033[91m",   # red
    "skipped":  "\033[90m",   # gray
}

_RESET = "\033[0m"


@dataclass
class ServiceStatus:
    name: str
    state: ServiceState = "pending"
    message: str = ""
    elapsed_s: float = 0.0
    _start: float = field(default_factory=time.time, repr=False)

    def mark_starting(self, message: str = "") -> None:
        self.state = "starting"
        self.message = message
        self._start = time.time()

    def mark_ready(self, message: str = "") -> None:
        self.state = "ready"
        self.elapsed_s = time.time() - self._start
        self.message = message

    def mark_failed(self, message: str = "") -> None:
        self.state = "failed"
        self.elapsed_s = time.time() - self._start
        self.message = message

    def mark_skipped(self, message: str = "") -> None:
        self.state = "skipped"
        self.elapsed_s = time.time() - self._start
        self.message = message


class DeploymentReporter:
    """
    Simple console reporter — prints one line per service event.
    Designed to be readable without any curses/rich dependency.
    """

    HEADER = """
╔══════════════════════════════════════════════════╗
║   SEC Filings Intelligence — System Deployment   ║
╚══════════════════════════════════════════════════╝
"""

    def __init__(self):
        self._statuses: list[ServiceStatus] = []
        print(self.HEADER)

    def register(self, name: str) -> ServiceStatus:
        """Create and track a new service status object."""
        s = ServiceStatus(name=name)
        self._statuses.append(s)
        return s

    def update(self, status: ServiceStatus) -> None:
        """Print the current state of a service."""
        color = _STATE_COLOR.get(status.state, "")
        icon = _STATE_ICON.get(status.state, "?")
        elapsed = f" ({status.elapsed_s:.1f}s)" if status.elapsed_s > 0 else ""
        msg = f"  {msg_text}" if (msg_text := status.message) else ""
        print(f"  {color}{icon} {status.name:<18}{_RESET}{elapsed}{msg}")

    def print_summary(self, streamlit_port: int = 8501) -> None:
        """Print the final deployment summary table."""
        print("\n" + "─" * 52)
        all_ready = all(s.state in ("ready", "skipped") for s in self._statuses)

        for s in self._statuses:
            color = _STATE_COLOR.get(s.state, "")
            icon = _STATE_ICON.get(s.state, "?")
            elapsed = f"{s.elapsed_s:.1f}s" if s.elapsed_s > 0 else "—"
            print(f"  {color}{icon} {s.name:<18} {elapsed:>6}{_RESET}  {s.message}")

        print("─" * 52)
        if all_ready:
            print(f"\n  \033[92m✓ System ready — http://localhost:{streamlit_port}\033[0m\n")
        else:
            failed = [s.name for s in self._statuses if s.state == "failed"]
            print(f"\n  \033[91m✗ Deployment failed — {', '.join(failed)}\033[0m\n")
