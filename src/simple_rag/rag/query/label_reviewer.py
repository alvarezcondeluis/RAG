#!/usr/bin/env python3
"""
Interactive label reviewer for training_data.json.

Usage:
    uv run src/simple_rag/rag/query/label_reviewer.py
    uv run src/simple_rag/rag/query/label_reviewer.py --split test
    uv run src/simple_rag/rag/query/label_reviewer.py --start 50

Controls (during review):
    1-6      Toggle a label on/off
    Enter    Accept current labels and advance
    p        Go back to previous entry
    s        Skip (keep labels as-is) and advance
    q        Save and quit
"""

import json
import sys
import os
import argparse
import termios
import tty
from pathlib import Path
from typing import List, Dict, Any

# ─── Constants ────────────────────────────────────────────────────────────────

DATA_FILE = Path(__file__).parent / "training_data.json"
CURSOR_FILE = Path(__file__).parent / ".reviewer_cursor.json"

LABELS = [
    "not_related",
    "fund_basic",
    "fund_portfolio",
    "fund_profile",
    "company_filing",
    "company_people",
]

LABEL_COLORS = {
    "not_related":    "\033[90m",   # gray
    "fund_basic":     "\033[94m",   # blue
    "fund_portfolio": "\033[96m",   # cyan
    "fund_profile":   "\033[93m",   # yellow
    "company_filing": "\033[91m",   # red
    "company_people": "\033[95m",   # magenta
}
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
DIM   = "\033[2m"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def getch() -> str:
    """Read a single character from stdin without requiring Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def clear() -> None:
    os.system("clear")


def label_badge(label: str, active: bool) -> str:
    color = LABEL_COLORS.get(label, "")
    if active:
        return f"{BOLD}{color}[✓ {label}]{RESET}"
    return f"{DIM}[ {label}]{RESET}"


def migrate_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure entry uses 'labels' (list) instead of 'label' (str)."""
    if "label" in entry and "labels" not in entry:
        entry["labels"] = [entry.pop("label")]
    elif "labels" not in entry:
        entry["labels"] = []
    # Clean up stale 'label' key if both exist
    entry.pop("label", None)
    return entry


def load_data(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    # Migrate all entries to multi-label format
    for split in ("train", "test"):
        data[split] = [migrate_entry(e) for e in data.get(split, [])]
    return data


def save_data(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cursor() -> Dict[str, Any]:
    if CURSOR_FILE.exists():
        with open(CURSOR_FILE) as f:
            return json.load(f)
    return {}


def save_cursor(split: str, idx: int) -> None:
    with open(CURSOR_FILE, "w") as f:
        json.dump({"split": split, "idx": idx}, f)


def print_header(split: str, idx: int, total: int) -> None:
    pct = (idx / total * 100) if total else 0
    bar_len = 40
    filled = int(bar_len * idx / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n{BOLD}{'─'*70}{RESET}")
    print(f"  {BOLD}Label Reviewer{RESET}  |  split: {BOLD}{split}{RESET}  |  "
          f"{BOLD}{idx+1}/{total}{RESET}  ({pct:.0f}%)")
    print(f"  [{bar}]")
    print(f"{'─'*70}")


def print_controls() -> None:
    print(f"\n  {DIM}Controls:{RESET}")
    for i, label in enumerate(LABELS, 1):
        color = LABEL_COLORS.get(label, "")
        print(f"    {BOLD}{i}{RESET} → {color}{label}{RESET}")
    print(f"    {BOLD}Enter{RESET} → accept & next  |  "
          f"{BOLD}p{RESET} → previous  |  "
          f"{BOLD}s{RESET} → skip  |  "
          f"{BOLD}q{RESET} → save & quit")


def review_entry(entry: Dict[str, Any], split: str, idx: int, total: int) -> str:
    """
    Interactive review loop for a single entry.

    Returns:
        'next'  — advance to next
        'prev'  — go back
        'skip'  — keep unchanged, advance
        'quit'  — save and exit
    """
    active: List[str] = list(entry.get("labels", []))

    while True:
        clear()
        print_header(split, idx, total)

        # Question
        text = entry["text"]
        print(f"\n  {BOLD}Question:{RESET}")
        # Word-wrap at 65 chars
        words = text.split()
        line, lines = "", []
        for w in words:
            if len(line) + len(w) + 1 > 65:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        for l in lines:
            print(f"    {l}")

        # Current labels
        print(f"\n  {BOLD}Labels:{RESET}")
        for i, label in enumerate(LABELS, 1):
            is_active = label in active
            print(f"    {BOLD}{i}{RESET}  {label_badge(label, is_active)}")

        print_controls()
        print(f"\n  Press a key: ", end="", flush=True)

        ch = getch()

        if ch in ("q", "Q"):
            entry["labels"] = active
            return "quit"
        elif ch in ("p", "P"):
            entry["labels"] = active
            return "prev"
        elif ch in ("s", "S"):
            # Skip — don't modify labels
            return "skip"
        elif ch in ("\r", "\n"):
            entry["labels"] = active
            return "next"
        elif ch.isdigit():
            n = int(ch)
            if 1 <= n <= len(LABELS):
                label = LABELS[n - 1]
                if label in active:
                    active.remove(label)
                else:
                    active.append(label)
        # Any other key — re-render


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive label reviewer for training_data.json")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="Which split to review (default: both — train first, then test)")
    parser.add_argument("--start", type=int, default=None,
                        help="Start at this index (0-based, within the chosen split)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore saved cursor position and start from 0")
    args = parser.parse_args()

    data = load_data(DATA_FILE)
    cursor = {} if args.no_resume else load_cursor()

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        entries = data[split]
        total = len(entries)

        # Determine start index
        if args.start is not None:
            start = args.start
        elif cursor.get("split") == split:
            start = cursor.get("idx", 0)
        else:
            start = 0

        print(f"\n{GREEN}▶ Reviewing '{split}' split ({total} entries) — starting at {start}{RESET}")
        if start > 0:
            print(f"  (resuming from saved cursor — use --no-resume to restart)")
        input("  Press Enter to begin...")

        idx = start
        while 0 <= idx < total:
            entry = entries[idx]
            action = review_entry(entry, split, idx, total)

            if action == "quit":
                save_data(DATA_FILE, data)
                save_cursor(split, idx)
                print(f"\n{GREEN}✓ Saved. Resume with: uv run label_reviewer.py (will pick up at {split}[{idx}]){RESET}\n")
                sys.exit(0)
            elif action == "prev":
                save_data(DATA_FILE, data)
                idx = max(0, idx - 1)
            elif action in ("next", "skip"):
                save_data(DATA_FILE, data)
                save_cursor(split, idx + 1)
                idx += 1

        # Finished this split
        save_cursor(split, total)
        clear()
        print(f"\n{GREEN}✓ Finished reviewing '{split}' split!{RESET}")
        if split != splits[-1]:
            input("  Press Enter to continue to next split...")

    save_data(DATA_FILE, data)
    CURSOR_FILE.unlink(missing_ok=True)
    print(f"\n{GREEN}✓ All splits reviewed. training_data.json saved.{RESET}\n")


if __name__ == "__main__":
    main()
