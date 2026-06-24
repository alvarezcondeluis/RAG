"""
Pre-LLM result enhancement: truncation and empty-result handling.

Call `enhance(records)` before building the answer prompt. It returns an
`EnhancedResults` object that tells callers whether to skip the LLM entirely
and provides a cleaned, token-safe record list with an optional truncation note
appended to the prompt context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_DEFAULT_ROW_LIMIT = 50


@dataclass
class EnhancedResults:
    records: List[Dict[str, Any]]
    is_empty: bool = False
    empty_message: str = ""
    truncated: bool = False
    truncation_note: str = ""    # injected into the LLM prompt when non-empty
    original_count: int = 0


def enhance(
    records: Optional[List[Dict[str, Any]]],
    row_limit: int = _DEFAULT_ROW_LIMIT,
    query: str = "",
) -> EnhancedResults:
    """Enhance raw Neo4j records before passing them to the answer-generation LLM.

    Two responsibilities:
      1. Empty detection  — returns is_empty=True so callers can skip the LLM.
      2. Row truncation   — caps at row_limit and builds a truncation_note with
                           summary statistics so the LLM still understands scale.

    Args:
        records:   Raw Neo4j result rows (list of dicts). None treated as empty.
        row_limit: Maximum rows to keep (default 50).
        query:     Original user query — used only to personalise the empty message.

    Returns:
        EnhancedResults with .records (safe to pass to prompt), .is_empty, and
        .truncation_note (append to prompt when non-empty).
    """
    if not records:
        msg = (
            "No matching data was found in the database for your query."
            + (f' Try rephrasing or verifying the entity name in: "{query}".' if query else "")
        )
        return EnhancedResults(
            records=[],
            is_empty=True,
            empty_message=msg,
            original_count=0,
        )

    original_count = len(records)

    if original_count <= row_limit:
        return EnhancedResults(
            records=records,
            original_count=original_count,
        )

    # Truncate and build summary note
    truncated_records = records[:row_limit]
    note = _build_truncation_note(records, original_count, row_limit)

    return EnhancedResults(
        records=truncated_records,
        truncated=True,
        truncation_note=note,
        original_count=original_count,
    )


# ── Summary stats for the truncation note ────────────────────────────────────

def _build_truncation_note(
    records: List[Dict[str, Any]],
    original_count: int,
    row_limit: int,
) -> str:
    """Build a brief summary of omitted rows to include in the LLM prompt."""
    omitted = original_count - row_limit
    lines = [
        f"[NOTE: Output truncated. Showing {row_limit} of {original_count} total records"
        f" ({omitted} rows omitted to fit context).]"
    ]

    # Detect column types and add relevant aggregate stats
    all_keys: set[str] = set()
    for r in records:
        all_keys.update(r.keys())

    stats: List[str] = []

    # Weight / allocation summary
    if "weight" in all_keys:
        weights = [_to_float(r.get("weight")) for r in records]
        weights = [w for w in weights if w is not None]
        if weights:
            stats.append(f"weight range shown: {min(weights):.4f}–{max(weights):.4f}")

    # Market value summary
    if "marketValue" in all_keys:
        mvs = [_to_float(r.get("marketValue")) for r in records]
        mvs = [v for v in mvs if v is not None]
        if mvs:
            total = sum(mvs)
            stats.append(f"total market value (shown rows): {total:,.0f}")

    # Year range
    if "year" in all_keys:
        years = [r.get("year") for r in records if r.get("year") is not None]
        if years:
            stats.append(f"years covered (shown): {min(years)}–{max(years)}")

    if stats:
        lines.append("Summary stats for shown rows: " + "; ".join(stats) + ".")

    lines.append(
        "Consider these aggregate figures when answering — the full dataset may contain additional entries."
    )
    return " ".join(lines)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
