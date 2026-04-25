"""
Prospectus extraction verifier.

Accepts a list of ``FundData`` objects and checks that all summary-prospectus
fields have been populated correctly.

Usage
-----
from simple_rag.extraction.prospectus_verifier import verify_funds
from simple_rag.models.fund import FundData

results = verify_funds([fund_ibmx, fund_vmgmx, fund_femv])
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Callable
from datetime import date
import re
from src.simple_rag.models.fund import FundData

# ---------------------------------------------------------------------------
# Validators  — each returns (ok: bool, note: str)
# ---------------------------------------------------------------------------

def _non_empty_str(val: Any) -> Tuple[bool, str]:
    if val and isinstance(val, str) and val.strip() not in ("", "N/A", "n/a", "NA"):
        return True, ""
    return False, "missing / N/A"


def _non_empty_list(val: Any) -> Tuple[bool, str]:
    if val and isinstance(val, list) and len(val) > 0:
        return True, ""
    return False, "empty list"


def _date_field(val: Any) -> Tuple[bool, str]:
    """Accept a date object or a date-like string."""
    if isinstance(val, date):
        return True, ""
    if isinstance(val, str) and re.match(r"[A-Z][a-z]+ \d{1,2}, \d{4}", val.strip()):
        return True, ""
    if val is None:
        return False, "missing"
    return False, f"unexpected format: {val!r}"


def _ticker_field(val: Any) -> Tuple[bool, str]:
    if val and isinstance(val, str) and re.match(r"^[A-Z]{1,6}$", val.strip()):
        return True, ""
    return False, f"unexpected: {val!r}"


def _metadata_field(val: Any) -> Tuple[bool, str]:
    """FilingMetadata object — just check it is not None."""
    if val is not None:
        return True, ""
    return False, "missing"


# ---------------------------------------------------------------------------
# Field definitions: (fund_attr, display_label, validator)
# ---------------------------------------------------------------------------

FIELDS: List[Tuple[str, str, Callable]] = [
    ("ticker",                       "Ticker",               _ticker_field),
    ("name",                         "Fund Name",            _non_empty_str),
    ("report_date",                  "Report Date",          _date_field),
    ("objective",                    "Objective",            _non_empty_str),
    ("strategies",                   "Strategies",           _non_empty_str),
    ("risks",                        "Risks",                _non_empty_str),
    ("managers",                     "Managers",             _non_empty_list),
    ("summary_prospectus",           "Summary Prospectus",   _non_empty_str),
    ("summary_prospectus_metadata",  "Prospectus Metadata",  _metadata_field),
]


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def _check_fund(fund: FundData) -> Tuple[Dict[str, Any], Dict[str, Tuple[bool, str]]]:
    """Validate all prospectus fields on a single FundData object."""
    data: Dict[str, Any] = {}
    results: Dict[str, Tuple[bool, str]] = {}

    for attr, _label, validator in FIELDS:
        value = getattr(fund, attr, None)
        data[attr] = value
        results[attr] = validator(value)

    return data, results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_funds(
    funds: List[FundData],
    *,
    print_details: bool = True,
    min_field_chars: int = 80,
) -> List[Dict[str, Any]]:
    """
    Verify prospectus fields across a list of FundData objects.

    Parameters
    ----------
    funds : List[FundData]
        Fund objects to validate.
    print_details : bool
        Print per-fund breakdown and summary table (default True).
    min_field_chars : int
        Text fields shorter than this trigger a soft ⚠ warning (not a failure).

    Returns
    -------
    List of dicts with keys: ticker, name, provider, data, results,
    ok_count, fail_count.
    """
    all_results: List[Dict[str, Any]] = []
    total_ok = 0
    total_fail = 0
    n_fields = len(FIELDS)
    label_width = max(len(lbl) for _, lbl, _ in FIELDS) + 2

    SEP = "─" * 72

    for fund in funds:
        ticker   = fund.ticker or "UNKNOWN"
        name     = fund.name   or "UNKNOWN"
        provider = fund.provider or "unknown provider"

        data, results = _check_fund(fund)

        ok_count   = sum(1 for ok, _ in results.values() if ok)
        fail_count = n_fields - ok_count
        total_ok   += ok_count
        total_fail += fail_count

        record = {
            "ticker":     ticker,
            "name":       name,
            "provider":   provider,
            "data":       data,
            "results":    results,
            "ok_count":   ok_count,
            "fail_count": fail_count,
        }
        all_results.append(record)

        if not print_details:
            continue

        # ── per-fund detail block ───────────────────────────────────────────
        status_icon = "✅" if fail_count == 0 else ("⚠️ " if ok_count >= n_fields // 2 else "❌")
        print(f"\n{SEP}")
        print(f"{status_icon}  {ticker}  —  {name}  [{provider}]")
        print(SEP)

        for attr, label, _ in FIELDS:
            ok, note = results[attr]
            value    = data[attr]

            # Build display string
            if isinstance(value, list):
                display = ", ".join(str(v) for v in value) if value else "—"
            elif value is None:
                display = "—"
            else:
                display = str(value)

            if len(display) > 60:
                display = display[:57] + "..."

            short_warn = ""
            if ok and isinstance(value, str) and len(value) < min_field_chars:
                short_warn = f"  ⚠  short ({len(value)} chars)"

            icon     = "✓" if ok else "✗"
            note_str = f"  [{note}]" if note and not ok else ""
            print(f"  {icon}  {label:<{label_width}} {display}{note_str}{short_warn}")

        print(f"\n  → {ok_count}/{n_fields} fields verified  |  {fail_count} issue(s)")

    # ── summary table ───────────────────────────────────────────────────────
    if print_details and funds:
        grand_total = n_fields * len(funds)
        print(f"\n{'═' * 72}")
        print(f"  SUMMARY  —  {len(funds)} fund(s) checked")
        print(f"{'═' * 72}")

        for rec in all_results:
            bar = "█" * rec["ok_count"] + "░" * rec["fail_count"]
            print(
                f"  {rec['ticker']:<8}  [{bar}]  "
                f"{rec['ok_count']}/{n_fields}  "
                f"({'✓ ok' if rec['fail_count'] == 0 else str(rec['fail_count']) + ' missing'})"
            )

        pct = 100 * total_ok / grand_total if grand_total else 0
        print(f"{'─' * 72}")
        print(f"  Total verified : {total_ok}/{grand_total} fields  ({pct:.0f}%)")
        print(f"  Total issues   : {total_fail}")
        print(f"{'═' * 72}\n")

    return all_results
