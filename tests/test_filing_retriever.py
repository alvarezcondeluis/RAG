"""
Test FilingRetriever: fetch N-CSR filings by CIK and accession number.

Usage:
    cd /home/luis/Desktop/code/RAG
    .venv/bin/python tests/test_filing_retriever.py
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simple_rag.extraction.filing_retriever import FilingRetriever


EMAIL = "luis.alvarez.conde@alumnos.upm.es"


def print_funds_summary(result, label: str):
    """Pretty-print the extraction results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Filings processed : {result.filings_processed}")
    print(f"  Funds extracted   : {len(result.funds)}")
    print(f"  Highlights DFs    : {len(result.financial_highlights)}")
    print(f"{'-'*60}")

    for fund in result.funds:
        print(f"\n  Fund: {fund.name}")
        print(f"    Ticker          : {fund.ticker}")
        print(f"    Context ID      : {fund.context_id}")
        print(f"    Report date     : {fund.report_date}")
        print(f"    Share class     : {fund.share_class}")
        print(f"    Expense ratio   : {fund.expense_ratio}")
        print(f"    Net assets      : {fund.net_assets}")
        print(f"    Turnover rate   : {fund.turnover_rate}")
        print(f"    N holdings      : {fund.n_holdings}")

        if fund.top_holdings is not None:
            print(f"    Top holdings    : {len(fund.top_holdings)} rows")
        if fund.performance_table is not None:
            print(f"    Performance tbl : {len(fund.performance_table)} rows")
        if fund.annual_returns:
            print(f"    Annual returns  : {fund.annual_returns}")
        if fund.financial_highlights:
            years = sorted(fund.financial_highlights.keys())
            print(f"    Fin. highlights : {len(years)} years ({years[0]}–{years[-1]})")

        if fund.ncsr_metadata:
            m = fund.ncsr_metadata
            print(f"    Metadata        : accession={m.accession_number}, "
                  f"filed={m.filing_date}")

    # Show highlights DataFrame summary
    for i, df in enumerate(result.financial_highlights):
        if df is not None and not df.empty:
            print(f"\n  Highlights DF [{i}]: {len(df)} rows, "
                  f"columns={list(df.columns)}")
    print()


def test_by_cik():
    """Test fetching N-CSR filings by CIK (Vanguard Index Funds: 0000036405)."""
    retriever = FilingRetriever(email_identity=EMAIL)
    result = retriever.get_ncsr_by_cik(
        "0000036405",
        max_filings=1,
        enrich=False,
        verbose=True,
    )
    print_funds_summary(result, "TEST: get_ncsr_by_cik('0000036405')")
    assert len(result.funds) > 0, "Expected at least one fund from CIK 0000036405"
    print("✓ CIK test passed\n")
    return result


def test_by_accession():
    """Test fetching a single N-CSR filing by accession number."""
    retriever = FilingRetriever(email_identity=EMAIL)
    result = retriever.get_ncsr_by_accession(
        "0001104659-25-020270",
        enrich=False,
    )
    print_funds_summary(result, "TEST: get_ncsr_by_accession('0001104659-25-020270')")
    assert len(result.funds) > 0, "Expected at least one fund from accession number"
    print("✓ Accession number test passed\n")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  FilingRetriever Integration Tests")
    print("=" * 60)

    print("\n[1/2] Testing get_ncsr_by_cik ...")
    test_by_cik()

    print("\n[2/2] Testing get_ncsr_by_accession ...")
    test_by_accession()

    print("=" * 60)
    print("  All tests passed ✓")
    print("=" * 60)
