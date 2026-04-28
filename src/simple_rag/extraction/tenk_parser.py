import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from typing import List, Dict, Optional, Any

import pandas as pd
from tqdm import tqdm
from edgar import set_identity, Company, use_local_storage

from .form4_parser import Form4Parser
from .def14a_parser import Def14AParser
from ..models.company import (
    CompanyEntity,
    FilingMetadata,
    Filing10K,
    IncomeStatement,
    FinancialMetric,
    FinancialSegment,
)
from ..utils.cache_manager import is_cached, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XBRL Concept → IncomeStatement field mapping
# ---------------------------------------------------------------------------
# Maps XBRL concept tags (with prefix) to IncomeStatement Pydantic properties.

CONCEPT_TO_PROPERTY: Dict[str, str] = {
    # --- REVENUE (Top Line) ---
    "us-gaap_Revenues": "revenue",
    "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",  # Apple, Nvidia standard
    "us-gaap_SalesRevenueNet": "revenue",
    "us-gaap_RevenueFromContractWithCustomer": "revenue",
    # For banks: only map the net number (Total Net Revenue)
    "us-gaap_RevenuesNetOfInterestExpense": "revenue",  # WFC: $82B — correct top-line for banks

    # --- COST OF SALES ---
    "us-gaap_CostOfRevenue": "cost_of_sales",
    "us-gaap_CostOfGoodsAndServicesSold": "cost_of_sales",
    "us-gaap_CostOfGoodsSold": "cost_of_sales",
    # Bank-specific: provision for loan losses
    "us-gaap_ProvisionForLoanLeaseAndOtherLosses": "cost_of_sales",  # WFC: $4.3B

    # --- GROSS PROFIT ---
    "us-gaap_GrossProfit": "gross_profit",
    # For banks: Net Interest Income is their "Gross Profit"
    "us-gaap_InterestIncomeExpenseNet": "gross_profit",  # WFC: $47B

    # --- OPERATING EXPENSES ---
    "us-gaap_OperatingExpenses": "operating_expenses",
    "us-gaap_NoninterestExpense": "operating_expenses",  # WFC: $54B — bank total non-interest expenses

    # --- OPERATING INCOME ---
    "us-gaap_OperatingIncomeLoss": "operating_income",

    # --- OTHER INCOME / EXPENSE ---
    "us-gaap_NonoperatingIncomeExpense": "other_income_expense",

    # --- PRE-TAX INCOME ---
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "pretax_income",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments": "pretax_income",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxes": "pretax_income",

    # --- INCOME TAXES ---
    "us-gaap_IncomeTaxExpenseBenefit": "provision_for_income_taxes",
    "us-gaap_IncomeTaxesPaid": "provision_for_income_taxes",

    # --- NET INCOME (Bottom Line) ---
    "us-gaap_NetIncomeLoss": "net_income",
    "us-gaap_ProfitLoss": "net_income",
    "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic": "net_income",

    # --- EPS ---
    "us-gaap_EarningsPerShareBasic": "basic_earnings_per_share",
    "us-gaap_EarningsPerShareDiluted": "diluted_earnings_per_share",

    # --- SHARES OUTSTANDING ---
    "us-gaap_WeightedAverageNumberOfSharesOutstandingBasic": "basic_shares_outstanding",
    "us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding": "diluted_shares_outstanding",
}

# Definitive totals that should overwrite any previously set partial value.
PRIORITY_CONCEPTS = {
    "us-gaap_RevenuesNetOfInterestExpense",  # Bank Total Revenue
    "us-gaap_NetIncomeLoss",                 # Real bottom line
    "us-gaap_OperatingExpenses",             # Real total operating expenses
    "us-gaap_NoninterestExpense",            # Bank total operating expenses
    "us-gaap_Revenues",                      # Standard total revenue
    "us-gaap_GrossProfit",                   # Total gross profit
}


# ---------------------------------------------------------------------------
# Module-level helper for multiprocessing
# ---------------------------------------------------------------------------

def _process_insider_batch(batch_data: List[Any]) -> List[Any]:
    """
    Module-level helper required by ``ProcessPoolExecutor`` (must be picklable).
    Delegates to :class:`Form4Parser` so that worker processes remain
    stateless and lightweight.
    """
    try:
        parser = Form4Parser()
        return parser.extract_insider_transactions_batch(batch_data)
    except Exception as e:
        logger.error(f"Batch error: {e}")
        return []


# ---------------------------------------------------------------------------
# TenKParser
# ---------------------------------------------------------------------------

class TenKParser:
    """
    Parser for SEC Form 10-K (Annual Report) filings.

    Orchestrates the download and extraction of three related filing types:

    * **10-K** — income statement, balance sheet, cash flow, business sections.
    * **DEF 14A** — executive compensation (Pay-vs-Performance table).
    * **Form 4** — insider transactions (open-market buys and sells).

    Income statement parsing is handled internally via XBRL concept mapping
    (see :data:`CONCEPT_TO_PROPERTY`).  The two supplementary filing types
    delegate to :class:`Def14AParser` and :class:`Form4Parser` respectively.

    Typical usage::

        parser = TenKParser(email_identity="you@example.com")
        companies = parser.process_companies(["AAPL", "MSFT"])
    """

    def __init__(
        self,
        email_identity: str,
        cache_dir: str = "./edgar_cache",
        max_workers: int = 8,
    ):
        """
        Args:
            email_identity: SEC EDGAR identity e-mail (required by EDGAR fair-use policy).
            cache_dir: Directory used to persist downloaded filings and avoid
                hitting EDGAR rate limits on repeated runs.
            max_workers: Number of parallel worker processes used when
                processing Form 4 insider-transaction batches.
        """
        set_identity(email_identity)
        use_local_storage(cache_dir)

        self._form4_parser = Form4Parser()
        self._def14a_parser = Def14AParser()
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_companies(
        self,
        tickers: List[str],
        max_form4_date: str = "2025-09-01:",
    ) -> List[CompanyEntity]:
        """
        Download and parse filings for a list of ticker symbols.

        For each ticker the method fetches:

        1. The most recent **10-K** and extracts the income statement plus
           all narrative sections (Item 1, 1A, 2, 3, 7).
        2. The most recent **DEF 14A** for executive compensation data.
        3. All **Form 4** filings filed after *max_form4_date* for insider
           transaction data, processed in parallel batches.

        Args:
            tickers: List of NYSE/NASDAQ ticker symbols to process.
            max_form4_date: EDGAR date filter string (e.g. ``"2025-01-01:"``).

        Returns:
            List of fully populated :class:`CompanyEntity` objects.
        """
        companies: Dict[str, CompanyEntity] = {}

        for ticker in tqdm(tickers, desc="Processing companies"):
            try:
                company = Company(ticker)

                # --- 10-K ---
                tenk_filing = company.get_filings(form="10-K").latest()
                if is_cached(ticker, "10-K", tenk_filing.accession_number):
                    tqdm.write(f"[{ticker}] Loading 10-K from cache...")
                    tenk = load_from_cache(ticker, "10-K", tenk_filing.accession_number)
                else:
                    tqdm.write(f"[{ticker}] Downloading 10-K...")
                    tenk = tenk_filing.obj()
                    save_to_cache(ticker, "10-K", tenk_filing.accession_number, tenk)

                # --- DEF 14A ---
                exec_comp_filings = company.get_filings(form="DEF 14A")
                exec_comp_filing = exec_comp_filings.latest() if exec_comp_filings else None

                if exec_comp_filing:
                    if is_cached(ticker, "DEF14A", exec_comp_filing.accession_number):
                        tqdm.write(f"[{ticker}] Loading DEF 14A from cache...")
                        exec_comp_obj = load_from_cache(ticker, "DEF14A", exec_comp_filing.accession_number)
                    else:
                        tqdm.write(f"[{ticker}] Downloading DEF 14A...")
                        exec_comp_obj = exec_comp_filing.obj()
                        save_to_cache(ticker, "DEF14A", exec_comp_filing.accession_number, exec_comp_obj)
                else:
                    exec_comp_obj = None

                # --- Form 4s ---
                ins_filings = company.get_filings(form="4").filter(date=max_form4_date)

                # ── Build CompanyEntity ──────────────────────────────────
                tqdm.write(
                    f"[{ticker}] Period of report: {tenk_filing.period_of_report} | Company: {tenk.company}"
                )

                metadata = FilingMetadata(
                    accession_number=tenk_filing.accession_no,
                    filing_type=tenk_filing.form,
                    filing_date=tenk_filing.filing_date,
                    report_period_end=tenk_filing.period_of_report,
                    filing_url=tenk_filing.url,
                    cik=str(tenk_filing.cik),
                )

                companies[ticker] = CompanyEntity(
                    name=tenk.company,
                    cik=str(tenk_filing.cik),
                    ticker=ticker,
                )

                # ── Income Statement ─────────────────────────────────────
                income_statement = tenk.income_statement
                df = income_statement.to_dataframe()
                statement_dict = self.extract_income_statement_dict(df)
                self.process_income_statement_dict(
                    companies[ticker], statement_dict, metadata, str(income_statement)
                )

                # ── 10-K Narrative Sections ──────────────────────────────
                filing_10k = companies[ticker].filings_10k[metadata.filing_date]
                filing_10k.business_information = tenk["Item 1"]
                filing_10k.risk_factors = tenk["Item 1A"]
                filing_10k.balance_sheet_text = str(tenk.balance_sheet)
                filing_10k.cash_flow_text = str(tenk.cash_flow_statement)
                filing_10k.management_discussion_and_analysis = tenk["Item 7"]
                filing_10k.legal_proceedings = tenk["Item 3"]
                filing_10k.properties = tenk["Item 2"]

                # ── DEF 14A — Executive Compensation ─────────────────────
                if exec_comp_obj:
                    tqdm.write(f"[{ticker}] Exec comp: {exec_comp_obj}")
                    self._def14a_parser.extract_executive_compensation(
                        companies[ticker], exec_comp_obj, metadata, exec_comp_filing.url
                    )

                # ── Form 4 — Insider Transactions ────────────────────────
                tqdm.write(f"[{ticker}] Found {len(ins_filings)} insider filings")

                insider_args = []
                cached_count = 0
                downloaded_count = 0

                for ins_filing in ins_filings:
                    try:
                        if is_cached(ticker, "Form4", ins_filing.accession_number):
                            ins_obj = load_from_cache(ticker, "Form4", ins_filing.accession_number)
                            cached_count += 1
                        else:
                            ins_obj = ins_filing.obj()
                            save_to_cache(ticker, "Form4", ins_filing.accession_number, ins_obj)
                            downloaded_count += 1

                        summary = ins_obj.get_ownership_summary()
                        insider_args.append((summary, ins_filing.url))
                    except Exception as e:
                        logger.error(f"[{ticker}] Error processing insider filing: {e}")

                if cached_count > 0:
                    tqdm.write(f"[{ticker}] Loaded {cached_count} Form 4s from cache")
                if downloaded_count > 0:
                    tqdm.write(f"[{ticker}] Downloaded {downloaded_count} Form 4s")

                if insider_args:
                    batch_size = 10
                    batches = [
                        insider_args[i : i + batch_size]
                        for i in range(0, len(insider_args), batch_size)
                    ]
                    tqdm.write(
                        f"[{ticker}] Processing {len(batches)} batches "
                        f"with up to {batch_size} filings each"
                    )

                    with ProcessPoolExecutor(max_workers=self.max_workers) as insider_executor:
                        futures = {
                            insider_executor.submit(_process_insider_batch, batch): batch
                            for batch in batches
                        }
                        for future in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"[{ticker}] Insider batches",
                            leave=False,
                            unit="batch",
                        ):
                            transactions = future.result()
                            if transactions:
                                companies[ticker].insider_trades.extend(transactions)

                    tqdm.write(
                        f"[{ticker}] ✓ Processed "
                        f"{len(companies[ticker].insider_trades)} insider transactions\n"
                    )

            except Exception as e:
                tqdm.write(f"[{ticker}] ❌ Error processing company: {e}")
                traceback.print_exc()

        return list(companies.values())

    # ------------------------------------------------------------------
    # Income Statement Extraction (10-K / XBRL)
    # ------------------------------------------------------------------

    def extract_income_statement_dict(self, df: pd.DataFrame) -> Dict:
        """
        Transform an income-statement :class:`~pandas.DataFrame` (as returned
        by ``edgar``'s ``income_statement.to_dataframe()``) into a structured
        dictionary ready for :meth:`process_income_statement_dict`.

        The output groups line items by concept and attaches per-period values
        and any dimensional segments (e.g. geographic or product breakdowns).

        Args:
            df: Raw income-statement DataFrame containing at minimum a
                ``concept`` column and one or more date columns
                (``YYYY-MM-DD`` format).

        Returns:
            ``{"periods": [...], "line_items": [...]}`` dict, or
            ``{"error": "..."}`` if the DataFrame is missing the
            ``concept`` column.
        """
        date_cols = sorted(
            [col for col in df.columns if __import__("re").match(r"\d{4}-\d{2}-\d{2}", col)],
            reverse=True,
        )

        if "concept" not in df.columns:
            return {"error": "DataFrame missing 'concept' column"}

        final_items = []
        grouped = df.groupby("concept", sort=False)

        for concept, group in grouped:
            parent_rows = group[
                (group["dimension"] == False) | (group["dimension"].isna())
            ]
            if parent_rows.empty:
                continue

            parent = parent_rows.iloc[0]

            if parent.get("abstract") is True:
                logger.debug(f"Skipping abstract concept: {concept} - '{parent.get('label')}'")
                continue

            item = {
                "concept": str(concept),
                "label": str(parent["label"]),
                "values": {},
                "segments": {},
            }

            for d in date_cols:
                val = parent.get(d)
                if pd.notna(val):
                    item["values"][d] = float(val)

            segment_rows = group[group["dimension"] == True]
            for _, seg_row in segment_rows.iterrows():
                seg_obj: Dict = {"label": str(seg_row["label"]), "values": {}}
                has_data = False
                for d in date_cols:
                    val = seg_row.get(d)
                    if pd.notna(val):
                        seg_obj["values"][d] = float(val)
                        has_data = True

                if has_data:
                    import re as _re
                    axis_name = seg_row.get("dimension_axis")
                    if pd.isna(axis_name) or axis_name is None:
                        dim_label = seg_row.get("dimension_label")
                        if pd.notna(dim_label) and dim_label:
                            parts = str(dim_label).split(":")
                            if len(parts) >= 2:
                                axis_candidates = [
                                    p.strip() for p in str(dim_label).split(",")
                                ]
                                for candidate in reversed(axis_candidates):
                                    if ":" in candidate:
                                        axis_name = candidate.split(":")[0].strip()
                                        break
                            else:
                                axis_name = parts[0].strip()
                        else:
                            axis_name = "Other"

                    axis_name = str(axis_name)
                    item["segments"].setdefault(axis_name, []).append(seg_obj)

            final_items.append(item)

        return {"periods": date_cols, "line_items": final_items}

    def process_income_statement_dict(
        self,
        company: CompanyEntity,
        income_statement_dict: Dict,
        filing_metadata: FilingMetadata,
        income_statement: str = None,
    ) -> None:
        """
        Populate a company's :class:`Filing10K` with
        :class:`IncomeStatement` objects derived from the structured dict
        produced by :meth:`extract_income_statement_dict`.

        One :class:`IncomeStatement` is created per reporting period (date)
        found in the dict.  Concepts are mapped to model fields via
        :data:`CONCEPT_TO_PROPERTY`; high-priority concepts in
        :data:`PRIORITY_CONCEPTS` overwrite any previously set value for
        the same field.

        Args:
            company: :class:`CompanyEntity` to update in-place.
            income_statement_dict: Structured dict from
                :meth:`extract_income_statement_dict`.
            filing_metadata: Metadata for the parent 10-K filing — used to
                locate or create the :class:`Filing10K` record.
            income_statement: Raw text of the income statement (stored for
                RAG context).
        """
        periods = income_statement_dict["periods"]
        line_items = income_statement_dict["line_items"]

        filing_date = filing_metadata.filing_date
        if filing_date not in company.filings_10k:
            company.filings_10k[filing_date] = Filing10K(
                filing_metadata=filing_metadata,
                income_statement_text=income_statement,
            )

        filing_10k = company.filings_10k[filing_date]

        for period_date_str in periods:
            period_date = date.fromisoformat(period_date_str)
            income_stmt = IncomeStatement(
                period_end_date=period_date,
                fiscal_year=period_date.year,
            )
            mapped_count = 0
            skipped_count = 0

            for line_item in line_items:
                concept = line_item["concept"]
                label = line_item["label"]
                logger.debug(f"{concept} Label: {label}")

                property_name = CONCEPT_TO_PROPERTY.get(concept)
                if not property_name:
                    skipped_count += 1
                    logger.debug(f"  ⊘ SKIPPED (unmapped): {concept} - '{label}'")
                    continue

                value = line_item["values"].get(period_date_str)
                if value is None:
                    logger.debug(f"  ⊘ SKIPPED (no value): {concept} → {property_name}")
                    continue

                # EPS fields are plain floats, not FinancialMetric objects
                if property_name in ("basic_earnings_per_share", "diluted_earnings_per_share"):
                    setattr(income_stmt, property_name, float(value))
                    mapped_count += 1
                    logger.debug(
                        f"  ✓ MAPPED (EPS): {concept} → {property_name} = ${value:.2f}"
                    )
                    continue

                # Build segment list for this line item
                segments = []
                segment_count = 0
                for axis_name, segment_list in line_item.get("segments", {}).items():
                    for segment_data in segment_list:
                        segment_value = segment_data["values"].get(period_date_str)
                        if segment_value is not None:
                            segments.append(
                                FinancialSegment(
                                    label=segment_data["label"],
                                    amount=float(segment_value),
                                    axis=axis_name,
                                )
                            )
                            segment_count += 1

                existing_metric = getattr(income_stmt, property_name, None)
                should_set = (
                    existing_metric is None or existing_metric.value is None
                ) or (concept in PRIORITY_CONCEPTS)

                if should_set:
                    metric = FinancialMetric(
                        value=float(value), label=label, segments=segments
                    )
                    setattr(income_stmt, property_name, metric)
                    mapped_count += 1
                    priority_flag = " [PRIORITY]" if concept in PRIORITY_CONCEPTS else ""
                    segment_info = f" ({segment_count} segments)" if segment_count > 0 else ""
                    logger.debug(
                        f"  ✓ MAPPED{priority_flag}: {concept} → {property_name} "
                        f"= ${value:,.2f}{segment_info}"
                    )
                else:
                    logger.debug(
                        f"  ⊘ SKIPPED (already set): {concept} → {property_name} "
                        f"(existing: ${existing_metric.value:,.2f})"
                    )

            filing_10k.income_statements[period_date] = income_stmt
            logger.debug(
                f"\n  Summary for {period_date_str}:\n"
                f"    - Mapped: {mapped_count} line items\n"
                f"    - Skipped: {skipped_count} line items"
            )

        logger.info(
            f"COMPLETED: {len(periods)} income statement(s) added to {company.name}"
        )
