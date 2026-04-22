"""
EDGAR Filing Retriever
======================

Orchestrates downloading SEC filings from EDGAR and dispatching them to
the correct parser.  This is the single entry point for acquiring parsed
filing data — replacing the 40+ lines of boilerplate notebook code.

Usage::

    retriever = FilingRetriever(email_identity="you@example.com")

    # Fund filings (N-CSR)
    result = retriever.get_ncsr_filings("VOO", year="2025")

    # Portfolio holdings (NPORT)
    holdings = retriever.get_nport_holdings("VOO")

    # Summary prospectus (497K)
    prospectus = retriever.get_prospectus("VOO")

    # Company filings (10-K + DEF 14A + Form 4)
    companies = retriever.get_10k(["AAPL", "MSFT"])
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

import pandas as pd
from edgar import Company, Entity, set_identity, use_local_storage, find

from .ncsr_parser import NCSRExtractor
from .nport import NPortProcessor, process_portfolio_holdings
from .prospectus_parser import ProspectusExtractor
from .tenk_parser import TenKParser
from ..models.company import CompanyEntity
from ..models.fund import FundData, PortfolioHolding
from ..models.fund import FilingMetadata as FundFilingMetadata
from ..utils.fund_mapper import enrich_funds_with_annual_returns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class NCSRResult:
    """Result container for N-CSR filing extraction.

    Attributes:
        funds: List of :class:`FundData` objects extracted from the filing(s).
        financial_highlights: List of financial-highlights DataFrames, one
            per filing processed.
        filings_processed: Number of N-CSR filings that were parsed.
    """

    funds: List[FundData] = field(default_factory=list)
    financial_highlights: List[pd.DataFrame] = field(default_factory=list)
    filings_processed: int = 0


@dataclass
class ProspectusResult:
    """Result container for 497K summary-prospectus extraction.

    Attributes:
        structured_data: Dictionary of extracted fund metrics.
        markdown: Clean markdown profile suitable for RAG context.
        ticker: Resolved ticker symbol.
    """

    structured_data: Dict = field(default_factory=dict)
    markdown: str = ""
    ticker: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# FilingRetriever
# ---------------------------------------------------------------------------

class FilingRetriever:
    """Orchestrates downloading SEC filings from EDGAR and parsing them.

    This class provides a unified API for all supported filing types:

    * **N-CSR** — annual/semi-annual fund reports (via :class:`NCSRExtractor`)
    * **NPORT** — monthly portfolio holdings (via :class:`NPortProcessor`)
    * **497K** — summary prospectus (via :class:`ProspectusExtractor`)
    * **10-K** — annual company reports, plus DEF 14A and Form 4
      (via :class:`TenKParser`)

    Args:
        email_identity: E-mail address for SEC EDGAR fair-use identification.
        cache_dir: Directory to persist downloaded filings.
    """

    def __init__(
        self,
        email_identity: str,
        cache_dir: str = "./edgar_cache",
    ):
        set_identity(email_identity)
        use_local_storage(cache_dir)

        self._email_identity = email_identity
        self._cache_dir = cache_dir

    # ------------------------------------------------------------------
    # N-CSR
    # ------------------------------------------------------------------

    def get_ncsr_filings(
        self,
        ticker: str,
        *,
        year: Optional[str] = None,
        max_filings: Optional[int] = None,
        vanguard: bool = False,
        enrich: bool = True,
        verbose: bool = False,
    ) -> NCSRResult:
        """Download and parse N-CSR filings for a fund ticker.

        Iterates through the available N-CSR filings (optionally filtered by
        *year*), extracts fund data and financial highlights from each, and
        stops when a duplicate fund ticker is encountered (i.e. the same
        fund has already been seen in a prior filing).

        Args:
            ticker: Fund ticker symbol (e.g. ``"VOO"``).
            year: If provided, only filings whose ``report_date`` starts with
                this string are processed (e.g. ``"2025"``).
            max_filings: Maximum number of filings to process.  ``None``
                means process all available filings.
            vanguard: If ``True``, prepend *"Vanguard "* to fund names that
                do not already contain it.
            enrich: If ``True`` (default), automatically computes annual
                returns from performance tables and merges financial
                highlights into each fund.
            verbose: If ``True``, prints detailed processing info to stdout
                (fund names, context IDs, duplicate detection, counts).

        Returns:
            :class:`NCSRResult` containing all extracted funds and highlights.
        """
        company = Company(ticker)
        all_filings = company.get_filings(form="N-CSR")

        if not all_filings:
            logger.warning(f"[{ticker}] No N-CSR filings found")
            return NCSRResult()

        if verbose and year:
            year_filings = [
                f for f in all_filings
                if f.report_date and str(f.report_date).startswith(year)
            ]
            print(f"Found filings: {len(year_filings)} for year: {year}")

        return self._process_ncsr_filings(
            label=ticker,
            all_filings=all_filings,
            year=year,
            max_filings=max_filings,
            vanguard=vanguard,
            enrich=enrich,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # N-CSR — General-purpose access methods
    # ------------------------------------------------------------------

    @staticmethod
    def parse_ncsr(
        html_content: str,
        *,
        vanguard: bool = False,
    ) -> NCSRResult:
        """Parse raw N-CSR HTML content without downloading anything.

        This is the most flexible entry point — useful when you already
        have the filing HTML (e.g. from a local file, a database, or a
        custom download pipeline).

        Args:
            html_content: Raw HTML/iXBRL content of the N-CSR filing.
            vanguard: If ``True``, prepend *"Vanguard "* to fund names.

        Returns:
            :class:`NCSRResult` with funds and financial highlights from
            the single filing.
        """
        parser = NCSRExtractor(html_content)
        funds = parser.get_funds(vanguard=vanguard)
        highlights = parser.get_financial_highlights_vanguard() if vanguard else parser.get_financial_highlights_ishares()

        return NCSRResult(
            funds=funds,
            financial_highlights=[highlights],
            filings_processed=1,
        )

    def get_ncsr_by_cik(
        self,
        cik: str,
        *,
        year: Optional[str] = None,
        max_filings: Optional[int] = None,
        vanguard: bool = False,
        enrich: bool = True,
        verbose: bool = False,
    ) -> NCSRResult:
        """Download and parse N-CSR filings using a CIK number.

        Identical to :meth:`get_ncsr_filings` but accepts a SEC CIK
        instead of a ticker symbol.

        Args:
            cik: SEC Central Index Key (e.g. ``"0000102909"`` for Vanguard).
            year: Optional year filter (e.g. ``"2025"``).
            max_filings: Maximum number of filings to process.
            vanguard: Prepend *"Vanguard "* to fund names.
            enrich: Run the annual-returns + highlights enrichment pipeline.
            verbose: Print detailed processing info to stdout.

        Returns:
            :class:`NCSRResult` containing all extracted funds and highlights.
        """
        entity = Entity(cik)
        all_filings = entity.get_filings(form="N-CSR")

        if not all_filings:
            logger.warning(f"[CIK {cik}] No N-CSR filings found")
            return NCSRResult()

        return self._process_ncsr_filings(
            label=f"CIK {cik}",
            all_filings=all_filings,
            year=year,
            max_filings=max_filings,
            vanguard=vanguard,
            enrich=enrich,
            verbose=verbose,
        )

    def get_ncsr_by_cik_parallel(
        self,
        cik: str,
        *,
        year: Optional[str] = None,
        max_filings: Optional[int] = None,
        vanguard: bool = False,
        enrich: bool = True,
        max_workers: int = 4,
        verbose: bool = False,
    ) -> NCSRResult:
        """Download N-CSR filings sequentially, then parse them in parallel.

        Phase 1 (sequential): Downloads each filing's HTML on the main
        process so that EDGAR rate-limiting and cookies are respected.

        Phase 2 (parallel): Dispatches the already-downloaded HTML
        strings to a :class:`ProcessPoolExecutor` so CPU-bound
        BeautifulSoup / ``pd.read_html`` work runs concurrently.

        Duplicate-ticker detection and enrichment are applied after all
        workers finish, matching the behaviour of
        :meth:`get_ncsr_by_cik`.

        Args:
            cik: SEC Central Index Key (e.g. ``"0000102909"`` for Vanguard).
            year: Optional year filter (e.g. ``"2025"``).
            max_filings: Cap the number of filings to download/parse.
            vanguard: Prepend *"Vanguard "* to fund names.
            enrich: Run the annual-returns + highlights enrichment pipeline.
            max_workers: Number of parallel worker processes (default 4).
            verbose: Print progress info to stdout.

        Returns:
            :class:`NCSRResult` containing all extracted funds and highlights.
        """
        entity = Entity(cik)
        all_filings = entity.get_filings(form="N-CSR")

        if not all_filings:
            logger.warning(f"[CIK {cik}] No N-CSR filings found")
            return NCSRResult()

        if year:
            filings_to_process = [
                f for f in all_filings
                if f.report_date and str(f.report_date).startswith(year)
            ]
        else:
            filings_to_process = list(all_filings)

        if max_filings:
            filings_to_process = filings_to_process[:max_filings]

        # ── Phase 1: sequential HTML download ────────────────────────────
        filing_data_list: List[tuple] = []
        for filing in filings_to_process:
            if verbose:
                print(f"Downloading: {filing.report_date}")
            try:
                html_content = filing.html()
                if html_content:
                    filing_data_list.append((
                        html_content,
                        filing.report_date,
                        filing.accession_number,
                        filing.filing_date,
                        filing.form,
                        filing.url,
                    ))
            except Exception as e:
                logger.warning(f"[CIK {cik}] Failed to download {filing.report_date}: {e}")

        if verbose:
            print(f"Downloaded {len(filing_data_list)} filings — starting parallel parse with {max_workers} workers")

        # ── Phase 2: parallel parse ───────────────────────────────────────
        worker_fn = partial(NCSRExtractor.process_filing_data, vanguard=vanguard)

        raw_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_fn, fd): fd[1] for fd in filing_data_list}
            for future in as_completed(futures):
                report_date = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        raw_results.append(res)
                except Exception as e:
                    logger.warning(f"[CIK {cik}] Worker failed for {report_date}: {e}")

        # Sort by report_date descending (newest first) to match sequential behaviour
        raw_results.sort(key=lambda r: str(r["report_date"]), reverse=True)

        # ── Duplicate-ticker detection (same logic as _process_ncsr_filings) ─
        result = NCSRResult()
        seen_tickers: set = set()

        for res in raw_results:
            funds_to_add = []
            duplicate_found = False

            for fund in res["funds"]:
                if fund.ticker in seen_tickers:
                    if verbose:
                        print(f"Duplicate ticker '{fund.ticker}' — skipping filing {res['report_date']}")
                    duplicate_found = True
                    break
                seen_tickers.add(fund.ticker)
                funds_to_add.append(fund)

            if not duplicate_found:
                result.funds.extend(funds_to_add)
                if res["df_performance"] is not None:
                    result.financial_highlights.append(res["df_performance"])
                result.filings_processed += 1

        if verbose:
            print(f"Extracted {len(result.funds)} unique funds from {result.filings_processed} filings")

        if enrich and result.funds:
            enrich_funds_with_annual_returns(
                funds=result.funds,
                financial_highlights_dfs=result.financial_highlights,
                debug=False,
            )
            logger.info(f"[CIK {cik}] Enrichment complete")

        return result

    def get_ncsr_by_accession(
        self,
        accession_number: str,
        *,
        vanguard: bool = False,
        enrich: bool = True,
    ) -> NCSRResult:
        """Download and parse a single N-CSR filing by accession number.

        Args:
            accession_number: SEC accession number
                (e.g. ``"0001193125-24-123456"``).
            vanguard: Prepend *"Vanguard "* to fund names.
            enrich: Run the annual-returns + highlights enrichment pipeline.

        Returns:
            :class:`NCSRResult` for the single filing.
        """
        filing = find(accession_number)

        if not filing:
            logger.warning(f"Filing not found: {accession_number}")
            return NCSRResult()

        logger.info(f"Processing filing: {accession_number}")
        html_content = filing.html()
        result = self.parse_ncsr(html_content, vanguard=vanguard)

        # Attach metadata to each fund (using getattr safely for single Filing objects)
        metadata = FundFilingMetadata(
            accession_number=getattr(filing, "accession_no", getattr(filing, "accession_number", "N/A")),
            reporting_date=getattr(filing, "report_date", getattr(filing, "period_of_report", None)),
            filing_date=getattr(filing, "filing_date", "N/A"),
            form=getattr(filing, "form", "N-CSR"),
            url=getattr(filing, "head_url", getattr(filing, "url", "N/A")),
        )
        for fund in result.funds:
            fund.ncsr_metadata = metadata

        if enrich and result.funds:
            enrich_funds_with_annual_returns(
                funds=result.funds,
                financial_highlights_dfs=result.financial_highlights,
            )

        return result

    # ------------------------------------------------------------------
    # N-CSR — Shared processing logic
    # ------------------------------------------------------------------

    def _process_ncsr_filings(
        self,
        label: str,
        all_filings,
        *,
        year: Optional[str] = None,
        max_filings: Optional[int] = None,
        vanguard: bool = False,
        enrich: bool = True,
        verbose: bool = False,
    ) -> NCSRResult:
        """Internal helper that drives the N-CSR download→parse loop.

        Both :meth:`get_ncsr_filings` and :meth:`get_ncsr_by_cik` delegate
        here so the duplicate-detection and enrichment logic lives in one
        place.
        """
        if year:
            filings_to_process = [
                f for f in all_filings
                if f.report_date and str(f.report_date).startswith(year)
            ]
            logger.info(
                f"[{label}] Found {len(filings_to_process)} N-CSR filings "
                f"for year {year} (out of {len(all_filings)} total)"
            )
        else:
            filings_to_process = list(all_filings)

        if max_filings:
            filings_to_process = filings_to_process[:max_filings]

        result = NCSRResult()
        seen_tickers: set = set()
        performance_count = 0
        abort = False

        for filing in filings_to_process:
            if abort:
                break

            logger.info(f"[{label}] Processing N-CSR filing: {filing.report_date}")
            if verbose:
                print(f"Processing filing:  {filing.report_date}")

            html_content = filing.html()
            metadata = FundFilingMetadata(
                accession_number=filing.accession_number,
                reporting_date=filing.report_date,
                filing_date=filing.filing_date,
                form=filing.form,
                url=filing.url,
            )

            parser = NCSRExtractor(html_content)
            funds = parser.get_funds(vanguard=vanguard)
            filing_perf_count = 0

            for fund in funds:
                fund.ncsr_metadata = metadata

                if verbose:
                    print(f"Processing: {fund.name} {fund.share_class.value}")
                    print(f"Extracting context: {fund.context_id}")

                if fund.performance_table is not None:
                    filing_perf_count += 1

                if fund.ticker in seen_tickers:
                    logger.info(
                        f"[{label}] Duplicate fund ticker '{fund.ticker}' "
                        f"— stopping iteration"
                    )
                    if verbose:
                        print(f"Fund already exists:  {fund.ticker} aborting")
                    abort = True
                    break
                seen_tickers.add(fund.ticker)

            if not abort:
                highlights_df = parser.get_financial_highlights_vanguard() if vanguard else parser.get_financial_highlights_ishares()
                result.financial_highlights.append(highlights_df)
                result.funds.extend(funds)
                result.filings_processed += 1
                performance_count += filing_perf_count

                logger.info(
                    f"[{label}] Filing {filing.filing_date}: "
                    f"{len(funds)} funds extracted"
                )
                
        if verbose:
            print(f"Number of funds with performance chart: {performance_count}")
            print(f"Funds extracted ({len(seen_tickers)}):  {seen_tickers}")

        logger.info(
            f"[{label}] N-CSR complete: {len(result.funds)} funds from "
            f"{result.filings_processed} filing(s)"
        )

        if enrich and result.funds:
            enrich_funds_with_annual_returns(
                funds=result.funds,
                financial_highlights_dfs=result.financial_highlights,
                debug=False,
            )
            logger.info(f"[{label}] Enrichment complete")

        return result

    @staticmethod
    def verify_ncsr_data(result: NCSRResult) -> Dict[str, List[str]]:
        """Verify the data quality of extracted N-CSR filings.
        
        Checks each fund for missing values and ensures numeric values 
        make logical sense (e.g. valid expense ratios, net assets > 0).

        Args:
            result: The :class:`NCSRResult` object to verify.
        
        Returns:
            A dictionary mapping fund tickers to a list of validation warnings.
            If a fund has no warnings, it will not appear in the dictionary.
        """
        warnings = {}

        if not result.funds:
            return {"GENERAL": ["No funds found in the result."]}

        for fund in result.funds:
            fund_warnings = []

            # 1. Check basic identifiers
            if not fund.ticker or fund.ticker == "N/A":
                fund_warnings.append("Missing or empty ticker.")
            if not fund.context_id:
                fund_warnings.append("Missing context ID.")
            if not fund.share_class:
                fund_warnings.append("Missing share class.")

            # 2. Check numeric fields for logical sense
            try:
                expense = float(fund.expense_ratio) if fund.expense_ratio and fund.expense_ratio != "N/A" else -1
                if expense < 0:
                    fund_warnings.append("Expense ratio is missing or negative.")
                elif expense > 5.0:
                    fund_warnings.append(f"Expense ratio seems unusually high: {expense}%")
            except ValueError:
                fund_warnings.append(f"Invalid expense ratio format: {fund.expense_ratio}")

            try:
                assets = float(fund.net_assets) if fund.net_assets and fund.net_assets != "N/A" else -1
                if assets <= 0:
                    fund_warnings.append("Net assets missing or zero.")
            except ValueError:
                fund_warnings.append(f"Invalid net assets format: {fund.net_assets}")

            try:
                turnover = float(fund.turnover_rate) if fund.turnover_rate and fund.turnover_rate != "N/A" else -1
                if turnover < 0:
                    fund_warnings.append("Turnover rate is missing or negative.")
                # turnover rates can technically be high, but negative is definitely wrong.
            except ValueError:
                fund_warnings.append(f"Invalid turnover format: {fund.turnover_rate}")

            try:
                holdings = int(float(fund.n_holdings)) if fund.n_holdings and fund.n_holdings != "N/A" else -1
                if holdings <= 0:
                    fund_warnings.append("Number of holdings is missing or zero.")
            except ValueError:
                fund_warnings.append(f"Invalid number of holdings format: {fund.n_holdings}")

            # Store if there are any warnings
            if fund_warnings:
                warnings[fund.ticker or fund.name] = fund_warnings

        return warnings

    # ------------------------------------------------------------------
    # NPORT
    # ------------------------------------------------------------------

    def get_nport_holdings(
        self,
        ticker: str,
        *,
        company_tickers_json: Optional[str] = None,
    ) -> List[PortfolioHolding]:
        """Download and parse the latest NPORT filing for a fund ticker.

        Args:
            ticker: Fund ticker symbol (e.g. ``"VOO"``).
            company_tickers_json: Optional path to a company-tickers JSON
                file for ticker enrichment.  If provided, holdings will be
                enriched with matched tickers via fuzzy matching.

        Returns:
            List of :class:`PortfolioHolding` objects.
        """
        company = Company(ticker)
        nport_filings = company.get_filings(form="NPORT-P")

        if not nport_filings:
            logger.warning(f"[{ticker}] No NPORT filings found")
            return []

        latest = nport_filings.latest()
        logger.info(f"[{ticker}] Processing NPORT filing: {latest.filing_date}")

        nport_obj = latest.obj()
        investments = nport_obj.investments if hasattr(nport_obj, "investments") else []

        holdings = process_portfolio_holdings(investments)

        if company_tickers_json:
            processor = NPortProcessor(company_tickers_json_path=company_tickers_json)
            processor.enrich_tickers(holdings, verbose=True)

        logger.info(f"[{ticker}] NPORT complete: {len(holdings)} holdings")
        return holdings

    # ------------------------------------------------------------------
    # 497K (Summary Prospectus)
    # ------------------------------------------------------------------

    def get_prospectus(
        self,
        ticker: str,
    ) -> ProspectusResult:
        """Download and parse the latest 497K summary prospectus.

        Args:
            ticker: Fund ticker symbol (e.g. ``"VOO"``).

        Returns:
            :class:`ProspectusResult` with structured data and markdown.
        """
        company = Company(ticker)
        prospectus_filings = company.get_filings(form="497K")

        if not prospectus_filings:
            logger.warning(f"[{ticker}] No 497K filings found")
            return ProspectusResult()

        latest = prospectus_filings.latest()
        logger.info(f"[{ticker}] Processing 497K filing: {latest.filing_date}")

        # 497K filings are plain text
        raw_text = latest.text() if hasattr(latest, "text") else str(latest.obj())

        parser = ProspectusExtractor(raw_text, ticker=ticker)

        return ProspectusResult(
            structured_data=parser.get_structured_data(),
            markdown=parser.get_clean_markdown(),
            ticker=parser.get_ticker(),
        )

    # ------------------------------------------------------------------
    # 10-K (+ DEF 14A + Form 4)
    # ------------------------------------------------------------------

    def get_10k(
        self,
        tickers: List[str],
        *,
        max_form4_date: str = "2025-09-01:",
        max_workers: int = 8,
    ) -> List[CompanyEntity]:
        """Download and parse 10-K filings for one or more company tickers.

        Delegates to :class:`TenKParser`, which also fetches DEF 14A
        (executive compensation) and Form 4 (insider transactions) filings.

        Args:
            tickers: List of company ticker symbols (e.g. ``["AAPL", "MSFT"]``).
            max_form4_date: EDGAR date filter for Form 4 filings
                (e.g. ``"2025-01-01:"``).
            max_workers: Number of parallel workers for Form 4 batch processing.

        Returns:
            List of fully populated :class:`CompanyEntity` objects.
        """
        parser = TenKParser(
            email_identity=self._email_identity,
            cache_dir=self._cache_dir,
            max_workers=max_workers,
        )
        return parser.process_companies(tickers, max_form4_date=max_form4_date)
