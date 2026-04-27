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
from ..models.fund import FundData, PortfolioHolding, NonDerivatives, Derivatives
from ..models.fund import FilingMetadata as FundFilingMetadata
from ..utils.fund_mapper import enrich_funds_with_annual_returns

logger = logging.getLogger(__name__)

def _process_nport_filing_worker(filing, ticker: str, company_json_path: str, email_identity: str, min_similarity: float = 0.74) -> Optional[dict]:
    """Parse a single NPORT-P filing in a worker process.

    Args:
        filing: EDGAR filing object.
        ticker: Seed ticker symbol used to fetch filings.
        company_json_path: Path to company_tickers.json for ticker enrichment.
        email_identity: EDGAR identity string.
        min_similarity: Minimum fuzzy-match similarity for ticker enrichment.

    Returns:
        Result dict or ``None`` on failure.
    """
    from edgar import set_identity
    set_identity(email_identity)

    try:
        import gc
        from .nport import NPortProcessor
        from ..models.fund import FilingMetadata as FundFilingMetadata

        xml_data = filing.obj()
        fund_series = xml_data.get_fund_series()
        fund_name = fund_series.name
        series_id = fund_series.series_id
        reporting_period = xml_data.reporting_period
        investments = xml_data.investments
        derivatives = getattr(xml_data, "derivatives", None)

        proc = NPortProcessor(company_tickers_json_path=company_json_path, min_similarity=min_similarity)
        holdings = proc.process_holdings(investments)
        result_df = proc.enrich_tickers(holdings, verbose=False)
        holdings_df = proc.to_df(holdings)

        filing_metadata = FundFilingMetadata(
            accession_number=filing.accession_number,
            reporting_date=reporting_period,
            filing_date=getattr(filing, "filing_date", None),
            form=getattr(filing, "form", "NPORT-P"),
            url=getattr(filing, "url", ""),
        )

        not_matches = result_df[result_df["matched_ticker"].isna() | (result_df["matched_ticker"] == "")]

        del xml_data, proc
        gc.collect()

        return {
            "fund_name": fund_name,
            "series_id": series_id,
            "reporting_period": reporting_period,
            "holdings": holdings,
            "holdings_df": holdings_df,
            "result": result_df,
            "derivatives": derivatives,
            "not_matches": not_matches,
            "ticker": ticker,
            "report_date": reporting_period,
            "nport_metadata": filing_metadata,
        }
    except Exception as e:
        logger.error("_process_nport_filing_worker failed for %s / %s: %s", ticker, getattr(filing, "accession_number", "?"), e)
        return None


def _parse_prospectus_worker(filing_data: tuple, email_identity: str) -> tuple:
    """Parse a single 497K filing in a worker process.

    Args:
        filing_data: ``(raw_text, accession_number, report_date,
                        filing_date, form, url)``
        email_identity: EDGAR identity string for rate-limit compliance.

    Returns:
        ``(ticker, ProspectusResult | None, status)`` where *status* is one
        of ``"success"``, ``"aborted"``, ``"failed_no_ticker"``,
        ``"failed_extraction"``, or ``"error"``.
    """
    try:
        from edgar import set_identity
        set_identity(email_identity)

        raw_text, accession_number, report_date, filing_date, form, url = filing_data

        from .prospectus_parser import ProspectusExtractor
        from ..models.fund import FilingMetadata as FundFilingMetadata

        parser = ProspectusExtractor.from_text(raw_text)
        extracted_ticker = parser.get_ticker()

        if not extracted_ticker:
            return None, None, "failed_no_ticker"

        extracted_ticker = str(extracted_ticker).strip().upper()

        fund_data = parser.get_structured_data()
        if not fund_data.get("objective") and not fund_data.get("strategies"):
            return extracted_ticker, None, "failed_extraction"

        try:
            import pandas as pd
            def _safe_date(v):
                if v is None:
                    return None
                try:
                    dt = pd.to_datetime(v)
                    return dt.date() if not pd.isna(dt) else None
                except Exception:
                    return None

            metadata = FundFilingMetadata(
                accession_number=accession_number,
                reporting_date=_safe_date(report_date),
                filing_date=_safe_date(filing_date),
                form=form,
                url=url,
            )
        except Exception:
            metadata = None

        result = dict(
            structured_data=fund_data,
            markdown=parser.get_clean_markdown(),
            ticker=extracted_ticker,
            managers=fund_data.get("managers"),
            strategies=fund_data.get("strategies"),
            risks=fund_data.get("risks"),
            objective=fund_data.get("objective"),
            filing_metadata=metadata,
        )

        return extracted_ticker, result, "success"

    except Exception as e:
        logger.error("_parse_prospectus_worker failed for %s: %s", filing_data[1] if filing_data else "?", e)
        return None, None, "error"


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
        managers: Fund manager information extracted from the prospectus.
        strategies: Investment strategies extracted from the prospectus.
        risks: Risk factors extracted from the prospectus.
        objective: Fund objective extracted from the prospectus.
        filing_metadata: Metadata about the source filing.
    """

    structured_data: Dict = field(default_factory=dict)
    markdown: str = ""
    ticker: str = "UNKNOWN"
    managers: Optional[str] = None
    strategies: Optional[str] = None
    risks: Optional[str] = None
    objective: Optional[str] = None
    filing_metadata: Optional[object] = None


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

    def get_nport_bulk(
        self,
        ciks: List[str],
        company_tickers_json_path: str,
        *,
        max_workers: int = 6,
        batch_size: int = 4,
        min_similarity: float = 0.74,
        verbose: bool = False,
    ) -> List[dict]:
        """Download and parse NPORT-P filings for multiple CIKs in parallel.

        Mirrors the notebook pattern:

        * Fetches filings for each CIK sorted newest-first.
        * Processes filings in chunks of ``batch_size`` using
          :class:`ProcessPoolExecutor`.
        * Stops a CIK's processing as soon as a duplicate fund name is
          encountered — older filings are not downloaded.

        Args:
            ciks: List of SEC CIK numbers
                (e.g. ``["0000036405", "0000052848"]``).
            company_tickers_json_path: Path to ``company_tickers.json`` used
                for holding ticker enrichment.
            max_workers: Number of parallel worker processes per batch
                (default 6).
            batch_size: Number of filings to process in parallel before
                checking for duplicates (default 4).
            min_similarity: Minimum fuzzy-match score for ticker enrichment
                (default 0.74).
            verbose: Print progress info to stdout.

        Returns:
            List of result dicts, one per unique (cik, reporting_period)
            combination. Each dict contains ``fund_name``, ``series_id``,
            ``reporting_period``, ``holdings``, ``holdings_df``, ``result``,
            ``derivatives``, ``not_matches``, ``ticker``, ``report_date``,
            and ``nport_metadata``.
        """
        from tqdm.auto import tqdm
        from functools import partial

        all_results: List[dict] = []

        for cik_idx, cik in enumerate(ciks, 1):
            if verbose:
                tqdm.write(f"[{cik_idx}/{len(ciks)}] 📂 Fetching NPORT-P filings for CIK {cik}...")

            entity = Entity(cik)
            filings = sorted(
                entity.get_filings(form="NPORT-P"),
                key=lambda f: f.report_date,
                reverse=True,
            )

            if not filings:
                logger.warning(f"[CIK {cik}] No NPORT-P filings found")
                continue

            if verbose:
                tqdm.write(f"   Found {len(filings)} filings — processing in batches of {batch_size}")

            funds_seen: set = set()
            duplicate_found = False
            cik_results: List[dict] = []

            worker_fn = partial(
                _process_nport_filing_worker,
                ticker=cik,
                company_json_path=company_tickers_json_path,
                email_identity=self._email_identity,
                min_similarity=min_similarity,
            )

            batches = range(0, len(filings), batch_size)
            for batch_start in tqdm(batches, desc=f"CIK {cik}", unit="batch", leave=False):
                if duplicate_found:
                    break

                batch = filings[batch_start: batch_start + batch_size]

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(worker_fn, f) for f in batch]

                    for future in as_completed(futures):
                        try:
                            res = future.result()
                        except Exception as e:
                            logger.warning(f"[CIK {cik}] Worker error: {e}")
                            continue

                        if res is None:
                            continue

                        fund_key = res["fund_name"].lower()
                        if fund_key in funds_seen:
                            if verbose:
                                tqdm.write(f"   🚨 Duplicate fund '{res['fund_name']}' — stopping CIK {cik}")
                            duplicate_found = True
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

                        funds_seen.add(fund_key)
                        cik_results.append(res)

                        if verbose:
                            tqdm.write(
                                f"   ✅ CIK {cik} — {res['fund_name']} "
                                f"({res['reporting_period']}) | "
                                f"Holdings: {len(res['holdings'])} | "
                                f"Unmatched: {len(res['not_matches'])}"
                            )

            all_results.extend(cik_results)
            if verbose:
                tqdm.write(f"   Done CIK {cik}: {len(cik_results)} unique filings saved")

        logger.info(f"[get_nport_bulk] Complete: {len(all_results)} total filings extracted")
        return all_results

    @staticmethod
    def enrich_funds_with_nport(
        funds: List[FundData],
        nport_results: List[dict],
        *,
        verbose: bool = False,
    ) -> None:
        """Merge NPORT-P results into a list of :class:`FundData` objects in-place.

        Matches each NPORT result to a fund by ``series_id`` and populates
        ``non_derivatives``, ``derivatives``, ``series_id``, and
        ``nport_metadata`` on the matched fund.

        Args:
            funds: List of :class:`FundData` objects to enrich (mutated in-place).
            nport_results: Output of :meth:`get_nport_bulk` — list of result
                dicts, each containing ``series_id``, ``holdings_df``,
                ``derivatives``, ``reporting_period``, and ``nport_metadata``.
            verbose: Print match/miss info to stdout.
        """
        proc = NPortProcessor()
        matched = 0

        
        for res in nport_results:
            fund_name = res.get("fund_name")
            if not fund_name:
                if verbose:
                    print(f"[enrich_nport] ⚠  result missing series_id — skipping (keys: {list(res.keys())})")
                continue

            for fund in funds:
                name = getattr(fund, "name", None)
                if fund_name.lower() != name.lower():
                    if verbose:
                        print(f"[enrich_nport]    no match: nport={fund_name!r}  fund={fund.ticker} name={name!r}")
                    continue

                reporting_period = str(res["reporting_period"])

                fund.non_derivatives = NonDerivatives(
                    date=reporting_period,
                    holdings_df=res["holdings_df"],
                )

                raw_derivatives = res.get("derivatives")
                derivatives_df = proc.to_df(raw_derivatives) if raw_derivatives else None
                fund.derivatives = Derivatives(
                    date=reporting_period,
                    derivatives_df=derivatives_df,
                )

                fund.nport_metadata = res["nport_metadata"]
                matched += 1

                break

        if verbose:
            unmatched = len(nport_results) - matched
            print(f"\nEnrichment complete: {matched} matched, {unmatched} unmatched")

    @staticmethod
    def verify_nport_integrity(funds: List[FundData]) -> None:
        """Print a data-integrity report for NPORT holdings on a fund list.

        Checks each fund for a populated ``non_derivatives.holdings_df`` and
        prints a summary of valid vs missing entries.

        Args:
            funds: List of :class:`FundData` objects to inspect.
        """
        print("\n" + "=" * 40)
        print("DATA INTEGRITY VERIFICATION")
        print("=" * 40)

        valid_count = 0
        none_count = 0

        for fund in funds:
            has_data = False
            try:
                if (
                    hasattr(fund, "non_derivatives")
                    and fund.non_derivatives is not None
                    and fund.non_derivatives.holdings_df is not None
                ):
                    has_data = True
            except Exception:
                has_data = False

            if has_data:
                valid_count += 1
            else:
                none_count += 1
                print(f"❌ {fund.name:<20} | Status: DATAFRAME IS NONE/MISSING")

        print("-" * 40)
        print(f"Total Funds Checked : {len(funds)}")
        print(f"Valid DataFrames    : {valid_count}")
        print(f"None/Missing Values : {none_count}")
        print("=" * 40)

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

    def get_prospectus_bulk(
        self,
        ciks: List[str],
        *,
        max_workers: int = 4,
        batch_size: int = 5,
        grace_batches: int = 5,
        verbose: bool = False,
    ) -> Dict[str, ProspectusResult]:
        """Download and parse 497K prospectus filings for multiple CIKs.

        Uses a streaming batch approach to avoid downloading all filings
        upfront:

        1. Fetch filing metadata for a CIK (no HTML yet).
        2. Download + parse ``batch_size`` filings at a time in parallel.
        3. On first duplicate ticker, continue for up to ``grace_batches``
           more batches to catch any remaining unique funds, then stop.

        Args:
            ciks: List of SEC CIK numbers
                (e.g. ``["0000102909", "0000895421"]``).
            max_workers: Number of parallel worker processes (default 4).
            batch_size: Number of filings to download and parse per batch
                before checking for duplicates (default 5).
            grace_batches: Number of additional batches to process after the
                first duplicate is detected (default 5).
            verbose: Print progress info to stdout.

        Returns:
            Dict mapping each extracted ticker to its
            :class:`ProspectusResult`.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        from tqdm.auto import tqdm

        results: Dict[str, ProspectusResult] = {}
        seen_tickers: set = set()
        worker_fn = partial(_parse_prospectus_worker, email_identity=self._email_identity)

        for cik_idx, cik in enumerate(ciks, 1):
            if verbose:
                tqdm.write(f"[{cik_idx}/{len(ciks)}] 📂 Fetching 497K filings for CIK {cik}...")

            entity = Entity(cik)
            filings = entity.get_filings(form="497K")

            if not filings:
                logger.warning(f"[CIK {cik}] No 497K filings found")
                continue

            # De-duplicate accession numbers (metadata only, no download yet)
            unique_filings = []
            seen_accessions: set = set()
            for f in filings:
                if f.accession_number not in seen_accessions:
                    unique_filings.append(f)
                    seen_accessions.add(f.accession_number)

            if verbose:
                tqdm.write(f"   Found {len(unique_filings)} unique filings — processing in batches of {batch_size}")

            # ── Streaming batch loop ──────────────────────────────────────
            # grace_remaining > 0 means we are in the grace period after the
            # first duplicate was seen.  -1 means no duplicate yet.
            grace_remaining: int = -1
            batches = range(0, len(unique_filings), batch_size)

            for batch_start in tqdm(batches, desc=f"CIK {cik}", unit="batch", leave=False):
                # In grace period: decrement and stop when exhausted
                if grace_remaining == 0:
                    if verbose:
                        tqdm.write(f"   ⏹ Grace period exhausted — stopping CIK {cik}")
                    break
                if grace_remaining > 0:
                    grace_remaining -= 1

                batch = unique_filings[batch_start: batch_start + batch_size]

                # Phase 1: download this batch sequentially
                batch_data: List[tuple] = []
                for filing in batch:
                    try:
                        raw_text = filing.text() if hasattr(filing, "text") else str(filing.obj())
                        if raw_text:
                            batch_data.append((
                                raw_text,
                                filing.accession_number,
                                getattr(filing, "report_date", None),
                                getattr(filing, "filing_date", None),
                                getattr(filing, "form", "497K"),
                                getattr(filing, "url", ""),
                            ))
                    except Exception as e:
                        logger.warning(f"[CIK {cik}] Failed to download {filing.accession_number}: {e}")

                # Phase 2: parse this batch in parallel
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(worker_fn, fd): fd[1]
                        for fd in batch_data
                    }

                    for future in as_completed(futures):
                        accession = futures[future]
                        try:
                            extracted_ticker, res, status = future.result()
                        except Exception as e:
                            logger.warning(f"[CIK {cik}] Worker error on {accession}: {e}")
                            continue

                        if status == "aborted" or extracted_ticker in seen_tickers:
                            if grace_remaining == -1:
                                # First duplicate — start grace period
                                grace_remaining = grace_batches
                                if verbose and extracted_ticker:
                                    tqdm.write(
                                        f"   🚨 Duplicate '{extracted_ticker}' — "
                                        f"entering grace period ({grace_batches} batches left)"
                                    )
                            continue

                        if status == "success" and res is not None:
                            seen_tickers.add(extracted_ticker)
                            results[extracted_ticker] = ProspectusResult(**res)
                            if verbose:
                                tqdm.write(f"   ✅ Extracted: {extracted_ticker}")

        logger.info(f"[get_prospectus_bulk] Complete: {len(results)} unique funds extracted")
        return results

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

        parser = ProspectusExtractor.from_text(raw_text, ticker=ticker)

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
