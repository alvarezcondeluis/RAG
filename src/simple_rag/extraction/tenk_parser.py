import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Any

from tqdm import tqdm
from edgar import set_identity, Company, use_local_storage

from src.simple_rag.extraction.company_extractor import CompanyExtractor
from src.simple_rag.models.company import CompanyEntity, FilingMetadata
from src.simple_rag.utils.cache_manager import is_cached, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

def _process_insider_batch(batch_data: List[Any]) -> List[Any]:
    """
    Helper function for multiprocess execution.
    Must be declared at the module level for ProcessPoolExecutor to pickle it.
    """
    try:
        extractor = CompanyExtractor()
        return extractor.extract_insider_transactions_batch(batch_data)
    except Exception as e:
        logger.error(f"Batch error: {e}")
        return []

class TenKParser:
    """
    Parser for company 10-K SEC filings.
    Also extracts DEF 14A (Executive Compensation) and Form 4s (Insider Trades).
    """

    def __init__(self, email_identity: str, cache_dir: str = "./edgar_cache", max_workers: int = 8):
        """
        Args:
            email_identity (str): SEC EDGAR identity email (e.g., your@email.com).
            cache_dir (str): Directory to cache downloaded filings to avoid rate limits.
            max_workers (int): Number of parallel workers for processing insider transactions.
        """
        set_identity(email_identity)
        use_local_storage(cache_dir)
        
        self.extractor = CompanyExtractor()
        self.max_workers = max_workers

    def process_companies(self, tickers: List[str], max_form4_date: str = '2025-09-01:') -> List[CompanyEntity]:
        """
        Process filings for a list of tickers.
        Returns a list of CompanyEntity models.
        """
        companies: Dict[str, CompanyEntity] = {}

        for ticker in tqdm(tickers, desc="Processing companies"):
            try:
                company = Company(ticker)
                
                # --- Get 10-K ---
                tenk_filing = company.get_filings(form="10-K").latest()
                if is_cached(ticker, "10-K", tenk_filing.accession_number):
                    tqdm.write(f"[{ticker}] Loading 10-K from cache...")
                    tenk = load_from_cache(ticker, "10-K", tenk_filing.accession_number)
                else:
                    tqdm.write(f"[{ticker}] Downloading 10-K...")
                    tenk = tenk_filing.obj()
                    save_to_cache(ticker, "10-K", tenk_filing.accession_number, tenk)

                # --- Get DEF 14A ---
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

                # --- Get Form 4s ---
                ins_filings = company.get_filings(form="4").filter(date=max_form4_date)
                
                # --- Extract 10-K Data ---
                income_statement = tenk.income_statement
                
                tqdm.write(f"[{ticker}] Period of report: {tenk_filing.period_of_report} | Company: {tenk.company}")
                
                metadata = FilingMetadata(
                    accession_number=tenk_filing.accession_no,
                    filing_type=tenk_filing.form,
                    filing_date=tenk_filing.filing_date,
                    report_period_end=tenk_filing.period_of_report,
                    filing_url=tenk_filing.url,
                    cik=str(tenk_filing.cik)
                )
                
                companies[ticker] = CompanyEntity(name=tenk.company, cik=str(tenk_filing.cik), ticker=ticker)
                
                df = income_statement.to_dataframe()
                statement_dict = self.extractor.extract_income_statement_dict(df)
                self.extractor.process_income_statement_dict(companies[ticker], statement_dict, metadata, str(income_statement))
                
                filing_10k = companies[ticker].filings_10k[metadata.filing_date]
                filing_10k.business_information = tenk["Item 1"]
                filing_10k.risk_factors = tenk["Item 1A"]
                filing_10k.balance_sheet_text = str(tenk.balance_sheet)
                filing_10k.cash_flow_text = str(tenk.cash_flow_statement)
                filing_10k.management_discussion_and_analysis = tenk["Item 7"]
                filing_10k.legal_proceedings = tenk["Item 3"]
                filing_10k.properties = tenk["Item 2"]
                
                # --- Extract DEF 14A Data ---
                if exec_comp_obj:
                    # Note: We pass the URL here based on previous user implementation
                    self.extractor.extract_executive_compensation(companies[ticker], exec_comp_obj, metadata, exec_comp_filing.url)
                
                # --- Extract Form 4 Data  ---
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
                    batches = [insider_args[i:i + batch_size] for i in range(0, len(insider_args), batch_size)]
                    tqdm.write(f"[{ticker}] Processing {len(batches)} batches with up to {batch_size} filings each")
                    
                    with ProcessPoolExecutor(max_workers=self.max_workers) as insider_executor:
                        futures = {insider_executor.submit(_process_insider_batch, batch): batch for batch in batches}
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{ticker}] Insider batches", leave=False, unit="batch"):
                            transactions = future.result()
                            if transactions:
                                companies[ticker].insider_trades.extend(transactions)
                    
                    tqdm.write(f"[{ticker}] ✓ Processed {len(companies[ticker].insider_trades)} insider transactions\n")
                
            except Exception as e:
                tqdm.write(f"[{ticker}] ❌ Error processing company: {e}")
                traceback.print_exc()

        return list(companies.values())

