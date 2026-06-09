"""
Company data ingestion operations for Neo4j database.
Includes bulk ingestion of company data from EDGAR filings.
"""

from typing import Optional, List, Dict, Any
from datetime import date
import logging
from tqdm import tqdm
from .company_crud_ops import CompanyCrudOperations
from src.simple_rag.models.company import CompanyEntity

logger = logging.getLogger(__name__)


class CompanyIngestionOperations(CompanyCrudOperations):
    """Bulk company data ingestion operations.
    
    Inherits from CompanyCrudOperations so that batch methods like
    ingest_companies_batch() can call self.create_or_update_company(),
    self.add_10k_filing(), self.add_risk_factors_section(), etc.
    """
    
    def ingest_companies_batch(
        self,
        companies: List['CompanyEntity'],
        verbose: bool = False
    ) -> Dict[str, int]:
        """
        Batch ingest a list of CompanyEntity objects into Neo4j.
        
        This method processes each company and all its associated data:
        - Company node
        - 10-K filings with all sections
        - Financial metrics with segments
        - CEO and compensation data
        - Insider transactions
        
        Args:
            companies: List of CompanyEntity objects to ingest
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with ingestion statistics
            
        """
        from datetime import datetime
        
        stats = {
            'companies': 0,
            'filings_10k': 0,
            'sections': 0,
            'financial_metrics': 0,
            'segments': 0,
            'compensation_packages': 0,
            'insider_transactions': 0,
            'errors': 0
        }
        
        total_companies = len(companies)
        
        for idx, company in enumerate(companies, 1):
            try:
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"Processing Company {idx}/{total_companies}: {company.name} ({company.ticker})")
                    print(f"{'='*80}")
                
                # 1. Create/Update Company Node
                self.create_or_update_company(
                    ticker=company.ticker,
                    name=company.name,
                    cik=company.cik
                )
                stats['companies'] += 1
                
                # 2. Process 10-K Filings
                for filing_date_key, filing in company.filings_10k.items():
                    if verbose:
                        print(f"\n  📄 Processing 10-K filing from {filing_date_key}")
                    
                    # Add the 10-K filing
                    self.add_10k_filing(
                        company_ticker=company.ticker,
                        accession_number=filing.filing_metadata.accession_number,
                        filing_url=filing.filing_metadata.filing_url,
                        filing_date=filing.filing_metadata.filing_date,
                        filing_type=filing.filing_metadata.filing_type,
                        report_period_end=filing.filing_metadata.report_period_end
                    )
                    stats['filings_10k'] += 1
                    
                    # Add sections — pass report_period_end so year resolution uses
                    # the fiscal year, not the calendar year of the filing date.
                    rpe = filing.filing_metadata.report_period_end
                    if filing.risk_factors:
                        self.add_risk_factors_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            risk_factors_text=filing.risk_factors,
                            report_period_end=rpe,
                        )
                        stats['sections'] += 1

                    if filing.business_information:
                        self.add_business_information_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            business_info_text=filing.business_information,
                            report_period_end=rpe,
                        )
                        stats['sections'] += 1

                    if filing.legal_proceedings:
                        self.add_legal_proceedings_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            legal_proceedings_text=filing.legal_proceedings,
                            report_period_end=rpe,
                        )
                        stats['sections'] += 1

                    if filing.management_discussion_and_analysis:
                        self.add_management_discussion_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            mda_text=filing.management_discussion_and_analysis,
                            report_period_end=rpe,
                        )
                        stats['sections'] += 1

                    if filing.properties:
                        self.add_properties_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            properties_text=filing.properties,
                            report_period_end=rpe,
                        )
                        stats['sections'] += 1
                    
                    # Process income statements
                    for period_date, income_stmt in filing.income_statements.items():
                        if verbose:
                            print(f"    💰 Processing financials for period {period_date}")
                        
                        # Add financials section
                        self.add_financials_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            fiscal_year=income_stmt.fiscal_year or period_date.year,
                            income_statement_text=filing.income_statement_text,
                            balance_sheet_text=filing.balance_sheet_text,
                            cash_flow_text=filing.cash_flow_text
                        )
                        stats['sections'] += 1
                        
                        # Add individual financial metrics
                        fiscal_year = income_stmt.fiscal_year or period_date.year
                        
                        # Revenue
                        if income_stmt.revenue and income_stmt.revenue.value:
                            segments = [
                                {
                                    "label": seg.label,
                                    "value": seg.amount,
                                    "percentage": seg.percentage
                                }
                                for seg in income_stmt.revenue.segments
                            ] if income_stmt.revenue.segments else None
                             
                            self.add_financial_metric(
                                company_ticker=company.ticker,
                                filing_date=filing.filing_metadata.filing_date,
                                fiscal_year=fiscal_year,
                                metric_label="Revenue",
                                metric_value=income_stmt.revenue.value,
                                segments=segments
                            )
                            stats['financial_metrics'] += 1
                            if segments:
                                stats['segments'] += len(segments)
                        
                        # Cost of Sales
                        if income_stmt.cost_of_sales and income_stmt.cost_of_sales.value:
                            self.add_financial_metric(
                                company_ticker=company.ticker,
                                filing_date=filing.filing_metadata.filing_date,
                                fiscal_year=fiscal_year,
                                metric_label="CostOfSales",
                                metric_value=income_stmt.cost_of_sales.value
                            )
                            stats['financial_metrics'] += 1
                        
                        # Gross Profit
                        if income_stmt.gross_profit and income_stmt.gross_profit.value:
                            self.add_financial_metric(
                                company_ticker=company.ticker,
                                filing_date=filing.filing_metadata.filing_date,
                                fiscal_year=fiscal_year,
                                metric_label="GrossProfit",
                                metric_value=income_stmt.gross_profit.value
                            )
                            stats['financial_metrics'] += 1
                        
                        # Operating Expenses
                        if income_stmt.operating_expenses and income_stmt.operating_expenses.value:
                            self.add_financial_metric(
                                company_ticker=company.ticker,
                                filing_date=filing.filing_metadata.filing_date,
                                fiscal_year=fiscal_year,
                                metric_label="OperatingExpenses",
                                metric_value=income_stmt.operating_expenses.value
                            )
                            stats['financial_metrics'] += 1
                        
                        # Operating Income
                        if income_stmt.operating_income and income_stmt.operating_income.value:
                            self.add_financial_metric(
                                company_ticker=company.ticker,
                                filing_date=filing.filing_metadata.filing_date,
                                fiscal_year=fiscal_year,
                                metric_label="OperatingIncome",
                                metric_value=income_stmt.operating_income.value
                            )
                            stats['financial_metrics'] += 1
                        
                        # Net Income
                        if income_stmt.net_income and income_stmt.net_income.value:
                            self.add_financial_metric(
                                company_ticker=company.ticker,
                                filing_date=filing.filing_metadata.filing_date,
                                fiscal_year=fiscal_year,
                                metric_label="NetIncome",
                                metric_value=income_stmt.net_income.value
                            )
                            stats['financial_metrics'] += 1
                
               # 3. Process Executive Compensation
                if company.executive_compensation:
                    if verbose:
                        print(f"\n  👔 Processing executive compensation")
                    
                    exec_comp = company.executive_compensation
                    
                    # 1. Extract the Accession Number from the SEC URL
                    import re
                    accession_num = None
                    if exec_comp.url:
                        # Looks for the standard SEC pattern: "0001308179-26-000008" before "-index.html"
                        match = re.search(r'/([^/]+)-index\.html$', exec_comp.url)
                        if match:
                            accession_num = match.group(1)
                    
                    # Fallback ID just in case the regex fails
                    if not accession_num:
                        accession_num = f"{company.ticker}_DEF14A_auto"

                    # 2. Extract the Date from the text field (e.g., "DEF 14A: Apple Inc. - 2026-01-08")
                    exec_filing_date = None
                    if exec_comp.text:
                        date_match = re.search(r'\d{4}-\d{2}-\d{2}', exec_comp.text)
                        if date_match:
                            from datetime import datetime
                            exec_filing_date = datetime.strptime(date_match.group(), "%Y-%m-%d").date()
                    
                    # 3. Send it to the database!
                    self.add_ceo_and_compensation(
                        company_ticker=company.ticker,
                        ceo_name=exec_comp.ceo_name,
                        ceo_compensation=exec_comp.ceo_compensation,
                        ceo_actually_paid=exec_comp.ceo_actually_paid,
                        shareholder_return=exec_comp.shareholder_return,
                        accession_number=accession_num,
                        filing_url=exec_comp.url,
                        filing_date=exec_filing_date,
                        fiscal_year_end=exec_comp.fiscal_year_end,
                    )
                    stats['compensation_packages'] += 1
                    
                # 4. Process Insider Transactions
                if company.insider_trades:
                    if verbose:
                        print(f"\n  📊 Processing {len(company.insider_trades)} insider transactions")
                    
                    for trade in company.insider_trades:
                        # Parse date string to date object
                        trade_date = None
                        if trade.date:
                            try:
                                if isinstance(trade.date, str):
                                    trade_date = datetime.strptime(trade.date, "%Y-%m-%d").date()
                                else:
                                    trade_date = trade.date
                            except:
                                trade_date = datetime.now().date()
                        else:
                            trade_date = datetime.now().date()
                        
                        self.add_insider_transaction(
                            company_ticker=company.ticker,
                            insider_name=trade.insider_name,
                            transaction_date=trade_date,
                            position=trade.position,
                            transaction_type=trade.transaction_type,
                            shares=float(trade.shares) if trade.shares is not None else None,
                            price=trade.price,
                            value=trade.value,
                            remaining_shares=float(trade.remaining_shares) if trade.remaining_shares is not None else None,
                            filing_url=trade.filing_url,
                            form=trade.form or "4"
                        )
                        stats['insider_transactions'] += 1
                
                if verbose:
                    print(f"  ✅ Completed {company.ticker}")
                    
            except Exception as e:
                stats['errors'] += 1
                print(f"  ❌ Error processing {company.ticker}: {e}")
                logger.error(f"Error ingesting company {company.ticker}: {e}", exc_info=True)
                continue
        
        # Print summary
        if verbose:
            print(f"\n{'='*80}")
            print("📊 BATCH INGESTION SUMMARY")
            print(f"{'='*80}")
            print(f"Companies Processed: {stats['companies']}")
            print(f"10-K Filings: {stats['filings_10k']}")
            print(f"Sections Added: {stats['sections']}")
            print(f"Financial Metrics: {stats['financial_metrics']}")
            print(f"Segments: {stats['segments']}")
            print(f"Compensation Packages: {stats['compensation_packages']}")
            print(f"Insider Transactions: {stats['insider_transactions']}")
            print(f"Errors: {stats['errors']}")
            print(f"{'='*80}")
        
        return stats

    def ingest_filing_chunks(
        self,
        companies: List['CompanyEntity'],
        verbose: bool = True
    ) -> Dict[str, int]:
        stats: Dict[str, int] = {"chunks_created": 0, "errors": 0}

        # Updated Map: (chunk_list_attr, full_text_attr, section_id_suffix)
        SECTION_MAP = [
            ("risk_factors_chunks", "risk_factors", "risk_factors"),
            ("business_information_chunks", "business_information", "business_info"),
            ("management_discussion_chunks", "management_discussion_and_analysis", "mda"),
            ("legal_proceedings_chunks", "legal_proceedings", "legal_proceedings"),
            ("properties_chunks", "properties", "properties"),
        ]

        total = len(companies)

        for idx, company in enumerate(companies, 1):
            try:
                if verbose:
                    print(f"\n[{idx}/{total}] 🧩 Ingesting chunks & full text for {company.name}")

                for _filing_date, filing in company.filings_10k.items():
                    for chunk_attr, text_attr, section_key in SECTION_MAP:
                        # 1. Get the lists and the full text
                        chunk_list: list = getattr(filing, chunk_attr, [])
                        full_section_text: str = getattr(filing, text_attr, None)

                        if not chunk_list and not full_section_text:
                            continue

                        # 2. Convert models to dicts for Neo4j UNWIND
                        chunk_dicts = [c.model_dump() for c in chunk_list]

                        # 3. Call the writer — pass report_period_end for correct year matching
                        created = self.add_section_chunks(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            section_name=section_key,
                            full_text=full_section_text,
                            chunks=chunk_dicts,
                            report_period_end=filing.filing_metadata.report_period_end,
                        )
                        stats["chunks_created"] += created

                        if verbose and created > 0:
                            print(f"  ✅ {company.ticker} / {section_key}: {created} chunks + Full Text")

            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Error for {company.ticker}: {e}", exc_info=True)
                continue

        return stats