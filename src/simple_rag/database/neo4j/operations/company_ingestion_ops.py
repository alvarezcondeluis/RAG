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
    
    def ingest_company_data(self, companies: Dict[str, CompanyEntity]) -> Dict[str, Any]:
        """
        Ingest company data from EDGAR filings into Neo4j following the specified schema.
        
        This method creates a comprehensive graph structure for company data including:
        - Company nodes with basic information
        - 10KFiling nodes for each 10-K filing
        - Document nodes for source SEC documents
        - Section nodes (Financials, RiskFactor, BusinessInformation, etc.)
        - FinancialMetric and Segment nodes for financial data breakdowns
        - CompensationPackage nodes for executive compensation
        - InsiderTransaction nodes for Form 4 filings
        - Person nodes for executives and insiders
        
        Args:
            companies: Dictionary mapping ticker symbols to CompanyEntity objects
            
        Returns:
            Dictionary with ingestion statistics
            
        Example:
            from src.simple_rag.models.company import CompanyEntity
            
            companies = {
                'AAPL': CompanyEntity(name='Apple Inc.', ticker='AAPL', cik='0000320193'),
                'MSFT': CompanyEntity(name='Microsoft Corporation', ticker='MSFT', cik='0000789019')
            }
            
            db = Neo4jDatabase(auto_start=True)
            stats = db.ingest_company_data(companies)
            print(f"Ingested {stats['companies_created']} companies")
            db.close()
        """
        stats = {
            "companies_created": 0,
            "documents_created": 0,
            "filings_10k_created": 0,
            "section_nodes_created": 0,
            "financial_metrics_created": 0,
            "segments_created": 0,
            "compensation_packages_created": 0,
            "insider_transactions_created": 0,
            "person_nodes_created": 0
        }
        
        print(f"📊 Ingesting data for {len(companies)} companies...")
        
        for ticker, company in tqdm(companies.items(), desc="Ingesting companies"):
            try:
                # 1. Create/Update Company Node
                company_query = """
                MERGE (c:Company {ticker: $ticker})
                ON CREATE SET
                    c.name = $name,
                    c.cik = $cik,
                    c.createdAt = timestamp()
                ON MATCH SET
                    c.name = $name,
                    c.cik = $cik,
                    c.updatedAt = timestamp()
                RETURN c
                """
                
                company_result = self._execute_write(company_query, {
                    "ticker": company.ticker,
                    "name": company.name,
                    "cik": company.cik
                })
                
                if company_result:
                    stats["companies_created"] += 1
                    logger.info(f"✅ Created/Updated company: {company.ticker}")
                
                # 2. Process Executive Compensation (to create Person/CEO relationship)
                if company.executive_compensation:
                    exec_comp = company.executive_compensation
                    
                    if exec_comp.ceo_name:
                        # Create Person node for CEO
                        person_query = """
                        MERGE (p:Person {name: $ceo_name})
                        ON CREATE SET p.createdAt = timestamp()
                        ON MATCH SET p.updatedAt = timestamp()
                        RETURN p
                        """
                        
                        person_result = self._execute_write(person_query, {
                            "ceo_name": exec_comp.ceo_name
                        })
                        
                        if person_result:
                            stats["person_nodes_created"] += 1
                        
                        # Create CEO employment relationship
                        ceo_relationship_query = """
                        MATCH (c:Company {ticker: $ticker})
                        MATCH (p:Person {name: $ceo_name})
                        MERGE (c)-[r:EMPLOYED_AS_CEO]->(p)
                        SET r.ceoCompensation = $ceo_compensation,
                            r.ceoActuallyPaid = $ceo_actually_paid,
                            r.date = $date
                        RETURN r
                        """
                        
                        self._execute_write(ceo_relationship_query, {
                            "ticker": company.ticker,
                            "ceo_name": exec_comp.ceo_name,
                            "ceo_compensation": exec_comp.ceo_compensation,
                            "ceo_actually_paid": exec_comp.ceo_actually_paid,
                            "date": str(date.today())
                        })
                        
                        # Create Document node for DEF 14A
                        if exec_comp.url:
                            # Extract accession number from URL or use a generated one
                            exec_comp_accession = f"{company.ticker}_DEF14A_{date.today().isoformat()}"
                            
                            doc_query = """
                            MERGE (doc:Document {accesionNumber: $accession_number})
                            ON CREATE SET
                                doc.url = $url,
                                doc.type = $doc_type,
                                doc.filingdate = $filing_date,
                                doc.createdAt = timestamp()
                            ON MATCH SET
                                doc.updatedAt = timestamp()
                            RETURN doc
                            """
                            
                            doc_result = self._execute_write(doc_query, {
                                "accession_number": exec_comp_accession,
                                "url": exec_comp.url,
                                "doc_type": exec_comp.form or "DEF 14A",
                                "filing_date": str(date.today())
                            })
                            
                            if doc_result:
                                stats["documents_created"] += 1
                            
                            # Create CompensationPackage node
                            comp_package_id = f"{company.ticker}_comp_{date.today().isoformat()}"
                            
                            comp_package_query = """
                            MATCH (c:Company {ticker: $ticker})
                            MATCH (p:Person {name: $ceo_name})
                            MATCH (doc:Document {accesionNumber: $accession_number})
                            
                            MERGE (cp:CompensationPackage {id: $comp_package_id})
                            SET cp.totalCompensation = $total_compensation,
                                cp.shareholderReturn = $shareholder_return,
                                cp.date = $date
                            
                            MERGE (p)-[:RECEIVED_COMPENSATION]->(cp)
                            MERGE (c)<-[:AWARDED_BY]-(cp)
                            MERGE (cp)-[:DISCLOSED_IN]->(doc)
                            RETURN cp
                            """
                            
                            comp_result = self._execute_write(comp_package_query, {
                                "ticker": company.ticker,
                                "ceo_name": exec_comp.ceo_name,
                                "accession_number": exec_comp_accession,
                                "comp_package_id": comp_package_id,
                                "total_compensation": exec_comp.ceo_compensation,
                                "shareholder_return": exec_comp.shareholder_return,
                                "date": str(date.today())
                            })
                            
                            if comp_result:
                                stats["compensation_packages_created"] += 1
                
                # 3. Ingest 10-K Filings
                for filing_date, filing_10k in company.filings_10k.items():
                    filing_id = f"{company.ticker}_{filing_date.isoformat()}"
                    
                    # Create Document node for 10-K
                    doc_query = """
                    MERGE (doc:Document {accesionNumber: $accession_number})
                    ON CREATE SET
                        doc.url = $filing_url,
                        doc.type = $filing_type,
                        doc.filingdate = $filing_date,
                        doc.reportingDate = $report_period_end,
                        doc.createdAt = timestamp()
                    ON MATCH SET
                        doc.updatedAt = timestamp()
                    RETURN doc
                    """
                    
                    doc_result = self._execute_write(doc_query, {
                        "accession_number": filing_10k.filing_metadata.accession_number,
                        "filing_url": filing_10k.filing_metadata.filing_url,
                        "filing_type": filing_10k.filing_metadata.filing_type,
                        "filing_date": filing_date,
                        "report_period_end": filing_10k.filing_metadata.report_period_end
                    })
                    
                    if doc_result:
                        stats["documents_created"] += 1
                    
                    # Create 10KFiling node (note the label change from Filing10K to 10KFiling)
                    filing_query = """
                    MATCH (c:Company {ticker: $ticker})
                    MATCH (doc:Document {accesionNumber: $accession_number})
                    
                    MERGE (f:Filing10K {id: $filing_id})
                    ON CREATE SET f.createdAt = timestamp()
                    ON MATCH SET f.updatedAt = timestamp()
                    
                    MERGE (c)-[r:HAS_FILING]->(f)
                    SET r.date = $filing_date
                    
                    MERGE (f)-[:EXTRACTED_FROM]->(doc)
                    RETURN f
                    """
                    
                    # Note: Neo4j doesn't allow labels starting with numbers in Cypher,
                    # so we'll use a backtick workaround
                    filing_query = filing_query.replace(":10KFiling", ":`10KFiling`")
                    
                    filing_result = self._execute_write(filing_query, {
                        "ticker": company.ticker,
                        "accession_number": filing_10k.filing_metadata.accession_number,
                        "filing_id": filing_id,
                        "filing_date": filing_date
                    })
                    
                    if filing_result:
                        stats["filings_10k_created"] += 1
                    
                    # Create Section nodes for different parts of the 10-K
                    
                    # Risk Factors
                    if filing_10k.risk_factors:
                        risk_section_query = """
                        MATCH (f:`10KFiling` {id: $filing_id})
                        MERGE (f)-[:HAS_RISK_FACTORS]->(rf:Section:RiskFactor {id: $section_id})
                        SET rf.fullText = $full_text
                        RETURN rf
                        """
                        
                        risk_result = self._execute_write(risk_section_query, {
                            "filing_id": filing_id,
                            "section_id": f"{filing_id}_risk_factors",
                            "full_text": filing_10k.risk_factors
                        })
                        
                        if risk_result:
                            stats["section_nodes_created"] += 1
                    
                    # Business Information
                    if filing_10k.business_information:
                        business_section_query = """
                        MATCH (f:`10KFiling` {id: $filing_id})
                        MERGE (f)-[:HAS_BUSINESS_INFORMATION]->(bi:Section:BusinessInformation {id: $section_id})
                        SET bi.fullText = $full_text
                        RETURN bi
                        """
                        
                        business_result = self._execute_write(business_section_query, {
                            "filing_id": filing_id,
                            "section_id": f"{filing_id}_business_info",
                            "full_text": filing_10k.business_information
                        })
                        
                        if business_result:
                            stats["section_nodes_created"] += 1
                    
                    # Legal Proceedings
                    if filing_10k.legal_proceedings:
                        legal_section_query = """
                        MATCH (f:`10KFiling` {id: $filing_id})
                        MERGE (f)-[:HAS_LEGAL_PROCEEDINGS]->(lp:Section:LegalProceeding {id: $section_id})
                        SET lp.fullTitleSection = $title,
                            lp.fullText = $full_text
                        RETURN lp
                        """
                        
                        legal_result = self._execute_write(legal_section_query, {
                            "filing_id": filing_id,
                            "section_id": f"{filing_id}_legal_proceedings",
                            "title": "Item 3 - Legal Proceedings",
                            "full_text": filing_10k.legal_proceedings
                        })
                        
                        if legal_result:
                            stats["section_nodes_created"] += 1
                    
                    # Management Discussion & Analysis
                    if filing_10k.management_discussion_and_analysis:
                        mda_section_query = """
                        MATCH (f:`10KFiling` {id: $filing_id})
                        MERGE (f)-[:HAS_MANAGEMENT_DISCUSSION]->(md:Section:ManagemetDiscussion {id: $section_id})
                        SET md.fullText = $full_text
                        RETURN md
                        """
                        
                        mda_result = self._execute_write(mda_section_query, {
                            "filing_id": filing_id,
                            "section_id": f"{filing_id}_mda",
                            "full_text": filing_10k.management_discussion_and_analysis
                        })
                        
                        if mda_result:
                            stats["section_nodes_created"] += 1
                    
                    # Properties
                    if filing_10k.properties:
                        properties_section_query = """
                        MATCH (f:`10KFiling` {id: $filing_id})
                        MERGE (f)-[:HAS_PROPERTIES]->(prop:Section:Properties {id: $section_id})
                        SET prop.fullText = $full_text
                        RETURN prop
                        """
                        
                        properties_result = self._execute_write(properties_section_query, {
                            "filing_id": filing_id,
                            "section_id": f"{filing_id}_properties",
                            "full_text": filing_10k.properties
                        })
                        
                        if properties_result:
                            stats["section_nodes_created"] += 1
                    
                    # 4. Process Income Statements (as Section:Financials)
                    for period_date, income_stmt in filing_10k.income_statements.items():
                        financials_id = f"{filing_id}_financials_{period_date.isoformat()}"
                        
                        # Create Section:Financials node
                        financials_section_query = """
                        MATCH (f:`10KFiling` {id: $filing_id})
                        MERGE (fin:Section:Financials {id: $financials_id})
                        SET fin.incomeStatement = $income_statement_text,
                            fin.balanceSheet = $balance_sheet_text,
                            fin.cashFlow = $cash_flow_text,
                            fin.fiscalYear = $fiscal_year
                        
                        MERGE (f)-[:HAS_FINACIALS]->(fin)
                        RETURN fin
                        """
                        
                        financials_result = self._execute_write(financials_section_query, {
                            "filing_id": filing_id,
                            "financials_id": financials_id,
                            "income_statement_text": filing_10k.income_statement_text,
                            "balance_sheet_text": filing_10k.balance_sheet_text,
                            "cash_flow_text": filing_10k.cash_flow_text,
                            "fiscal_year": income_stmt.fiscal_year
                        })
                        
                        if financials_result:
                            stats["section_nodes_created"] += 1
                        
                        # Helper function to create FinancialMetric and Segment nodes
                        def create_financial_metric(metric_label, metric_obj):
                            if not metric_obj or metric_obj.value is None:
                                return
                            
                            metric_id = f"{financials_id}_{metric_label}"
                            
                            # Create FinancialMetric node
                            metric_query = """
                            MATCH (fin:Section:Financials {id: $financials_id})
                            MERGE (fm:FinancialMetric {id: $metric_id})
                            SET fm.value = $value,
                                fm.label = $label
                            
                            MERGE (fin)-[:HAS_METRIC]->(fm)
                            RETURN fm
                            """
                            
                            metric_result = self._execute_write(metric_query, {
                                "financials_id": financials_id,
                                "metric_id": metric_id,
                                "value": metric_obj.value,
                                "label": metric_obj.label or metric_label
                            })
                            
                            if metric_result:
                                stats["financial_metrics_created"] += 1
                            
                            # Create Segment nodes for this metric
                            if metric_obj.segments:
                                for seg in metric_obj.segments:
                                    segment_id = f"{metric_id}_{seg.label.replace(' ', '_')}"
                                    
                                    segment_query = """
                                    MATCH (fm:FinancialMetric {id: $metric_id})
                                    MERGE (seg:Segment {id: $segment_id})
                                    SET seg.label = $label,
                                        seg.value = $value,
                                        seg.percentage = $percentage
                                    
                                    MERGE (fm)-[:HAS_SEGMENT]->(seg)
                                    RETURN seg
                                    """
                                    
                                    segment_result = self._execute_write(segment_query, {
                                        "metric_id": metric_id,
                                        "segment_id": segment_id,
                                        "label": seg.label,
                                        "value": seg.amount,
                                        "percentage": seg.percentage
                                    })
                                    
                                    if segment_result:
                                        stats["segments_created"] += 1
                        
                        # Create metrics for all financial line items
                        create_financial_metric("Revenue", income_stmt.revenue)
                        create_financial_metric("CostOfSales", income_stmt.cost_of_sales)
                        create_financial_metric("GrossProfit", income_stmt.gross_profit)
                        create_financial_metric("OperatingExpenses", income_stmt.operating_expenses)
                        create_financial_metric("OperatingIncome", income_stmt.operating_income)
                        create_financial_metric("OtherIncome", income_stmt.other_income_expense)
                        create_financial_metric("PretaxIncome", income_stmt.pretax_income)
                        create_financial_metric("TaxProvision", income_stmt.provision_for_income_taxes)
                        create_financial_metric("NetIncome", income_stmt.net_income)
                
                # 5. Ingest Insider Transactions (Batched for performance)
                if company.insider_trades:
                    BATCH_SIZE = 100
                    insider_trades = company.insider_trades
                    
                    for i in range(0, len(insider_trades), BATCH_SIZE):
                        batch = insider_trades[i:i + BATCH_SIZE]
                        
                        for trade in batch:
                            if not trade.date or not trade.insider_name:
                                continue
                            
                            # Create Person node for insider
                            person_query = """
                            MERGE (p:Person {name: $insider_name})
                            ON CREATE SET p.createdAt = timestamp()
                            ON MATCH SET p.updatedAt = timestamp()
                            RETURN p
                            """
                            
                            person_result = self._execute_write(person_query, {
                                "insider_name": trade.insider_name
                            })
                            
                            if person_result:
                                stats["person_nodes_created"] += 1
                            
                            # Create Document node for Form 4
                            form4_accession = f"{company.ticker}_{trade.insider_name}_{trade.date}_Form4".replace(" ", "_")
                            
                            doc_query = """
                            MERGE (doc:Document {accesionNumber: $accession_number})
                            ON CREATE SET
                                doc.url = $url,
                                doc.type = $doc_type,
                                doc.filingdate = $filing_date,
                                doc.createdAt = timestamp()
                            ON MATCH SET
                                doc.updatedAt = timestamp()
                            RETURN doc
                            """
                            
                            doc_result = self._execute_write(doc_query, {
                                "accession_number": form4_accession,
                                "url": trade.filing_url,
                                "doc_type": trade.form or "4",
                                "filing_date": trade.date
                            })
                            
                            if doc_result:
                                stats["documents_created"] += 1
                            
                            # Create InsiderTransaction node
                            trade_id = f"{company.ticker}_{trade.insider_name}_{trade.date}_{trade.transaction_type}".replace(" ", "_")
                            
                            transaction_query = """
                            MATCH (c:Company {ticker: $ticker})
                            MATCH (p:Person {name: $insider_name})
                            MATCH (doc:Document {accesionNumber: $accession_number})
                            
                            MERGE (it:InsiderTransaction {id: $trade_id})
                            SET it.position = $position,
                                it.transactionType = $transaction_type,
                                it.shares = $shares,
                                it.price = $price,
                                it.value = $value,
                                it.remainingShares = $remaining_shares
                            
                            MERGE (p)<-[:MADE_BY]-(it)
                            MERGE (c)-[:HAS_INSIDER_TRANSACTION]->(it)
                            MERGE (it)-[:EXTRACTED_FROM]->(doc)
                            RETURN it
                            """
                            
                            transaction_result = self._execute_write(transaction_query, {
                                "ticker": company.ticker,
                                "insider_name": trade.insider_name,
                                "accession_number": form4_accession,
                                "trade_id": trade_id,
                                "position": trade.position,
                                "transaction_type": trade.transaction_type,
                                "shares": trade.shares,
                                "price": trade.price,
                                "value": trade.value,
                                "remaining_shares": trade.remaining_shares
                            })
                            
                            if transaction_result:
                                stats["insider_transactions_created"] += 1
                
                logger.info(f"✅ Completed ingestion for {company.ticker}")
                
            except Exception as e:
                logger.error(f"❌ Error ingesting company {ticker}: {e}")
                print(f"❌ Error ingesting company {ticker}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 INGESTION SUMMARY")
        print("=" * 80)
        print(f"Companies Created/Updated: {stats['companies_created']}")
        print(f"Documents Created: {stats['documents_created']}")
        print(f"10K Filings Created: {stats['filings_10k_created']}")
        print(f"Section Nodes Created: {stats['section_nodes_created']}")
        print(f"Financial Metrics Created: {stats['financial_metrics_created']}")
        print(f"Segments Created: {stats['segments_created']}")
        print(f"Compensation Packages: {stats['compensation_packages_created']}")
        print(f"Insider Transactions: {stats['insider_transactions_created']}")
        print(f"Person Nodes: {stats['person_nodes_created']}")
        print("=" * 80)
        
        return stats

    def ingest_companies_batch(
        self,
        companies: List['CompanyEntity'],
        verbose: bool = True
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
                    
                    # Add sections
                    if filing.risk_factors:
                        self.add_risk_factors_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            risk_factors_text=filing.risk_factors
                        )
                        stats['sections'] += 1
                    
                    if filing.business_information:
                        self.add_business_information_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            business_info_text=filing.business_information
                        )
                        stats['sections'] += 1
                    
                    if filing.legal_proceedings:
                        self.add_legal_proceedings_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            legal_proceedings_text=filing.legal_proceedings
                        )
                        stats['sections'] += 1
                    
                    if filing.management_discussion_and_analysis:
                        self.add_management_discussion_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            mda_text=filing.management_discussion_and_analysis
                        )
                        stats['sections'] += 1
                    
                    if filing.properties:
                        self.add_properties_section(
                            company_ticker=company.ticker,
                            filing_date=filing.filing_metadata.filing_date,
                            properties_text=filing.properties
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
                    
                    # Parse filing date from URL or use a default
                    exec_filing_date = None
                    if exec_comp.url and 'accession_number' in dir(exec_comp):
                        # Try to extract date from accession number if available
                        pass
                    
                    self.add_ceo_and_compensation(
                        company_ticker=company.ticker,
                        ceo_name=exec_comp.ceo_name,
                        ceo_compensation=exec_comp.ceo_compensation,
                        ceo_actually_paid=exec_comp.ceo_actually_paid,
                        shareholder_return=exec_comp.shareholder_return,
                        accession_number=None,  # You may need to add this to ExecutiveCompensation model
                        filing_url=exec_comp.url,
                        filing_date=exec_filing_date
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
                            shares=float(trade.shares) if trade.shares else None,
                            price=trade.price,
                            value=trade.value,
                            remaining_shares=float(trade.remaining_shares) if trade.remaining_shares else None,
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
