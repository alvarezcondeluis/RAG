"""
Company CRUD operations for Neo4j database.
Includes company creation, 10-K filings, sections, financial metrics, 
CEO/compensation, and insider transactions.
"""

from typing import Optional, List, Dict, Any
from datetime import date
import logging
from ..base import Neo4jDatabaseBase

logger = logging.getLogger(__name__)


class CompanyCrudOperations(Neo4jDatabaseBase):
    """Company CRUD operations: create/update companies, filings, sections, metrics."""
    
    def create_or_update_company(
        self,
        ticker: str,
        name: str,
        cik: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create or update a Company node.
        
        Args:
            ticker: Company ticker symbol
            name: Company name
            cik: SEC Central Index Key (CIK)
            
        Returns:
            Created/updated Company node or None if failed
            
        Example:
            company = db.create_or_update_company("AAPL", "Apple Inc.", "0000320193")
        """
        try:
            query = """
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
            
            result = self._execute_write(query, {
                "ticker": ticker,
                "name": name,
                "cik": cik
            })
            
            if result:
                logger.info(f"✅ Created/Updated company: {ticker}")
                return result[0]["c"]
            return None
            
        except Exception as e:
            logger.error(f"❌ Error creating company {ticker}: {e}")
            return None
    
    def add_10k_filing(
        self,
        company_ticker: str,
        accession_number: str,
        filing_url: str,
        filing_date: date,
        filing_type: str = "10-K",
        report_period_end: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add a 10-K filing to a company.
        
        Args:
            company_ticker: Company ticker symbol
            accession_number: SEC accession number
            filing_url: URL to the filing
            filing_date: Date the filing was submitted
            filing_type: Type of filing (default: "10-K")
            report_period_end: End date of the reporting period
            
        Returns:
            Created 10KFiling node or None if failed
        """
        try:
            year = (report_period_end or filing_date).year
            
            # Create Document node
            doc_query = """
            MERGE (doc:Document {accession_number: $accession_number})
            ON CREATE SET
                doc.url = $filing_url,
                doc.form = $filing_type,
                doc.filing_date = $filing_date,
                doc.reporting_date = $report_period_end,
                doc.createdAt = timestamp()
            ON MATCH SET
                doc.updatedAt = timestamp()
            RETURN doc
            """
            
            self._execute_write(doc_query, {
                "accession_number": accession_number,
                "filing_url": filing_url,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "report_period_end": report_period_end
            })
            
            # Create 10KFiling node and relationships
            filing_query = """
            MATCH (c:Company {ticker: $ticker})
            MATCH (doc:Document {accession_number: $accession_number})
            
            MERGE (c)-[r:REPORTS_IN {year: $year}]->(f:`Filing10K`)
            ON CREATE SET f.createdAt = timestamp()
            ON MATCH SET f.updatedAt = timestamp()
            
            MERGE (f)-[:EXTRACTED_FROM]->(doc)
            RETURN f
            """
            
            result = self._execute_write(filing_query, {
                "ticker": company_ticker,
                "accession_number": accession_number,
                "year": year
            })
            
            if result:
                print(f"✅ Added 10-K filing for {company_ticker} ({filing_date})")
                return result[0]["f"]
            return None
            
        except Exception as e:
            print(f"❌ Error adding 10-K filing for {company_ticker}: {e}")
            logger.error(f"Failed to add 10-K filing: {e}", exc_info=True)
            return None
    
    def add_risk_factors_section(
        self,
        company_ticker: str,
        filing_date: date,
        risk_factors_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Add Risk Factors section to a 10-K filing.
        
        Args:
            company_ticker: Company ticker symbol
            filing_date: Date of the filing
            risk_factors_text: Full text of the risk factors section
            
        Returns:
            Created RiskFactor section node or None if failed
        """
        try:
            year = filing_date.year
            
            query = """
            MATCH (c:Company {ticker: $ticker})-[:REPORTS_IN {year: $year}]->(f:Filing10K)
            MERGE (f)-[:HAS_SECTION]->(s:Section:RiskFactor)
            SET s.title = $title,
                s.text = $text,
                s.sectionType = 'risk_factors',
                s.secItem = 'Item 1A'
            RETURN s
            """
            
            result = self._execute_write(query, {
                "ticker": company_ticker,
                "year": year,
                "title": "Item 1A - Risk Factors",
                "text": risk_factors_text
            })
            
            if result:
                print(f"✅ Added Risk Factors section for {company_ticker}")
                return result[0]["s"]
            return None
            
        except Exception as e:
            print(f"❌ Error adding Risk Factors for {company_ticker}: {e}")
            return None
    
    def add_business_information_section(
        self,
        company_ticker: str,
        filing_date: date,
        business_info_text: str
    ) -> Optional[Dict[str, Any]]:
        """Add Business Information section to a 10-K filing."""
        try:
            year = filing_date.year
            
            query = """
            MATCH (c:Company {ticker: $ticker})-[:REPORTS_IN {year: $year}]->(f:Filing10K)
            MERGE (f)-[:HAS_SECTION]->(s:Section:BusinessInformation)
            SET s.title = $title,
                s.text = $text,
                s.sectionType = 'business_info',
                s.secItem = 'Item 1'
            RETURN s
            """
            
            result = self._execute_write(query, {
                "ticker": company_ticker,
                "year": year,
                "title": "Item 1 - Business",
                "text": business_info_text
            })
            
            if result:
                print(f"✅ Added Business Information section for {company_ticker}")
                return result[0]["s"]
            return None
            
        except Exception as e:
            print(f"❌ Error adding Business Information for {company_ticker}: {e}")
            return None
    
    def add_legal_proceedings_section(
        self,
        company_ticker: str,
        filing_date: date,
        legal_proceedings_text: str
    ) -> Optional[Dict[str, Any]]:
        """Add Legal Proceedings section to a 10-K filing."""
        try:
            year = filing_date.year
            
            query = """
            MATCH (c:Company {ticker: $ticker})-[:REPORTS_IN {year: $year}]->(f:Filing10K)
            MERGE (f)-[:HAS_SECTION]->(s:Section:LegalProceeding)
            SET s.title = $title,
                s.text = $text,
                s.sectionType = 'legal_proceedings',
                s.secItem = 'Item 3'
            RETURN s
            """
            
            result = self._execute_write(query, {
                "ticker": company_ticker,
                "year": year,
                "title": "Item 3 - Legal Proceedings",
                "text": legal_proceedings_text
            })
            
            if result:
                print(f"✅ Added Legal Proceedings section for {company_ticker}")
                return result[0]["s"]
            return None
            
        except Exception as e:
            print(f"❌ Error adding Legal Proceedings for {company_ticker}: {e}")
            return None
    
    def add_management_discussion_section(
        self,
        company_ticker: str,
        filing_date: date,
        mda_text: str
    ) -> Optional[Dict[str, Any]]:
        """Add Management Discussion & Analysis section to a 10-K filing."""
        try:
            year = filing_date.year
            
            query = """
            MATCH (c:Company {ticker: $ticker})-[:REPORTS_IN {year: $year}]->(f:Filing10K)
            MERGE (f)-[:HAS_SECTION]->(s:Section:ManagementDiscussion)
            SET s.title = $title,
                s.text = $text,
                s.sectionType = 'mda',
                s.secItem = 'Item 7'
            RETURN s
            """
            
            result = self._execute_write(query, {
                "ticker": company_ticker,
                "year": year,
                "title": "Item 7 - Management Discussion & Analysis",
                "text": mda_text
            })
            
            if result:
                print(f"✅ Added MD&A section for {company_ticker}")
                return result[0]["s"]
            return None
            
        except Exception as e:
            print(f"❌ Error adding MD&A for {company_ticker}: {e}")
            return None
    
    def add_properties_section(
        self,
        company_ticker: str,
        filing_date: date,
        properties_text: str
    ) -> Optional[Dict[str, Any]]:
        """Add Properties section to a 10-K filing."""
        try:
            year = filing_date.year
            
            query = """
            MATCH (c:Company {ticker: $ticker})-[:REPORTS_IN {year: $year}]->(f:Filing10K)
            MERGE (f)-[:HAS_SECTION]->(s:Section:Properties)
            SET s.title = $title,
                s.text = $text,
                s.sectionType = 'properties',
                s.secItem = 'Item 2'
            RETURN s
            """
            
            result = self._execute_write(query, {
                "ticker": company_ticker,
                "year": year,
                "title": "Item 2 - Properties",
                "text": properties_text
            })
            
            if result:
                print(f"✅ Added Properties section for {company_ticker}")
                return result[0]["s"]
            return None
            
        except Exception as e:
            print(f"❌ Error adding Properties for {company_ticker}: {e}")
            return None
    
    def add_financials_section(
        self,
        company_ticker: str,
        filing_date: date,
        fiscal_year: int,
        income_statement_text: Optional[str] = None,
        balance_sheet_text: Optional[str] = None,
        cash_flow_text: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add Financials section to a 10-K filing.
        
        Args:
            company_ticker: Company ticker symbol
            filing_date: Date of the filing
            fiscal_year: Fiscal year of the financials
            income_statement_text: Income statement text
            balance_sheet_text: Balance sheet text
            cash_flow_text: Cash flow statement text
            
        Returns:
            Created Financials section node or None if failed
        """
        try:
            year = filing_date.year
            
            query = """
            MATCH (c:Company {ticker: $ticker})-[:REPORTS_IN {year: $year}]->(f:Filing10K)
            MERGE (f)-[:HAS_FINACIALS]->(fin:Section:Financials)
            SET fin.incomeStatement = $income_statement,
                fin.balanceSheet = $balance_sheet,
                fin.cashFlow = $cash_flow,
                fin.fiscalYear = $fiscal_year
            RETURN fin
            """
            
            result = self._execute_write(query, {
                "ticker": company_ticker,
                "year": year,
                "income_statement": income_statement_text,
                "balance_sheet": balance_sheet_text,
                "cash_flow": cash_flow_text,
                "fiscal_year": fiscal_year
            })
            
            if result:
                print(f"✅ Added Financials section for {company_ticker} (FY{fiscal_year})")
                return result[0]["fin"]
            return None
            
        except Exception as e:
            print(f"❌ Error adding Financials for {company_ticker}: {e}")
            return None
    
    def add_financial_metric(
        self,
        company_ticker: str,
        filing_date: date,
        fiscal_year: int,
        metric_label: str,
        metric_value: float,
        segments: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add a financial metric with optional segments to a Financials section.
        
        Args:
            company_ticker: Company ticker symbol
            filing_date: Date of the filing
            fiscal_year: Fiscal year
            metric_label: Label for the metric (e.g., "Revenue", "NetIncome")
            metric_value: Value of the metric
            segments: Optional list of segment dictionaries with keys:
                     - label: Segment label
                     - value: Segment value
                     - percentage: Percentage of total (optional)
                     
        Returns:
            Created FinancialMetric node or None if failed
            
        Example:
            db.add_financial_metric(
                "AAPL",
                date(2024, 11, 1),
                2024,
                "Revenue",
                394328000000,
                segments=[
                    {"label": "iPhone", "value": 200583000000, "percentage": 50.9},
                    {"label": "Services", "value": 85200000000, "percentage": 21.6}
                ]
            )
        """
        try:
            year = filing_date.year
            
            # Create FinancialMetric node
            metric_query = """
            MATCH (c:Company {ticker: $ticker})-[:REPORTS_IN {year: $year}]->(f:Filing10K)
                  -[:HAS_FINACIALS]->(fin:Section:Financials)
            MERGE (fin)-[:HAS_METRIC]->(fm:FinancialMetric {label: $label})
            SET fm.value = $value
            RETURN fm
            """
            
            result = self._execute_write(metric_query, {
                "ticker": company_ticker,
                "year": year,
                "value": metric_value,
                "label": metric_label
            })
            
            if not result:
                return None
            
            # Create Segment nodes if provided
            if segments:
                for seg in segments:
                    segment_query = """
                    MATCH (c:Company {ticker: $ticker})-[:REPORTS_IN {year: $year}]->(f:Filing10K)
                          -[:HAS_FINACIALS]->(fin:Section:Financials)
                          -[:HAS_METRIC]->(fm:FinancialMetric {label: $metric_label})
                    MERGE (fm)-[:HAS_SEGMENT]->(seg:Segment {label: $label})
                    SET seg.value = $value,
                        seg.percentage = $percentage
                    RETURN seg
                    """
                    
                    self._execute_write(segment_query, {
                        "ticker": company_ticker,
                        "year": year,
                        "metric_label": metric_label,
                        "label": seg["label"],
                        "value": seg.get("value"),
                        "percentage": seg.get("percentage")
                    })
            
            print(f"✅ Added financial metric '{metric_label}' for {company_ticker}")
            return result[0]["fm"]
            
        except Exception as e:
            print(f"❌ Error adding financial metric for {company_ticker}: {e}")
            return None
    
    def add_ceo_and_compensation(
        self,
        company_ticker: str,
        ceo_name: str,
        ceo_compensation: Optional[float] = None,
        ceo_actually_paid: Optional[float] = None,
        shareholder_return: Optional[float] = None,
        accession_number: Optional[str] = None,
        filing_url: Optional[str] = None,
        filing_date: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add CEO and compensation package information to a company.
        
        Args:
            company_ticker: Company ticker symbol
            ceo_name: Name of the CEO
            ceo_compensation: Total CEO compensation
            ceo_actually_paid: Actually paid compensation
            shareholder_return: Shareholder return percentage
            accession_number: SEC accession number for DEF 14A
            filing_url: URL to the proxy statement
            filing_date: Date of the filing
            
        Returns:
            Created CompensationPackage node or None if failed
        """
        try:
            # Create Person node for CEO
            person_query = """
            MERGE (p:Person {name: $ceo_name})
            ON CREATE SET p.createdAt = timestamp()
            ON MATCH SET p.updatedAt = timestamp()
            RETURN p
            """
            
            self._execute_write(person_query, {"ceo_name": ceo_name})
            
            # Create CEO employment relationship
            ceo_rel_query = """
            MATCH (c:Company {ticker: $ticker})
            MATCH (p:Person {name: $ceo_name})
            MERGE (c)-[r:HAS_CEO]->(p)
            SET r.ceoCompensation = $ceo_compensation,
                r.ceoActuallyPaid = $ceo_actually_paid,
                r.date = $date
            RETURN r
            """
            
            self._execute_write(ceo_rel_query, {
                "ticker": company_ticker,
                "ceo_name": ceo_name,
                "ceo_compensation": ceo_compensation,
                "ceo_actually_paid": ceo_actually_paid,
                "date": filing_date or date.today()
            })
            
            # Create Document node if filing info provided
            if accession_number and filing_url:
                doc_query = """
                MERGE (doc:Document {accession_number: $accession_number})
                ON CREATE SET
                    doc.url = $url,
                    doc.form = $form,
                    doc.filing_date = $filing_date,
                    doc.createdAt = timestamp()
                ON MATCH SET
                    doc.updatedAt = timestamp()
                RETURN doc
                """
                
                self._execute_write(doc_query, {
                    "accession_number": accession_number,
                    "url": filing_url,
                    "form": "DEF 14A",
                    "filing_date": filing_date or date.today()
                })
                
                # Create CompensationPackage node
                comp_query = """
                MATCH (c:Company {ticker: $ticker})
                MATCH (p:Person {name: $ceo_name})
                MATCH (doc:Document {accession_number: $accession_number})
                
                MERGE (p)-[:RECEIVED_COMPENSATION]->(cp:CompensationPackage)
                SET cp.totalCompensation = $total_compensation,
                    cp.shareholderReturn = $shareholder_return,
                    cp.date = $date
                
                MERGE (c)<-[:AWARDED_BY]-(cp)
                MERGE (cp)-[:DISCLOSED_IN]->(doc)
                RETURN cp
                """
                
                result = self._execute_write(comp_query, {
                    "ticker": company_ticker,
                    "ceo_name": ceo_name,
                    "accession_number": accession_number,
                    "total_compensation": ceo_compensation,
                    "shareholder_return": shareholder_return,
                    "date": filing_date or date.today()
                })
                
                if result:
                    print(f"✅ Added CEO and compensation for {company_ticker}")
                    return result[0]["cp"]
            else:
                print(f"✅ Added CEO relationship for {company_ticker}")
                return {"ceo_name": ceo_name}
            
            return None
            
        except Exception as e:
            print(f"❌ Error adding CEO/compensation for {company_ticker}: {e}")
            return None
    
    def add_insider_transaction(
        self,
        company_ticker: str,
        insider_name: str,
        transaction_date: date,
        position: Optional[str] = None,
        transaction_type: Optional[str] = None,
        shares: Optional[float] = None,
        price: Optional[float] = None,
        value: Optional[float] = None,
        remaining_shares: Optional[float] = None,
        filing_url: Optional[str] = None,
        form: str = "4"
    ) -> Optional[Dict[str, Any]]:
        """
        Add an insider transaction (Form 4) to a company.
        
        Args:
            company_ticker: Company ticker symbol
            insider_name: Name of the insider
            transaction_date: Date of the transaction
            position: Position/title of the insider
            transaction_type: Type of transaction (e.g., "Purchase", "Sale")
            shares: Number of shares transacted
            price: Price per share
            value: Total value of transaction
            remaining_shares: Shares remaining after transaction
            filing_url: URL to the Form 4 filing
            form: Form type (default: "4")
            
        Returns:
            Created InsiderTransaction node or None if failed
        """
        try:
            # Create Person node
            person_query = """
            MERGE (p:Person {name: $insider_name})
            ON CREATE SET p.createdAt = timestamp()
            ON MATCH SET p.updatedAt = timestamp()
            RETURN p
            """
            
            self._execute_write(person_query, {"insider_name": insider_name})
            
            # Create Document node for Form 4
            form4_accession = f"{company_ticker}_{insider_name}_{transaction_date}_Form4".replace(" ", "_")
            
            doc_query = """
            MERGE (doc:Document {accession_number: $accession_number})
            ON CREATE SET
                doc.url = $url,
                doc.form = $form,
                doc.filing_date = $filing_date,
                doc.createdAt = timestamp()
            ON MATCH SET
                doc.updatedAt = timestamp()
            RETURN doc
            """
            
            self._execute_write(doc_query, {
                "accession_number": form4_accession,
                "url": filing_url,
                "form": form,
                "filing_date": transaction_date
            })
            
            # Create InsiderTransaction node
            transaction_query = """
            MATCH (c:Company {ticker: $ticker})
            MATCH (p:Person {name: $insider_name})
            MATCH (doc:Document {accession_number: $accession_number})
            
            MERGE (c)-[:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction {transactionDate: $transaction_date, transactionType: $transaction_type})
            SET it.position = $position,
                it.shares = $shares,
                it.price = $price,
                it.value = $value,
                it.remainingShares = $remaining_shares
            
            MERGE (it)-[:MADE_BY]->(p)
            MERGE (it)-[:EXTRACTED_FROM]->(doc)
            RETURN it
            """
            
            result = self._execute_write(transaction_query, {
                "ticker": company_ticker,
                "insider_name": insider_name,
                "accession_number": form4_accession,
                "transaction_date": transaction_date,
                "position": position,
                "transaction_type": transaction_type,
                "shares": shares,
                "price": price,
                "value": value,
                "remaining_shares": remaining_shares
            })
            
            if result:
                print(f"✅ Added insider transaction for {company_ticker} ({insider_name})")
                return result[0]["it"]
            return None
            
        except Exception as e:
            print(f"❌ Error adding insider transaction for {company_ticker}: {e}")
            return None

    def add_section_chunks(
        self,
        company_ticker: str,
        filing_date: date,
        section_name: str,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Batch-create SectionChunk nodes linked to their parent Section node.
        Architecture:
            Filing10K → Section (full text) → [:HAS_CHUNK] → SectionChunk (chunk 0)
                                                           → SectionChunk (chunk 1)
                                                           → SectionChunk (chunk 2)
        Each chunk stores text, metadata, and an embedding vector for
        similarity search via the ``section_chunk_vector_index``.

        Args:
            company_ticker: Company ticker symbol (e.g. "AAPL")
            filing_date: Date of the 10-K filing
            section_name: One of 'risk_factors', 'business_info', 'mda',
                          'legal_proceedings', 'properties'
            chunks: List of dicts with keys:
                    id, title, text, embedding, section_type, subsection,
                    chunk_index, filing_cik, filing_date, section_name

        Returns:
            Number of SectionChunk nodes created/updated
            
        Example Usage:
            # Step 1: Create parent section with full text
            self.add_risk_factors_section(ticker, date, full_text)
            
            # Step 2: Add chunks linked to that section
            self.add_section_chunks(ticker, date, "risk_factors", chunked_data)
        """
        if not chunks:
            return 0

        year = filing_date.year

        try:
            # Prepare chunk data list for UNWIND
            chunk_list = []
            for c in chunks:
                chunk_list.append({
                    "title": c["title"],
                    "text": c["text"],
                    "embedding": c.get("embedding"),
                    "sectionType": c.get("section_type"),
                    "subsection": c.get("subsection"),
                    "chunkIndex": c.get("chunk_index"),
                    "sectionName": c.get("section_name"),
                })

            query = """
            MATCH (c:Company {ticker: $ticker})-[:REPORTS_IN {year: $year}]->(f:Filing10K)
                  -[:HAS_SECTION]->(sec:Section {sectionType: $section_type})
            WITH sec
            UNWIND $chunks AS chunk
            MERGE (sec)-[:HAS_CHUNK]->(sc:Chunk:SectionChunk {text: chunk.text})
            SET sc.title        = chunk.title,
                sc.embedding    = chunk.embedding,
                sc.chunkType    = chunk.sectionType,
                sc.subsection   = chunk.subsection,
                sc.chunkIndex   = chunk.chunkIndex,
                sc.sectionName  = chunk.sectionName
            RETURN count(sc) AS created
            """

            result = self._execute_write(query, {
                "ticker": company_ticker,
                "year": year,
                "section_type": section_name,
                "chunks": chunk_list,
            })

            created = result[0]["created"] if result else 0
            if created:
                print(f"✅ Added {created} SectionChunk nodes for {company_ticker}/{section_name}")
            return created

        except Exception as e:
            print(f"❌ Error adding section chunks for {company_ticker}/{section_name}: {e}")
            logger.error(f"Failed to add section chunks: {e}", exc_info=True)
            return 0
