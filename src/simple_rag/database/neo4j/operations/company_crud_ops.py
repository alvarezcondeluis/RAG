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
            filing_id = f"{company_ticker}_{filing_date.isoformat()}"
            
            # Create Document node
            doc_query = """
            MERGE (doc:Document {id: $accession_number})
            ON CREATE SET
                doc.accession_number = $accession_number,
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
            MATCH (doc:Document {id: $accession_number})
            
            MERGE (f:`Filing10K` {id: $filing_id})
            ON CREATE SET f.createdAt = timestamp()
            ON MATCH SET f.updatedAt = timestamp()
            
            MERGE (c)-[r:HAS_FILING]->(f)
            SET r.date = $report_period_end
            
            MERGE (f)-[:EXTRACTED_FROM]->(doc)
            RETURN f
            """
            
            result = self._execute_write(filing_query, {
                "ticker": company_ticker,
                "accession_number": accession_number,
                "filing_id": filing_id,
                "report_period_end": report_period_end
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
            filing_id = f"{company_ticker}_{filing_date.isoformat()}"
            section_id = f"{filing_id}_risk_factors"
            
            query = """
            MATCH (f:`Filing10K` {id: $filing_id})
            MERGE (rf:Section:RiskFactor {id: $section_id})
            SET rf.fullText = $full_text,
                rf.updatedAt = timestamp()
            MERGE (f)-[:HAS_RISK_FACTORS]->(rf)
            RETURN rf
            """
            
            result = self._execute_write(query, {
                "filing_id": filing_id,
                "section_id": section_id,
                "full_text": risk_factors_text
            })
            
            if result:
                print(f"✅ Added Risk Factors section for {company_ticker}")
                return result[0]["rf"]
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
            filing_id = f"{company_ticker}_{filing_date.isoformat()}"
            section_id = f"{filing_id}_business_info"
            
            query = """
            MATCH (f:`Filing10K` {id: $filing_id})
            MERGE (bi:Section:BusinessInformation {id: $section_id})
            SET bi.fullText = $full_text,
                bi.updatedAt = timestamp()
            MERGE (f)-[:HAS_BUSINESS_INFORMATION]->(bi)
            RETURN bi
            """
            
            result = self._execute_write(query, {
                "filing_id": filing_id,
                "section_id": section_id,
                "full_text": business_info_text
            })
            
            if result:
                print(f"✅ Added Business Information section for {company_ticker}")
                return result[0]["bi"]
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
            filing_id = f"{company_ticker}_{filing_date.isoformat()}"
            section_id = f"{filing_id}_legal_proceedings"
            
            query = """
            MATCH (f:`Filing10K` {id: $filing_id})
            MERGE (lp:Section:LegalProceeding {id: $section_id})
            SET lp.fullTitleSection = $title,
                lp.fullText = $full_text,
                lp.updatedAt = timestamp()
            MERGE (f)-[:HAS_LEGAL_PROCEEDINGS]->(lp)
            RETURN lp
            """
            
            result = self._execute_write(query, {
                "filing_id": filing_id,
                "section_id": section_id,
                "title": "Item 3 - Legal Proceedings",
                "full_text": legal_proceedings_text
            })
            
            if result:
                print(f"✅ Added Legal Proceedings section for {company_ticker}")
                return result[0]["lp"]
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
            filing_id = f"{company_ticker}_{filing_date.isoformat()}"
            section_id = f"{filing_id}_mda"
            
            query = """
            MATCH (f:`Filing10K` {id: $filing_id})
            MERGE (md:Section:ManagemetDiscussion {id: $section_id})
            SET md.fullText = $full_text,
                md.updatedAt = timestamp()
            MERGE (f)-[:HAS_MANAGEMENT_DISCUSSION]->(md)
            RETURN md
            """
            
            result = self._execute_write(query, {
                "filing_id": filing_id,
                "section_id": section_id,
                "full_text": mda_text
            })
            
            if result:
                print(f"✅ Added MD&A section for {company_ticker}")
                return result[0]["md"]
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
            filing_id = f"{company_ticker}_{filing_date.isoformat()}"
            section_id = f"{filing_id}_properties"
            
            query = """
            MATCH (f:Filing10K {id: $filing_id})
            MERGE (prop:Section:Properties {id: $section_id})
            SET prop.fullText = $full_text,
                prop.updatedAt = timestamp()
            MERGE (f)-[:HAS_PROPERTIES_CHUNK]->(prop)
            RETURN prop
            """
            
            result = self._execute_write(query, {
                "filing_id": filing_id,
                "section_id": section_id,
                "full_text": properties_text
            })
            
            if result:
                print(f"✅ Added Properties section for {company_ticker}")
                return result[0]["prop"]
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
            filing_id = f"{company_ticker}_{filing_date.isoformat()}"
            financials_id = f"{filing_id}_financials_{fiscal_year}"
            
            query = """
            MATCH (f:Filing10K {id: $filing_id})
            MERGE (fin:Section:Financials {id: $financials_id})
            SET fin.incomeStatement = $income_statement,
                fin.balanceSheet = $balance_sheet,
                fin.cashFlow = $cash_flow,
                fin.fiscalYear = $fiscal_year,
                fin.updatedAt = timestamp()
            MERGE (f)-[:HAS_FINACIALS]->(fin)
            RETURN fin
            """
            
            result = self._execute_write(query, {
                "filing_id": filing_id,
                "financials_id": financials_id,
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
            filing_id = f"{company_ticker}_{filing_date.isoformat()}"
            financials_id = f"{filing_id}_financials_{fiscal_year}"
            metric_id = f"{financials_id}_{metric_label}"
            
            # Create FinancialMetric node
            metric_query = """
            MATCH (fin:Section:Financials {id: $financials_id})
            MERGE (fm:FinancialMetric {id: $metric_id})
            SET fm.value = $value,
                fm.label = $label,
                fm.updatedAt = timestamp()
            MERGE (fin)-[:HAS_METRIC]->(fm)
            RETURN fm
            """
            
            result = self._execute_write(metric_query, {
                "financials_id": financials_id,
                "metric_id": metric_id,
                "value": metric_value,
                "label": metric_label
            })
            
            if not result:
                return None
            
            # Create Segment nodes if provided
            if segments:
                for seg in segments:
                    segment_id = f"{metric_id}_{seg['label'].replace(' ', '_')}"
                    
                    segment_query = """
                    MATCH (fm:FinancialMetric {id: $metric_id})
                    MERGE (seg:Segment {id: $segment_id})
                    SET seg.label = $label,
                        seg.value = $value,
                        seg.percentage = $percentage,
                        seg.updatedAt = timestamp()
                    MERGE (fm)-[:HAS_SEGMENT]->(seg)
                    RETURN seg
                    """
                    
                    self._execute_write(segment_query, {
                        "metric_id": metric_id,
                        "segment_id": segment_id,
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
                MERGE (doc:Document {id: $accession_number})
                ON CREATE SET
                    doc.accession_number = $accession_number,
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
                comp_package_id = f"{company_ticker}_comp_{(filing_date or date.today()).isoformat()}"
                
                comp_query = """
                MATCH (c:Company {ticker: $ticker})
                MATCH (p:Person {name: $ceo_name})
                MATCH (doc:Document {id: $accession_number})
                
                MERGE (cp:CompensationPackage {id: $comp_package_id})
                SET cp.totalCompensation = $total_compensation,
                    cp.shareholderReturn = $shareholder_return,
                    cp.date = $date,
                    cp.updatedAt = timestamp()
                
                MERGE (p)-[:RECEIVED_COMPENSATION]->(cp)
                MERGE (c)<-[:AWARDED_BY]-(cp)
                MERGE (cp)-[:DISCLOSED_IN]->(doc)
                RETURN cp
                """
                
                result = self._execute_write(comp_query, {
                    "ticker": company_ticker,
                    "ceo_name": ceo_name,
                    "accession_number": accession_number,
                    "comp_package_id": comp_package_id,
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
            MERGE (doc:Document {id: $accession_number})
            ON CREATE SET
                doc.accession_number = $accession_number,
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
            trade_id = f"{company_ticker}_{insider_name}_{transaction_date}_{transaction_type}".replace(" ", "_")
            
            transaction_query = """
            MATCH (c:Company {ticker: $ticker})
            MATCH (p:Person {name: $insider_name})
            MATCH (doc:Document {id: $accession_number})
            
            MERGE (it:InsiderTransaction {id: $trade_id})
            SET it.position = $position,
                it.transactionType = $transaction_type,
                it.shares = $shares,
                it.price = $price,
                it.value = $value,
                it.remainingShares = $remaining_shares,
                it.updatedAt = timestamp()
            
            MERGE (p)<-[:MADE_BY]-(it)
            MERGE (c)-[:HAS_INSIDER_TRANSACTION]->(it)
            MERGE (it)-[:EXTRACTED_FROM]->(doc)
            RETURN it
            """
            
            result = self._execute_write(transaction_query, {
                "ticker": company_ticker,
                "insider_name": insider_name,
                "accession_number": form4_accession,
                "trade_id": trade_id,
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
