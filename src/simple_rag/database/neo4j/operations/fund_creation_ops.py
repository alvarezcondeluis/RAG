"""
Fund creation and management operations for Neo4j database.
Includes fund creation, managers, financial highlights, and allocations.
"""

from typing import Optional, List, Dict, Any
import logging
from ..base import Neo4jDatabaseBase
from src.simple_rag.models.fund import FundData

logger = logging.getLogger(__name__)


class FundCreationOperations(Neo4jDatabaseBase):
    """Fund creation, managers, financial highlights, and allocation operations."""
    
    def create_fund(self, fund: FundData):
        """
        Create a Fund and a time-specific Profile, linking all chunks to the Profile.
        
        """
        try:
            # 1. VALIDATION & DEFAULTS
            if not fund.ticker:
                print(f"⚠️ Skipping fund with no ticker: {fund.name}")
                return None
            
            date_str = fund.report_date.isoformat() if fund.report_date else "LATEST"
            print(f"📊 Creating Profile {date_str} for: {fund.ticker}")

            # 2. PREPARE CHUNK LISTS FOR CYPHER
            risk_list_data = []
            if hasattr(fund, 'risks_chunks') and fund.risks_chunks:
                for chunk in fund.risks_chunks:
                    if not hasattr(chunk, 'text'):
                        continue
                    risk_list_data.append({
                        "title": getattr(chunk, 'title', 'Untitled'),
                        "text": chunk.text,
                        "vector": getattr(chunk, 'embedding', None)
                    })

            strategy_list_data = []
            if hasattr(fund, 'strategies_chunks') and fund.strategies_chunks:
                for chunk in fund.strategies_chunks:
                    if not hasattr(chunk, 'text'):
                        continue
                    strategy_list_data.append({
                        "title": getattr(chunk, 'title', 'Untitled'),
                        "text": chunk.text,
                        "vector": getattr(chunk, 'embedding', None)
                    })

            metadata = fund.ncsr_metadata

            # 4. CYPHER QUERY
            query = """
            // --- A. STATIC FUND NODES (Merge ensures we reuse existing) ---
            MERGE (t:Trust {name: $trust})
            MERGE (p:Provider {name: $provider})
            MERGE (f:Fund {ticker: $ticker})
            MERGE (sc:ShareClass {name: $share_class})
            
            // Connect Static Relationships
            MERGE (t)-[:ISSUES]->(f)
            MERGE (p)-[:MANAGES]->(t)
            MERGE (f)-[:HAS_SHARE_CLASS]->(sc)
            
            // Update Static Properties (Name might change rarely)
            SET f.name = $name,
                f.securityExchange = $exchange,
                f.cik = $cik,
                f.updatedAt = timestamp()

            // Metadata of the ncsr filing
            MERGE (d:Document {accession_number: $accession_number})
            ON CREATE SET
                d.filing_date = $filing_date,
                d.form = $form,
                d.url = $url
            
            // Link Document to Fund
            MERGE (f)-[:EXTRACTED_FROM]->(d)

            // --- B. PROFILE NODE (one per Fund, upserted) ---
            MERGE (f)-[rel:DEFINED_BY]->(prof:Profile)
            SET rel.year = datetime($report_date).year,
                prof.summaryProspectus = $summary_prospectus
            
            // Link Profile to Summary Prospectus Document (if available)
            FOREACH (_ IN CASE WHEN $sp_accession_number IS NOT NULL THEN [1] ELSE [] END |
                MERGE (sp_doc:Document {accession_number: $sp_accession_number})
                ON CREATE SET
                    sp_doc.filing_date = $sp_filing_date,
                    sp_doc.form = $sp_form,
                    sp_doc.url = $sp_url
                MERGE (prof)-[:EXTRACTED_FROM]->(sp_doc)
            )

            // --- C. SECTION NODES (Objective & Performance) ---
            // Single-text sections: Section holds the full text + embedding
            
            FOREACH (_ IN CASE WHEN $objective_text IS NOT NULL THEN [1] ELSE [] END |
                MERGE (prof)-[:HAS_SECTION]->(obj_sec:Section:Objective)
                SET obj_sec.title = 'Objective',
                    obj_sec.text = $objective_text,
                    obj_sec.embedding = $objective_vector
            )

            FOREACH (_ IN CASE WHEN $perf_text IS NOT NULL THEN [1] ELSE [] END |
                MERGE (prof)-[:HAS_SECTION]->(perf_sec:Section:PerformanceCommentary)
                SET perf_sec.title = 'Performance Commentary',
                    perf_sec.text = $perf_text,
                    perf_sec.embedding = $perf_vector
            )

            // --- D. SECTION + CHUNKS (Risk & Strategy) ---
            // Section node groups chunks; each Chunk holds text + embedding

            FOREACH (_ IN CASE WHEN size($risk_list) > 0 THEN [1] ELSE [] END |
                MERGE (prof)-[:HAS_SECTION]->(risk_sec:Section:RiskFactor)
                SET risk_sec.title = 'Risk Factors'
                FOREACH (r_item IN $risk_list |
                    MERGE (risk_sec)-[:HAS_CHUNK]->(rc:Chunk {text: r_item.text})
                    SET rc.embedding = r_item.vector
                )
            )

            FOREACH (_ IN CASE WHEN size($strat_list) > 0 THEN [1] ELSE [] END |
                MERGE (prof)-[:HAS_SECTION]->(strat_sec:Section:Strategy)
                SET strat_sec.title = 'Strategy'
                FOREACH (s_item IN $strat_list |
                    MERGE (strat_sec)-[:HAS_CHUNK]->(sc:Chunk {text: s_item.text})
                    SET sc.embedding = s_item.vector
                )
            )
            
            RETURN f.ticker as ticker
            """

            # 5. PARAMETERS
            # Extract summary prospectus metadata if available
            sp_metadata = fund.summary_prospectus_metadata
            sp_accession = sp_metadata.accession_number if sp_metadata else None
            sp_filing_date = sp_metadata.filing_date if sp_metadata else None
            sp_form = sp_metadata.form if sp_metadata else None
            sp_url = sp_metadata.url if sp_metadata else None
            
            params = {
                # Static Fund
                "ticker": fund.ticker,
                "name": fund.name or "Unknown",
                "trust": fund.registrant or "Unknown",
                "provider": fund.provider or "Unknown",
                "exchange": fund.security_exchange or "N/A",
                "share_class": fund.share_class or "N/A",
                "cik": getattr(fund, 'cik', None),
                
                # Profile
                "report_date": fund.report_date,
                "summary_prospectus": getattr(fund, 'summary_prospectus', ""),
                
                # NCSR Document
                "accession_number": metadata.accession_number,
                "filing_date": metadata.filing_date,
                "url": metadata.url,
                "form": metadata.form,
                
                # Summary Prospectus Document
                "sp_accession_number": sp_accession,
                "sp_filing_date": sp_filing_date,
                "sp_form": sp_form,
                "sp_url": sp_url,
                
                # Section data
                "objective_text": getattr(fund, 'objective', None) if getattr(fund, 'objective', None) not in [None, "N/A", ""] else None,
                "objective_vector": getattr(fund, 'objective_embedding', None),
                "perf_text": getattr(fund, 'performance_commentary', None) if getattr(fund, 'performance_commentary', None) not in [None, "N/A", ""] else None,
                "perf_vector": getattr(fund, 'performance_commentary_embedding', None),
                
                # Chunk lists
                "risk_list": risk_list_data,
                "strat_list": strategy_list_data
            }

            with self.driver.session() as session:
                result = session.run(query, params)
                return result.single()

        except Exception as e:
            print(f"❌ Error creating fund {fund.ticker}: {e}")
            return None
    
    def add_managers(self, fund_ticker: str, managers: List[Dict[str, Any]], year: Optional[int] = None):
        """
        Add managers to a fund by creating Person nodes and MANAGED_BY relationships.
        
        Args:
            fund_ticker: The ticker of the fund to add managers to
            managers: List of manager dictionaries with keys:
                     - name: Manager name (required)
                     - role: Manager role (optional)
                     - since: Start date (optional)
            year: Year associated with this management record
        """
        try:
            print(f"👥 Adding {len(managers)} managers to fund {fund_ticker}")
            
            for manager in managers:
                if not manager:
                    print(f"⚠️ Skipping manager without name: {manager}")
                    continue
                
                # Create Person node and link to fund
                query = """
                MATCH (f:Fund {ticker: $fund_ticker})
                MERGE (p:Person {name: $manager_name})
                ON CREATE SET
                    p.createdAt = timestamp()
                ON MATCH SET
                    p.updatedAt = timestamp()
                MERGE (f)-[r:MANAGED_BY]->(p)
                SET r.year = $year
                RETURN p, f, r
                """
                
                params = {
                    "fund_ticker": fund_ticker,
                    "manager_name": manager,
                    "year": year,
                }
                
                result = self._execute_write(query, params)
                if result:
                    print(f"✅ Added manager: {manager} to {fund_ticker}")
                else:
                    print(f"⚠️ Failed to add manager: {manager}")
            
        except Exception as e:
            print(f"❌ Error adding managers to fund {fund_ticker}: {e}")
            logger.error(f"Failed to add managers to fund {fund_ticker}: {e}", exc_info=True)

    def add_financial_highlight(
        self,
        fund_ticker: str,
        year: int,
        turnover: Optional[float] = None,
        expense_ratio: Optional[float] = None,
        total_return: Optional[float] = None,
        net_assets: Optional[float] = None,
        net_assets_value_beginning: Optional[float] = None,
        net_assets_value_end: Optional[float] = None,
        net_income_ratio: Optional[float] = None,
        number_holdings: Optional[int] = None,
        advisory_fees: Optional[float] = None,
        costs_per_10k: Optional[float] = None
    ):
        """
        Add financial highlights to a fund for a specific year.
        
        Args:
            fund_ticker: The ticker of the fund
            year: The year for these financial highlights (e.g., 2024)
            turnover: Portfolio turnover rate (percentage)
            expense_ratio: Total expense ratio (percentage)
            total_return: Total return for the period (percentage)
            net_assets: Total net assets under management (in millions)
            net_assets_value_beginning: Price of one share at period start
            net_assets_value_end: Price of one share at period end
            net_income_ratio: Net investment income ratio (percentage)
            number_holdings: Total number of holdings (integer)
            advisory_fees: Advisory fees (numeric)
            costs_per_10k: Costs per $10,000 invested (numeric)
        
        Returns:
            The created FinancialHighlights node or None if failed
        """
        try:
            # Ensure year is an integer
            year = int(year)
            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            MERGE (f)-[r:HAS_FINANCIAL_HIGHLIGHT {year: $year}]->(fh:FinancialHighlight)
            SET fh.turnover = $turnover,
                fh.expenseRatio = $expense_ratio,
                fh.totalReturn = $total_return,
                fh.netAssets = $net_assets,
                fh.netAssetsValueBeginning = $net_assets_value_beginning,
                fh.netAssetsValueEnd = $net_assets_value_end,
                fh.netIncomeRatio = $net_income_ratio,
                fh.numberHoldings = $number_holdings,
                fh.advisoryFees = $advisory_fees,
                fh.costsPer10k = $costs_per_10k
            RETURN fh, f, r
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "year": year,
                "turnover": turnover,
                "expense_ratio": expense_ratio,
                "total_return": total_return,
                "net_assets": net_assets,
                "net_assets_value_beginning": net_assets_value_beginning,
                "net_assets_value_end": net_assets_value_end,
                "net_income_ratio": net_income_ratio,
                "number_holdings": number_holdings,
                "advisory_fees": advisory_fees,
                "costs_per_10k": costs_per_10k,
            }
            
            result = self._execute_write(query, params)
            if result:
                print(f"✅ Added financial highlights for {fund_ticker} ({year})")
                return result[0]["fh"]
            else:
                print(f"⚠️ Fund {fund_ticker} not found")
                return None
                
        except Exception as e:
            print(f"❌ Error adding financial highlights to {fund_ticker}: {e}")
            logger.error(f"Failed to add financial highlights to {fund_ticker}: {e}", exc_info=True)
            return None

    def add_sector_allocations(
        self,
        sectors_df,
        fund_ticker: str,
        report_date: str
    ):
        """
        Add multiple sector allocations from a DataFrame.
        
        Args:
            sectors_df: DataFrame with 2 columns - first column is sector name, second is weight
            fund_ticker: The ticker of the fund
            report_date: The date of this allocation report (e.g., "2024-12-31")
        
        Returns:
            Number of sectors successfully added
        """
        try:
            if sectors_df is None or sectors_df.empty:
                print(f"⚠️ No sector data provided for {fund_ticker}")
                return 0
            
            cols = sectors_df.columns.tolist()
            if len(cols) < 2:
                print(f"❌ DataFrame must have at least 2 columns (sector name, weight)")
                return 0
            
            sector_col = cols[0]
            weight_col = cols[1]
            
            sectors_data = []
            for _, row in sectors_df.iterrows():
                sector_name = row[sector_col]
                weight = row[weight_col]
                
                if sector_name and weight is not None:
                    sectors_data.append({
                        "sector_name": str(sector_name).strip(),
                        "weight": float(weight)
                    })
            
            if not sectors_data:
                print(f"⚠️ No valid sector data found for {fund_ticker}")
                return 0
            
            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            UNWIND $sectors AS sector
            MERGE (s:Sector {name: sector.sector_name})
            ON CREATE SET
                s.createdAt = timestamp()
            MERGE (f)-[r:HAS_SECTOR_ALLOCATION]->(s)
            SET r.weight = sector.weight,
                r.year = $year
            RETURN count(s) as count
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "sectors": sectors_data,
                "year": int(report_date[:4]) if isinstance(report_date, str) else report_date,
            }
            
            result = self._execute_write(query, params)
            if result:
                count = result[0]["count"]
                print(f"✅ Added {count} sector allocations to {fund_ticker} ({report_date})")
                return count
            else:
                print(f"⚠️ Fund {fund_ticker} not found")
                return 0
                
        except Exception as e:
            print(f"❌ Error adding sector allocations to {fund_ticker}: {e}")
            logger.error(f"Failed to add sector allocations to {fund_ticker}: {e}", exc_info=True)
            return 0

    def add_geographic_allocations(
        self,
        geographic_df,
        fund_ticker: str,
        report_date: str
    ):
        """
        Add multiple geographic allocations from a DataFrame.
        
        Args:
            geographic_df: DataFrame with 2 columns - first column is region name, second is weight
            fund_ticker: The ticker of the fund
            report_date: The date of this allocation report (e.g., "2024-12-31")
        
        Returns:
            Number of regions successfully added
        """
        try:
            if geographic_df is None or geographic_df.empty:
                print(f"⚠️ No geographic data provided for {fund_ticker}")
                return 0
            
            cols = geographic_df.columns.tolist()
            if len(cols) < 2:
                print(f"❌ DataFrame must have at least 2 columns (region name, weight)")
                return 0
            
            region_col = cols[0]
            weight_col = cols[1]
            
            regions_data = []
            for _, row in geographic_df.iterrows():
                region_name = row[region_col]
                weight = row[weight_col]
                
                if region_name and weight is not None:
                    regions_data.append({
                        "region_name": str(region_name).strip(),
                        "weight": float(weight)
                    })
            
            if not regions_data:
                print(f"⚠️ No valid geographic data found for {fund_ticker}")
                return 0
            
            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            UNWIND $regions AS region
            MERGE (r:Region {name: region.region_name})
            ON CREATE SET
                r.createdAt = timestamp()
            MERGE (f)-[rel:HAS_REGION_ALLOCATION]->(r)
            SET rel.weight = region.weight,
                rel.year = $year
            RETURN count(r) as count
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "regions": regions_data,
                "year": int(report_date[:4]) if isinstance(report_date, str) else report_date,
            }
            
            result = self._execute_write(query, params)
            if result:
                count = result[0]["count"]
                print(f"✅ Added {count} geographic allocations to {fund_ticker} ({report_date})")
                return count
            else:
                print(f"⚠️ Fund {fund_ticker} not found")
                return 0
                
        except Exception as e:
            print(f"❌ Error adding geographic allocations to {fund_ticker}: {e}")
            logger.error(f"Failed to add geographic allocations to {fund_ticker}: {e}", exc_info=True)
            return 0

    def add_tables(
        self,
        fund_ticker: str,
        tables: List[Dict[str, Any]],
        year: int
    ) -> int:
        """
        Add table nodes to a fund with HAS_TABLE relationships.

        Args:
            fund_ticker: The ticker of the fund
            tables: List of dicts with keys:
                    - title: Table title/label
                    - content: Table content (HTML, text, or structured data)
            year: Year associated with these tables

        Returns:
            Number of Table nodes created/updated
        """
        try:
            if not tables:
                print(f"⚠️ No table data provided for {fund_ticker}")
                return 0

            tables_data = [
                {
                    "title": t.get("title", "Untitled"),
                    "content": t.get("content", ""),
                }
                for t in tables
                if t.get("content")
            ]

            if not tables_data:
                return 0

            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            UNWIND $tables AS tbl
            MERGE (f)-[r:HAS_TABLE {year: $year}]->(t:Table {title: tbl.title})
            SET t.content = tbl.content
            RETURN count(t) AS count
            """

            result = self._execute_write(query, {
                "fund_ticker": fund_ticker,
                "tables": tables_data,
                "year": year,
            })

            if result:
                count = result[0]["count"]
                print(f"✅ Added {count} tables to {fund_ticker} ({year})")
                return count
            return 0

        except Exception as e:
            print(f"❌ Error adding tables to {fund_ticker}: {e}")
            logger.error(f"Failed to add tables to {fund_ticker}: {e}", exc_info=True)
            return 0

    def get_all_fund_tickers(self) -> List[str]:
        """
        Retrieve all distinct fund tickers efficiently from the database.
        
        Returns:
            List[str]: A list of all fund tickers.
        """
        query = "MATCH (f:Fund) WHERE f.ticker IS NOT NULL RETURN DISTINCT f.ticker AS ticker"
        try:
            results = self._execute_query(query)
            return [record["ticker"] for record in results if "ticker" in record]
        except Exception as e:
            logger.error(f"Failed to retrieve fund tickers: {e}")
            return []

    def get_all_fund_names(self) -> List[str]:
        """
        Retrieve all distinct fund names efficiently from the database.
        
        Returns:
            List[str]: A list of all fund names.
        """
        query = "MATCH (f:Fund) WHERE f.name IS NOT NULL RETURN DISTINCT f.name AS name"
        try:
            results = self._execute_query(query)
            return [record["name"] for record in results if "name" in record]
        except Exception as e:
            logger.error(f"Failed to retrieve fund names: {e}")
            return []
