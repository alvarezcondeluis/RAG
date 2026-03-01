"""
Fund creation and management operations for Neo4j database.
Includes fund creation, managers, financial highlights, and allocations.
"""

from typing import Optional, List, Dict, Any
from datetime import date
import logging
import numpy as np
from ..base import Neo4jDatabaseBase
from src.simple_rag.models.fund import FundData

logger = logging.getLogger(__name__)


class FundCreationOperations(Neo4jDatabaseBase):
    """Fund creation, managers, financial highlights, and allocation operations."""
    
    def create_fund(self, fund: FundData):
        """
        Create a Fund and a time-specific Profile, linking all chunks to the Profile.
        PRESERVES HISTORY: Different report dates = Different profiles/risks.
        """
        try:
            # 1. VALIDATION & DEFAULTS
            if not fund.ticker:
                print(f"⚠️ Skipping fund with no ticker: {fund.name}")
                return None
            
            # Format Date for ID generation (e.g., "2024-09-30")
            date_str = fund.report_date.isoformat() if fund.report_date else "LATEST"
            
            # 2. GENERATE IDs (Time-Aware)
            # We append the date to IDs so we don't overwrite history
            profile_id = f"{fund.ticker}_{date_str}"
            obj_id = f"{fund.ticker}_{date_str}_obj"
            perf_id = f"{fund.ticker}_{date_str}_perf"

            print(f"📊 Creating Profile {date_str} for: {fund.ticker}")

            # 3. PREPARE LISTS FOR CYPHER (Time-Aware IDs)
            # We intentionally modify the chunk ID to include the date here
            risk_list_data = []
            if hasattr(fund, 'risks_chunks') and fund.risks_chunks:
                for chunk in fund.risks_chunks:
                    # Skip chunks without required fields
                    if not hasattr(chunk, 'id') or not hasattr(chunk, 'text'):
                        continue
                    risk_list_data.append({
                        # unique_id: "VTSAX_2024-09-30_risk_0"
                        "id": f"{getattr(chunk, 'id', 'unknown')}_{date_str}", 
                        "title": getattr(chunk, 'title', 'Untitled'),
                        "text": chunk.text,
                        "vector": getattr(chunk, 'embedding', None)
                    })

            strategy_list_data = []
            if hasattr(fund, 'strategies_chunks') and fund.strategies_chunks:
                for chunk in fund.strategies_chunks:
                    # Skip chunks without required fields
                    if not hasattr(chunk, 'id') or not hasattr(chunk, 'text'):
                        continue
                    strategy_list_data.append({
                        "id": f"{getattr(chunk, 'id', 'unknown')}_{date_str}", 
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
                f.costsPer10k = $costs_per_10k,
                f.advisoryFees = $advisory_fees,
                f.numberHoldings = $n_holdings,
                f.expenseRatio = $expense_ratio,
                f.turnoverRate = $turnover_rate,
                f.netAssets = $net_assets,
                f.updatedAt = timestamp()

            // Metadata of the ncsr filing
            MERGE (d:Document {id: $accession_number})
            ON CREATE SET
                d.accession_number = $accession_number,
                d.filing_date = $filing_date,
                d.form = $form,
                d.url = $url
            
            // Link Document to Fund
            MERGE (f)-[:EXTRACTED_FROM]->(d)

            // --- B. TEMPORAL PROFILE NODE (Create new if date differs) ---
            MERGE (prof:Profile {id: $profile_id})
            ON CREATE SET
                prof.summaryProspectus = $summary_prospectus,
                prof.createdAt = timestamp()
            ON MATCH SET
                prof.summaryProspectus = $summary_prospectus,
                prof.updatedAt = timestamp()
            
            // Connect Fund to Profile (History kept via 'date' property on rel)
            MERGE (f)-[rel:DEFINED_BY]->(prof)
            SET rel.date = $report_date
            
            // Link Profile to Summary Prospectus Document (if available)
            FOREACH (_ IN CASE WHEN $sp_accession_number IS NOT NULL THEN [1] ELSE [] END |
                MERGE (sp_doc:Document {id: $sp_accession_number})
                ON CREATE SET
                    sp_doc.accession_number = $sp_accession_number,
                    sp_doc.filing_date = $sp_filing_date,
                    sp_doc.form = $sp_form,
                    sp_doc.url = $sp_url
                MERGE (prof)-[:EXTRACTED_FROM]->(sp_doc)
            )

            // --- C. SINGLE NODES (Objective & Performance) ---
            // Connected to PROFILE, not Fund
            
            FOREACH (_ IN CASE WHEN $objective_text IS NOT NULL THEN [1] ELSE [] END |
                MERGE (obj:Objective {id: $obj_id})
                SET obj.text = $objective_text,
                    obj.embedding = $objective_vector
                MERGE (prof)-[:HAS_OBJECTIVE]->(obj)
            )

            FOREACH (_ IN CASE WHEN $perf_text IS NOT NULL THEN [1] ELSE [] END |
                MERGE (perf:PerformanceCommentary {id: $perf_id})
                SET perf.text = $perf_text,
                    perf.embedding = $perf_vector
                MERGE (prof)-[:HAS_PERFORMANCE_COMMENTARY]->(perf)
            )

            // --- D. CHUNK NODES (Risks & Strategies) ---
            // Connected to PROFILE

            WITH prof, f
            UNWIND $risk_list as r_item
            MERGE (rc:RiskChunk {id: r_item.id})
            SET rc.title = r_item.title,
                rc.text  = r_item.text,
                rc.embedding = r_item.vector
            MERGE (prof)-[:HAS_RISK_CHUNK]->(rc)

            WITH prof, f
            UNWIND $strat_list as s_item
            MERGE (sc:StrategyChunk {id: s_item.id})
            SET sc.title = s_item.title,
                sc.text  = s_item.text,
                sc.embedding = s_item.vector
            MERGE (prof)-[:HAS_STRATEGY]->(sc)
            
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
                # Static
                "ticker": fund.ticker,
                "name": fund.name or "Unknown",
                "trust": fund.registrant or "Unknown",
                "provider": fund.provider or "Unknown",
                "exchange": fund.security_exchange or "N/A",
                "share_class": fund.share_class or "N/A",
                "costs_per_10k": getattr(fund, 'costs_per_10k', 0),
                "advisory_fees": getattr(fund, 'advisory_fees', 0.0),
                "n_holdings": getattr(fund, 'n_holdings', 0),
                
                # Profile (Temporal)
                "profile_id": profile_id,
                "report_date": fund.report_date,
                "net_assets": fund.net_assets or 0.0,
                "expense_ratio": fund.expense_ratio or 0.0,
                "turnover_rate": fund.turnover_rate or 0.0,
                "summary_prospectus": getattr(fund, 'summary_prospectus', ""),
                
                # Single Nodes - Handle missing attributes safely
                "obj_id": obj_id,
                "objective_text": getattr(fund, 'objective', None) if getattr(fund, 'objective', None) not in [None, "N/A", ""] else None,
                "objective_vector": getattr(fund, 'objective_embedding', None),
                
                # NCSR Metadata Information (for Fund)
                "accession_number": metadata.accession_number,
                "filing_date": metadata.filing_date,
                "reporting_date": metadata.reporting_date,
                "url": metadata.url,
                "form": metadata.form,
                
                # Summary Prospectus Metadata (for Profile)
                "sp_accession_number": sp_accession,
                "sp_filing_date": sp_filing_date,
                "sp_form": sp_form,
                "sp_url": sp_url,

                "perf_id": perf_id,
                "perf_text": getattr(fund, 'performance_commentary', None) if getattr(fund, 'performance_commentary', None) not in [None, "N/A", ""] else None,
                "perf_vector": getattr(fund, 'performance_commentary_embedding', None),
                
                # Lists - Already safe since we check fund.risks_chunks and fund.strategies_chunks above
                "risk_list": risk_list_data,
                "strat_list": strategy_list_data
            }

            with self.driver.session() as session:
                result = session.run(query, params)
                return result.single()

        except Exception as e:
            print(f"❌ Error creating fund {fund.ticker}: {e}")
            return None
    
    def add_managers(self, fund_ticker: str, managers: List[Dict[str, Any]]):
        """
        Add managers to a fund by creating Person nodes and MANAGED_BY relationships.
        
        Args:
            fund_ticker: The ticker of the fund to add managers to
            managers: List of manager dictionaries with keys:
                     - name: Manager name (required)
                     - role: Manager role (optional)
                     - since: Start date (optional)
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
                SET r.createdAt = timestamp()
                RETURN p, f, r
                """
                
                params = {
                    "fund_ticker": fund_ticker,
                    "manager_name": manager,
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
        net_assets_value_begining: Optional[float] = None,
        net_assets_value_end: Optional[float] = None,
        net_income_ratio: Optional[float] = None
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
            net_assets_value_begining: Price of one share at period start
            net_assets_value_end: Price of one share at period end
            net_income_ratio: Net investment income ratio (percentage)
        
        Returns:
            The created FinancialHighlights node or None if failed
        """
        try:
            # Ensure year is an integer
            year = int(year)
            highlights_id = f"{fund_ticker}_{year}_highlights"
            
            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            MERGE (fh:FinancialHighlight {id: $highlights_id})
            ON CREATE SET
                fh.turnover = $turnover,
                fh.expenseRatio = $expense_ratio,
                fh.totalReturn = $total_return,
                fh.netAssets = $net_assets,
                fh.netAssetsValueBeginning = $net_assets_value_begining,
                fh.netAssetsValueEnd = $net_assets_value_end,
                fh.netIncomeRatio = $net_income_ratio,
                fh.createdAt = timestamp()
            ON MATCH SET
                fh.turnover = $turnover,
                fh.expenseRatio = $expense_ratio,
                fh.totalReturn = $total_return,
                fh.netAssets = $net_assets,
                fh.netAssetsValueBeginning = $net_assets_value_begining,
                fh.netAssetsValueEnd = $net_assets_value_end,
                fh.netIncomeRatio = $net_income_ratio,
                fh.updatedAt = timestamp()
            MERGE (f)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh)
            SET r.year = $year,
                r.createdAt = timestamp()
            RETURN fh, f, r
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "highlights_id": highlights_id,
                "year": year,
                "turnover": turnover,
                "expense_ratio": expense_ratio,
                "total_return": total_return,
                "net_assets": net_assets,
                "net_assets_value_begining": net_assets_value_begining,
                "net_assets_value_end": net_assets_value_end,
                "net_income_ratio": net_income_ratio,
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
                r.reportDate = $report_date,
                r.updatedAt = timestamp()
            RETURN count(s) as count
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "sectors": sectors_data,
                "report_date": report_date,
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
                rel.reportDate = $report_date,
                rel.updatedAt = timestamp()
            RETURN count(r) as count
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "regions": regions_data,
                "report_date": report_date,
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
