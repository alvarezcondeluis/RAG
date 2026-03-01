"""
Holdings, charts, and query operations for Neo4j database.
"""

from typing import Optional, List, Dict, Any
from datetime import date
import logging
import numpy as np
from ..base import Neo4jDatabaseBase

logger = logging.getLogger(__name__)


class HoldingsOperations(Neo4jDatabaseBase):
    """Fund holdings, charts, and general query operations."""

    def create_fund_holdings(self, ticker: str, series_id: str, report_date: Optional[date], holdings_df):
        """
        High-Performance Ingestion for Large Funds (>4k holdings).
        Uses Batching + CREATE logic for maximum speed.
        """
        if series_id is None:
            print(f"⚠️ No series_id provided for {ticker}, skipping portfolio and holdings creation")
            return
        
        if holdings_df is None or holdings_df.empty:
            print(f"⚠️ No holdings found for {series_id}")
            return

        print(f"💰 Processing {len(holdings_df)} holdings for {series_id}...")

        # 1. CLEANING & ID GENERATION (Python Side)
        
        df = holdings_df.replace({np.nan: None})
        
        # Ensure numerics
        for col in ['shares', 'market_value', 'weight_pct']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: float(x) if x is not None else 0.0)

        # ID Generation using ISIN only
        def generate_id(row):
            return str(row.get('isin', 'UNKNOWN')).strip()

        df['node_id'] = df.apply(generate_id, axis=1)
        
        # Convert to list
        all_holdings = df.to_dict('records')
        total_count = len(all_holdings)

        # 2. SETUP PARENT NODES (One-time setup for the Fund/Portfolio)
        # We do this OUTSIDE the batch loop so we don't repeat it
        
        portfolio_id = series_id
        
        setup_query = """
        MATCH (fund:Fund {ticker: $fund_ticker})
        MERGE (port:Portfolio {id: $portfolio_id})
        ON CREATE SET
            port.ticker = $fund_ticker,
            port.count = $total_count,
            port.createdAt = timestamp()
        MERGE (fund)-[r:HAS_PORTFOLIO]->(port)
        ON CREATE SET
            r.date = $report_date,
            r.createdAt = timestamp()
        ON MATCH SET
            r.date = $report_date,
            r.updatedAt = timestamp()
        """
        self._execute_write(setup_query, {
            "fund_ticker": ticker, 
            "portfolio_id": portfolio_id, 
            "report_date": report_date,
            "total_count": total_count
        })

        # 3. BATCHED INSERTION (The Speed Fix)
        # We process holdings in chunks of 1,000 to prevent memory overload
        BATCH_SIZE = 1000
        
        batch_query = """
        MATCH (port:Portfolio {id: $portfolio_id})
        UNWIND $batch as row
        
        // A. Create/Update Holding Node
        MERGE (h:Holding {id: row.node_id})
        ON CREATE SET
            h.name = row.name,
            h.ticker = row.ticker,
            h.cusip = row.cusip,
            h.isin = row.isin,
            h.lei = row.lei,
            h.country = row.country,
            h.sector = row.sector,
            h.assetCategory = row.asset_category,
            h.assetDesc = row.asset_category_desc,
            h.issuerCategory = row.issuer_category,
            h.issuerDesc = row.issuer_category_desc,
            h.createdAt = timestamp()

        // B. Link Portfolio to Holding
        MERGE (port)-[rel:HAS_HOLDING]->(h)
        ON CREATE SET
            rel.shares = row.shares,
            rel.marketValue = row.market_value,
            rel.weight = row.weight_pct,
            rel.currency = row.currency,
            rel.fairValueLevel = row.fair_value_level,
            rel.isRestricted = row.is_restricted,
            rel.payoffProfile = row.payoff_profile
        
        // C. Link Holding to Company (if ticker exists)
        // This creates the bridge between Fund holdings and Company data
        FOREACH (_ IN CASE WHEN row.ticker IS NOT NULL AND row.ticker <> '' THEN [1] ELSE [] END |
            MERGE (c:Company {ticker: row.ticker})
            ON CREATE SET
                c.name = row.name,
                c.createdAt = timestamp()
            MERGE (h)-[:REPRESENTS]->(c)
        )
        """

        try:
            with self.driver.session() as session:
                for i in range(0, total_count, BATCH_SIZE):
                    batch = all_holdings[i : i + BATCH_SIZE]
                    session.run(batch_query, {
                        "portfolio_id": portfolio_id,
                        "batch": batch
                    })
                    print(f"   ⏳ {ticker}: Ingested batch {i} to {i + len(batch)}...")
            
            print(f"✅ Successfully ingested {total_count} holdings for {ticker}")

        except Exception as e:
            print(f"❌ Error ingesting holdings for {ticker}: {e}")

    def add_chart_to_fund(
        self, 
        fund_ticker: str, 
        title: str, 
        category: str, 
        svg_content: str, 
        date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Add a chart (Image node) to a fund with HAS_CHART relationship.
        
        Args:
            fund_ticker: The ticker of the fund to add the chart to
            title: Title of the chart
            category: Category/type of the chart (e.g., 'performance', 'allocation', 'sector')
            svg_content: The SVG content as a string
            date: Date associated with the chart (e.g., report date)
            
        Returns:
            Dictionary with created Image node or None if fund not found
        """
        try:
            # Generate unique ID for the image using hash of content
            import hashlib
            content_hash = hashlib.md5(svg_content.encode()).hexdigest()[:12]
            image_id = f"{fund_ticker}_{category}_{content_hash}"
            
            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            MERGE (img:Image {id: $image_id})
            ON CREATE SET
                img.title = $title,
                img.category = $category,
                img.svg = $svg_content,
                img.createdAt = timestamp()
            ON MATCH SET
                img.title = $title,
                img.category = $category,
                img.svg = $svg_content,
                img.updatedAt = timestamp()
            MERGE (f)-[r:HAS_CHART]->(img)
            RETURN img, f
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "image_id": image_id,
                "title": title,
                "category": category,
                "svg_content": svg_content
            }
            
            result = self._execute_write(query, params)
            
            if result:
                print(f"✅ Added chart '{title}' ({category}) to fund {fund_ticker}")
                return result[0]
            else:
                print(f"⚠️ Fund {fund_ticker} not found")
                return None
                
        except Exception as e:
            print(f"❌ Error adding chart to fund {fund_ticker}: {e}")
            logger.error(f"Failed to add chart to fund {fund_ticker}: {e}", exc_info=True)
            return None
    
    def get_fund_charts(self, fund_ticker: str) -> List[Dict[str, Any]]:
        """
        Get all charts for a specific fund.
        
        Args:
            fund_ticker: The ticker of the fund
            
        Returns:
            List of chart dictionaries with image data and relationship date
        """
        query = """
        MATCH (f:Fund {ticker: $fund_ticker})-[r:HAS_CHART]->(img:Image)
        RETURN img.id as id,
               img.title as title,
               img.category as category,
               img.svg as svg_content,
               img.createdAt as created_at
        """
        
        result = self._execute_query(query, {"fund_ticker": fund_ticker})
        return result
    
    def get_all_company_tickers(self) -> set:
        """
        Get all unique company tickers from the database.
        
        Returns:
            Set of unique company ticker strings
            
        Example:
            db = Neo4jDatabase()
            tickers = db.get_all_company_tickers()
            print(f"Found {len(tickers)} companies: {tickers}")
        """
        query = """
        MATCH (c:Holding)
        WHERE c.ticker IS NOT NULL
        RETURN DISTINCT c.ticker as ticker
        ORDER BY c.ticker
        """
        
        result = self._execute_query(query)
        tickers = {record["ticker"] for record in result if record.get("ticker")}
        
        logger.info(f"Found {len(tickers)} unique company tickers in database")
        return tickers

    def link_holding_to_company(
        self,
        holding_id: str,
        company_ticker: str
    ) -> Optional[Dict[str, Any]]:
        """
        Manually link a Holding node to a Company node.
        
        This is useful if you want to create the link after the fact,
        though the create_fund_holdings() method now does this automatically.
        
        Args:
            holding_id: ID of the Holding node
            company_ticker: Ticker of the Company
            
        Returns:
            The relationship or None if failed
        """
        try:
            query = """
            MATCH (h:Holding {id: $holding_id})
            MERGE (c:Company {ticker: $company_ticker})
            ON CREATE SET
                c.name = h.name,
                c.createdAt = timestamp()
            MERGE (h)-[r:IS_EQUITY_OF]->(c)
            RETURN r
            """
            
            result = self._execute_write(query, {
                "holding_id": holding_id,
                "company_ticker": company_ticker
            })
            
            if result:
                print(f"✅ Linked holding {holding_id} to company {company_ticker}")
                return result[0]["r"]
            return None
            
        except Exception as e:
            print(f"❌ Error linking holding to company: {e}")
            return None
