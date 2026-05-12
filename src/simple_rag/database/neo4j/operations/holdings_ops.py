"""
Holdings, charts, and query operations for Neo4j database.
"""

from typing import Optional, List, Dict, Any
from datetime import date
import logging
import numpy as np
import re
from ..base import Neo4jDatabaseBase
from simple_rag.models.asset_categories import ASSET_CATEGORY_MAP

logger = logging.getLogger(__name__)


class HoldingsOperations(Neo4jDatabaseBase):
    """Fund holdings, charts, and general query operations."""

    def create_fund_holdings(self, ticker: str, series_id: str, report_date: Optional[date], holdings_df, nport_metadata: Optional[Dict[str, Any]] = None, verbose: bool = True):
        """
        High-Performance Ingestion for Large Funds (>4k holdings).
        Uses Batching + CREATE logic for maximum speed.
        
        Args:
            ticker: Fund ticker symbol
            series_id: Portfolio series ID
            report_date: Reporting date for the portfolio
            holdings_df: DataFrame containing holdings data
            nport_metadata: Optional metadata dict with keys: accession_number, filing_date, reporting_date, url, form
            verbose: If True, print progress messages during ingestion (default: True)
        """
        
        if holdings_df is None or holdings_df.empty:
            if verbose:
                print(f"⚠️ No holdings found for {series_id}")
            return

        if verbose:
            print(f"💰 Processing {len(holdings_df)} holdings for {series_id}...")

        df = holdings_df.replace({np.nan: None})

        for col in ['shares', 'market_value', 'weight_pct']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: float(x) if x is not None else 0.0)

        # Normalize identifier fields: empty strings → None so unique constraints don't collide
        _EMPTY = {'', 'None', 'nan', 'NaN'}
        for col in ['isin', 'cusip', 'lei', 'ticker']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: None if (x is None or str(x).strip() in _EMPTY) else str(x).strip()
                )

        import hashlib as _hl

        def generate_id(row):
            isin = row.get('isin')
            if isin:
                return isin
            cusip = row.get('cusip')
            if cusip:
                return f"cusip_{cusip}"
            key = f"{row.get('name', '')}|{row.get('ticker', '')}|{row.get('lei', '')}"
            return f"gen_{_hl.md5(key.encode()).hexdigest()[:16]}"

        df['node_id'] = df.apply(generate_id, axis=1)

        def enrich_category(row):
            code = row.get('asset_category')
            cat = ASSET_CATEGORY_MAP.get(code) if code else None
            row['category_type'] = cat.category if cat else None      # "Bonds" / "Equities" / "Alternatives"
            row['category_name'] = cat.name if cat else None
            row['category_subcategory'] = cat.subcategory if cat else None
            return row

        df = df.apply(enrich_category, axis=1)

        all_holdings = df.to_dict('records')
        total_count = len(all_holdings)

        portfolio_id = series_id
        
        setup_query = """
        MATCH (fund:Fund {ticker: $fund_ticker})
        MERGE (port:Portfolio {seriesId: $portfolio_id})
        ON CREATE SET
            port.count = $total_count,
            port.date = $report_date,
            port.createdAt = timestamp()
        MERGE (fund)-[r:HAS_PORTFOLIO]->(port)
       
        // Create Document node and EXTRACTED_FROM relationship if metadata provided
        WITH port
        FOREACH (_ IN CASE WHEN $has_metadata THEN [1] ELSE [] END |
            MERGE (doc:Document {accessionNumber: $accession_number})
            ON CREATE SET
                doc.url = $url,
                doc.type = $form,
                doc.filingDate = $filing_date,
                doc.reportingDate = $reporting_date,
                doc.createdAt = timestamp()
            MERGE (port)-[:EXTRACTED_FROM]->(doc)
        )
        """
        
        # Prepare metadata parameters
        has_metadata = nport_metadata is not None
        metadata_params = {
            "fund_ticker": ticker, 
            "portfolio_id": portfolio_id, 
            "report_date": report_date,
            "total_count": total_count,
            "has_metadata": has_metadata,
            "accession_number": nport_metadata.accession_number if has_metadata else None,
            "url": nport_metadata.url if has_metadata else None,
            "form": nport_metadata.form if has_metadata else None,
            "filing_date": nport_metadata.filing_date if has_metadata else None,
            "reporting_date": nport_metadata.reporting_date if has_metadata else None
        }
        
        self._execute_write(setup_query, metadata_params)

        BATCH_SIZE = 1000
        
        batch_query = """
        MATCH (port:Portfolio {seriesId: $portfolio_id})
        UNWIND $batch as row
        
        MERGE (h:Holding {nodeId: row.node_id})
        ON CREATE SET
            h.name = row.name,
            h.ticker = row.ticker,
            h.cusip = row.cusip,
            h.isin = row.isin,
            h.lei = row.lei,
            h.country = row.country,
            h.sector = row.sector,
            h.category = row.asset_category,
            h.categoryDesc = row.asset_category_desc,
            h.categoryType = row.category_type,
            h.issuerCategory = row.issuer_category,
            h.issuerDesc = row.issuer_category_desc,
            h.createdAt = timestamp()
        ON MATCH SET
            h.name = row.name,
            h.ticker = row.ticker,
            h.cusip = row.cusip,
            h.isin = row.isin,
            h.lei = row.lei,
            h.country = row.country,
            h.sector = row.sector,
            h.category = row.asset_category,
            h.categoryDesc = row.asset_category_desc,
            h.categoryType = row.category_type,
            h.issuerCategory = row.issuer_category,
            h.issuerDesc = row.issuer_category_desc,
            h.updatedAt = timestamp()

        MERGE (port)-[rel:HAS_HOLDING]->(h)
        ON CREATE SET
            rel.shares = row.shares,
            rel.marketValue = row.market_value,
            rel.weight = row.weight_pct,
            rel.currency = row.currency,
            rel.fairValueLevel = row.fair_value_level,
            rel.isRestricted = row.is_restricted,
            rel.payoffProfile = row.payoff_profile
        ON MATCH SET
            rel.shares = row.shares,
            rel.marketValue = row.market_value,
            rel.weight = row.weight_pct,
            rel.currency = row.currency,
            rel.fairValueLevel = row.fair_value_level,
            rel.isRestricted = row.is_restricted,
            rel.payoffProfile = row.payoff_profile

        // AssetCategory node + relationship (skipped when code is unknown)
        FOREACH (_ IN CASE WHEN row.category_type IS NOT NULL THEN [1] ELSE [] END |
            MERGE (cat:AssetCategory {code: row.asset_category})
            ON CREATE SET
                cat.name = row.category_name,
                cat.category = row.category_type,
                cat.subcategory = row.category_subcategory,
                cat.createdAt = timestamp()
            MERGE (h)-[:OF_ASSET_TYPE]->(cat)
        )

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
                    if verbose:
                        print(f"   ⏳ {ticker}: Ingested batch {i} to {i + len(batch)}...")
            
            if verbose:
                print(f"✅ Successfully ingested {total_count} holdings for {ticker}")

        except Exception as e:
            if verbose:
                print(f"❌ Error ingesting holdings for {ticker}: {e}")
            raise
    def add_chart_to_fund(
        self, 
        fund_ticker: str, 
        title: str, 
        category: str, 
        svg_content: str, 
        year: int
    ) -> Optional[Dict[str, Any]]:
        """
        Add a chart (Image node) to a fund with HAS_CHART relationship.
        
        Args:
            fund_ticker: The ticker of the fund to add the chart to
            title: Title of the chart
            category: Category/type of the chart (e.g., 'performance', 'allocation', 'sector')
            svg_content: The SVG content as a string
            year: Year associated with the chart (e.g., report year)
            
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
            ON CREATE SET
                r.year = $year,
                r.createdAt = timestamp()
            ON MATCH SET
                r.year = $year,
                r.updatedAt = timestamp()
            RETURN img, f
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "image_id": image_id,
                "title": title,
                "category": category,
                "svg_content": svg_content,
                "year": year
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
               img.createdAt as created_at,
               r.year as year
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

    def enrich_holdings_categories(self, verbose: bool = True) -> int:
        """
        Backfill AssetCategory nodes and IN_CATEGORY relationships for all
        existing Holding nodes that have a `category` property (the SEC code).

        Safe to re-run — all operations are MERGE-based.  Use this to apply
        category data to holdings that were ingested before this feature was added.

        Returns:
            Number of holdings linked to a category.
        """
        categories_payload = [
            {
                "code": cat.code,
                "name": cat.name,
                "category": cat.category,
                "subcategory": cat.subcategory,
            }
            for cat in ASSET_CATEGORY_MAP.values()
        ]

        query = """
        UNWIND $categories AS cat_data

        MERGE (cat:AssetCategory {code: cat_data.code})
        ON CREATE SET
            cat.name = cat_data.name,
            cat.category = cat_data.category,
            cat.subcategory = cat_data.subcategory,
            cat.createdAt = timestamp()

        WITH cat, cat_data
        MATCH (h:Holding)
        WHERE h.category = cat_data.code

        SET h.categoryType = cat_data.category

        MERGE (h)-[:OF_ASSET_TYPE]->(cat)

        RETURN count(h) AS linked
        """

        try:
            result = self._execute_write(query, {"categories": categories_payload})
            total = sum(r["linked"] for r in result) if result else 0
            if verbose:
                print(f"✅ Linked {total} holdings to AssetCategory nodes")
            return total
        except Exception as e:
            if verbose:
                print(f"❌ Error enriching holding categories: {e}")
            raise
