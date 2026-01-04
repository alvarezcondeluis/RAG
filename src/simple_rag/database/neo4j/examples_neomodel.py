"""
Example usage patterns for Neo4j fund database using neomodel ORM.

This module demonstrates how to:
1. Initialize and connect to Neo4j with neomodel
2. Create the graph schema
3. Ingest fund data from FundData objects
4. Query and retrieve fund information using ORM patterns
"""

from typing import List, Optional
import pandas as pd
from datetime import datetime

from .neo4j import Neo4jDatabase
from .models import (
    Provider, Trust, Fund, Asset, Person, Sector, CreditRating,
    Risks, TrailingPerformance, AnnualPerformance, InvestmentStrategy,
    Profile, Derivative
)
from ...models.fund import FundData, PortfolioHolding


def initialize_database() -> Neo4jDatabase:
    """
    Initialize Neo4j database with auto-start and create schema.
    
    Returns:
        Neo4jDatabase instance ready to use
    """
    # Auto-start Neo4j via Docker if not running
    db = Neo4jDatabase(auto_start=True)
    
    # Install all labels and constraints
    db.install_labels()
    
    return db


def ingest_fund_basic_info(db: Neo4jDatabase, fund_data: FundData, provider_name: str, trust_name: str) -> Fund:
    """
    Ingest basic fund information into Neo4j using neomodel.
    
    Args:
        db: Neo4jDatabase instance
        fund_data: FundData object with fund information
        provider_name: Provider name (e.g., "Vanguard", "BlackRock")
        trust_name: Trust name (e.g., "Vanguard Group")
    
    Returns:
        Created Fund node
    """
    # Create Provider and Trust nodes
    provider = db.get_or_create_provider(provider_name)
    trust = db.get_or_create_trust(trust_name)
    db.link_provider_trust(provider, trust)
    
    # Create Fund node
    fund = db.create_fund(
        ticker=fund_data.ticker,
        name=fund_data.name,
        expense_ratio=fund_data.expense_ratio,
        share_class=fund_data.share_class,
        net_assets=fund_data.net_assets,
        turnover_rate=fund_data.turnover_rate,
        advisory_fee=fund_data.advisory_fees,
        holdings=fund_data.n_holdings,
        costs_10k=fund_data.costs_per_10k,
    )
    
    db.link_trust_fund(trust, fund)
    
    print(f"✓ Ingested basic info for {fund_data.ticker}")
    return fund


def ingest_fund_managers(db: Neo4jDatabase, fund: Fund, managers: List[str]):
    """
    Ingest fund managers and create relationships.
    
    Args:
        db: Neo4jDatabase instance
        fund: Fund node
        managers: List of manager names
    """
    for manager_name in managers:
        person = db.get_or_create_person(manager_name)
        db.link_fund_manager(fund, person, role="Portfolio Manager")
    
    print(f"✓ Ingested {len(managers)} managers for {fund.ticker}")


def ingest_fund_holdings(
    db: Neo4jDatabase,
    fund: Fund,
    holdings: List[PortfolioHolding],
    date: Optional[str] = None
):
    """
    Ingest portfolio holdings as Asset nodes with relationships.
    
    Args:
        db: Neo4jDatabase instance
        fund: Fund node
        holdings: List of PortfolioHolding objects
        date: Holdings date (optional)
    """
    date = date or datetime.now().strftime("%Y-%m-%d")
    
    for holding in holdings:
        # Create Asset node
        asset = db.get_or_create_asset(
            name=holding.name,
            ticker=holding.ticker,
            category=holding.asset_category,
            issuer_category=holding.issuer_category,
            country=holding.country,
            sector=holding.sector,
            currency=holding.currency,
        )
        
        # Create HAS_ASSET relationship
        db.link_fund_asset(
            fund=fund,
            asset=asset,
            weight=str(holding.weight_pct),
            n_shares=str(holding.shares),
            market_value=str(holding.market_value),
            date=date,
        )
    
    print(f"✓ Ingested {len(holdings)} holdings for {fund.ticker}")


def ingest_fund_sectors(
    db: Neo4jDatabase,
    fund: Fund,
    sector_allocation: pd.DataFrame
):
    """
    Ingest sector allocation data.
    
    Args:
        db: Neo4jDatabase instance
        fund: Fund node
        sector_allocation: DataFrame with 'sector' and 'weight' columns
    """
    if sector_allocation is None or sector_allocation.empty:
        return
    
    for _, row in sector_allocation.iterrows():
        sector_name = row.get('sector') or row.get('Sector')
        weight = row.get('weight') or row.get('Weight') or row.get('%')
        
        if not sector_name:
            continue
        
        sector = db.get_or_create_sector(str(sector_name))
        db.link_fund_sector(fund, sector, weight=str(weight))
    
    print(f"✓ Ingested sector allocation for {fund.ticker}")


def ingest_fund_credit_ratings(
    db: Neo4jDatabase,
    fund: Fund,
    credit_rating: pd.DataFrame
):
    """
    Ingest credit rating distribution.
    
    Args:
        db: Neo4jDatabase instance
        fund: Fund node
        credit_rating: DataFrame with 'rating' and 'weight' columns
    """
    if credit_rating is None or credit_rating.empty:
        return
    
    for _, row in credit_rating.iterrows():
        rating_name = row.get('rating') or row.get('Rating')
        weight = row.get('weight') or row.get('Weight') or row.get('%')
        
        if not rating_name:
            continue
        
        rating = db.get_or_create_credit_rating(str(rating_name))
        db.link_fund_credit_rating(fund, rating, weight=str(weight))
    
    print(f"✓ Ingested credit ratings for {fund.ticker}")


def ingest_fund_risks(
    db: Neo4jDatabase,
    fund: Fund,
    risks_content: str,
    date: Optional[str] = None
):
    """
    Ingest fund risks information.
    
    Args:
        db: Neo4jDatabase instance
        fund: Fund node
        risks_content: Risk description text
        date: Report date (optional)
    """
    if not risks_content or risks_content == "N/A":
        return
    
    db.create_risks_node(fund, risks_content, date)
    print(f"✓ Ingested risks for {fund.ticker}")


def ingest_fund_performance(
    db: Neo4jDatabase,
    fund: Fund,
    performance_dict: dict,
    date: Optional[str] = None
):
    """
    Ingest trailing performance data.
    
    Args:
        db: Neo4jDatabase instance
        fund: Fund node
        performance_dict: Dict with return_1y, return_5y, return_10y, return_inception
        date: Performance date (optional)
    """
    if not performance_dict:
        return
    
    db.create_performance_node(
        fund=fund,
        return_1y=performance_dict.get('return_1y'),
        return_5y=performance_dict.get('return_5y'),
        return_10y=performance_dict.get('return_10y'),
        return_inception=performance_dict.get('return_inception'),
        date=date
    )
    
    print(f"✓ Ingested performance for {fund.ticker}")


def ingest_fund_strategy(
    db: Neo4jDatabase,
    fund: Fund,
    objective: str,
    strategies: str,
    date: Optional[str] = None
):
    """
    Ingest investment strategy and summary prospectus.
    
    Args:
        db: Neo4jDatabase instance
        fund: Fund node
        objective: Investment objective text
        strategies: Investment strategies text
        date: Report date (optional)
    """
    if not objective and not strategies:
        return
    
    full_text = f"## Objective\n{objective}\n\n## Strategies\n{strategies}"
    
    db.create_strategy_node(
        fund=fund,
        full_text_md=full_text,
        objective=objective,
        strategies=strategies,
        date=date
    )
    
    print(f"✓ Ingested strategy for {fund.ticker}")


def ingest_complete_fund(
    db: Neo4jDatabase,
    fund_data: FundData,
    provider_name: str,
    trust_name: str,
    holdings: Optional[List[PortfolioHolding]] = None
):
    """
    Complete ingestion of all fund data into Neo4j.
    
    Args:
        db: Neo4jDatabase instance
        fund_data: FundData object with all fund information
        provider_name: Provider name
        trust_name: Trust name
        holdings: Optional list of PortfolioHolding objects
    """
    print(f"\n{'='*60}")
    print(f"Ingesting fund: {fund_data.ticker} - {fund_data.name}")
    print(f"{'='*60}")
    
    # 1. Basic info
    fund = ingest_fund_basic_info(db, fund_data, provider_name, trust_name)
    
    # 2. Managers
    if fund_data.managers:
        ingest_fund_managers(db, fund, fund_data.managers)
    
    # 3. Holdings
    if holdings:
        ingest_fund_holdings(db, fund, holdings, fund_data.report_date)
    
    # 4. Sector allocation
    if fund_data.sector_allocation is not None:
        ingest_fund_sectors(db, fund, fund_data.sector_allocation)
    
    # 5. Credit ratings
    if fund_data.credit_rating is not None:
        ingest_fund_credit_ratings(db, fund, fund_data.credit_rating)
    
    # 6. Risks
    if fund_data.risks and fund_data.risks != "N/A":
        ingest_fund_risks(db, fund, fund_data.risks, fund_data.report_date)
    
    # 7. Performance
    if fund_data.performance:
        for period, perf_snapshot in fund_data.performance.items():
            perf_dict = {
                'return_1y': perf_snapshot.return_1y,
                'return_5y': perf_snapshot.return_5y,
                'return_10y': perf_snapshot.return_10y,
                'return_inception': perf_snapshot.return_inception,
            }
            ingest_fund_performance(db, fund, perf_dict, fund_data.report_date)
    
    # 8. Strategy
    if fund_data.objective or fund_data.strategies:
        ingest_fund_strategy(
            db,
            fund,
            fund_data.objective or "",
            fund_data.strategies or "",
            fund_data.report_date
        )
    
    print(f"\n✅ Complete ingestion finished for {fund_data.ticker}\n")


# ==================== QUERY EXAMPLES ====================

def example_queries(db: Neo4jDatabase):
    """Demonstrate various query patterns using neomodel."""
    
    print("\n" + "="*60)
    print("QUERY EXAMPLES (neomodel)")
    print("="*60)
    
    # 1. Get all funds
    print("\n1. All funds:")
    funds = db.get_all_funds()
    for fund in funds[:5]:  # Show first 5
        print(f"  - {fund.ticker}: {fund.name}")
    
    # 2. Get specific fund
    print("\n2. Specific fund (VTI):")
    fund = db.get_fund_by_ticker("VTI")
    if fund:
        print(f"  Name: {fund.name}")
        print(f"  Expense Ratio: {fund.expense_ratio or 'N/A'}")
    
    # 3. Get fund with holdings
    print("\n3. Fund with holdings:")
    result = db.get_fund_with_holdings("VTI")
    if result:
        print(f"  Fund: {result['fund'].name}")
        print(f"  Holdings count: {len(result['holdings'])}")
        if result['holdings']:
            print(f"  Top holding: {result['holdings'][0]['asset'].name}")
    
    # 4. Search by name
    print("\n4. Search funds by name (Vanguard):")
    funds = db.search_funds_by_name("Vanguard")
    print(f"  Found {len(funds)} funds")
    
    # 5. Get funds by provider
    print("\n5. Funds by provider (Vanguard):")
    funds = db.get_funds_by_provider("Vanguard")
    print(f"  Found {len(funds)} funds")
    
    # 6. Direct neomodel queries
    print("\n6. Direct neomodel query - funds with expense ratio < 0.1%:")
    low_cost_funds = Fund.nodes.filter(expense_ratio__icontains="0.0").order_by('ticker')
    for fund in list(low_cost_funds)[:5]:
        print(f"  - {fund.ticker}: {fund.expense_ratio}")


# ==================== MAIN USAGE EXAMPLE ====================

def main_example():
    """Complete example workflow using neomodel."""
    
    # 1. Initialize database
    print("Initializing Neo4j database with neomodel...")
    db = initialize_database()
    
    # 2. Example: Create a simple fund structure
    print("\n" + "="*60)
    print("Creating example fund structure...")
    print("="*60)
    
    provider = db.get_or_create_provider("Vanguard")
    trust = db.get_or_create_trust("Vanguard Group")
    db.link_provider_trust(provider, trust)
    
    fund = db.create_fund(
        ticker="VTI",
        name="Vanguard Total Stock Market ETF",
        expense_ratio="0.03%",
        net_assets="$1.3T"
    )
    db.link_trust_fund(trust, fund)
    
    # Add a manager
    manager = db.get_or_create_person("John Doe")
    db.link_fund_manager(fund, manager, role="Portfolio Manager")
    
    print(f"✓ Created fund: {fund.ticker}")
    
    # 3. Run example queries
    example_queries(db)
    
    # 4. Close connection
    db.close()
    print("\n✅ Database connection closed")


if __name__ == "__main__":
    main_example()
