#!/usr/bin/env python3
"""
Quick validation script for Neo4j knowledge graph.
Run this to check if your graph is properly constructed.
"""

from src.simple_rag.database.neo4j import Neo4jDatabase


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def check_connection():
    """Check database connection."""
    print_section("1. DATABASE CONNECTION")
    try:
        db = Neo4jDatabase(auto_start=True)
        print("✅ Connected to Neo4j successfully")
        return db
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return None


def check_schema(db):
    """Check schema constraints and indexes."""
    print_section("2. SCHEMA VALIDATION")
    
    # Check constraints
    query = "SHOW CONSTRAINTS"
    with db.driver.session() as session:
        result = session.run(query)
        constraints = list(result)
        print(f"✅ Found {len(constraints)} constraints")
        
    # Check indexes
    query = "SHOW INDEXES"
    with db.driver.session() as session:
        result = session.run(query)
        indexes = [r for r in result if "token" not in str(r.get("indexProvider", ""))]
        print(f"✅ Found {len(indexes)} user indexes")


def check_fund_data(db):
    """Check fund data completeness."""
    print_section("3. FUND DATA")
    
    queries = {
        "Funds": "MATCH (f:Fund) RETURN count(f) as count",
        "Portfolios": "MATCH (p:Portfolio) RETURN count(p) as count",
        "Holdings": "MATCH (h:Holding) RETURN count(h) as count",
        "Fund Documents": "MATCH (f:Fund)-[:EXTRACTED_FROM]->(d:Document) RETURN count(d) as count",
    }
    
    for name, query in queries.items():
        result = db._execute_query(query)
        count = result[0]["count"]
        status = "✅" if count > 0 else "⚠️ "
        print(f"{status} {name}: {count:,}")


def check_company_data(db):
    """Check company data completeness."""
    print_section("4. COMPANY DATA")
    
    queries = {
        "Companies": "MATCH (c:Company) RETURN count(c) as count",
        "10-K Filings": "MATCH (f:`10KFiling`) RETURN count(f) as count",
        "Section Nodes": "MATCH (s:Section) RETURN count(s) as count",
        "Financial Metrics": "MATCH (fm:FinancialMetric) RETURN count(fm) as count",
        "Segments": "MATCH (seg:Segment) RETURN count(seg) as count",
        "CEOs": "MATCH (c:Company)-[:EMPLOYED_AS_CEO]->(p:Person) RETURN count(p) as count",
        "Insider Transactions": "MATCH (it:InsiderTransaction) RETURN count(it) as count",
    }
    
    for name, query in queries.items():
        result = db._execute_query(query)
        count = result[0]["count"]
        status = "✅" if count > 0 else "⚠️ "
        print(f"{status} {name}: {count:,}")


def check_relationships(db):
    """Check key relationships."""
    print_section("5. FUND-COMPANY RELATIONSHIPS")
    
    # Holdings linked to companies
    query = """
    MATCH (h:Holding)-[:IS_EQUITY_OF]->(c:Company)
    RETURN count(c) as count
    """
    result = db._execute_query(query)
    count = result[0]["count"]
    status = "✅" if count > 0 else "❌"
    print(f"{status} Holdings linked to Companies: {count:,}")
    
    # Complete path from Fund to Company
    query = """
    MATCH (f:Fund)-[:HAS_PORTFOLIO]->(:Portfolio)
          -[:CONTAINS]->(h:Holding)
          -[:IS_EQUITY_OF]->(c:Company)
    RETURN count(DISTINCT c) as count
    """
    result = db._execute_query(query)
    count = result[0]["count"]
    status = "✅" if count > 0 else "❌"
    print(f"{status} Companies accessible from Funds: {count:,}")


def check_data_integrity(db):
    """Check for data integrity issues."""
    print_section("6. DATA INTEGRITY")
    
    # Orphan holdings
    query = """
    MATCH (h:Holding)
    WHERE NOT (h)<-[:CONTAINS]-(:Portfolio)
    RETURN count(h) as count
    """
    result = db._execute_query(query)
    count = result[0]["count"]
    status = "✅" if count == 0 else "❌"
    print(f"{status} Orphan Holdings: {count}")
    
    # Companies without tickers
    query = """
    MATCH (c:Company)
    WHERE c.ticker IS NULL OR c.ticker = ''
    RETURN count(c) as count
    """
    result = db._execute_query(query)
    count = result[0]["count"]
    status = "✅" if count == 0 else "❌"
    print(f"{status} Companies without tickers: {count}")
    
    # Documents without IDs
    query = """
    MATCH (d:Document)
    WHERE d.id IS NULL AND d.accession_number IS NULL
    RETURN count(d) as count
    """
    result = db._execute_query(query)
    count = result[0]["count"]
    status = "✅" if count == 0 else "❌"
    print(f"{status} Documents without IDs: {count}")


def show_sample_data(db):
    """Show sample data from the graph."""
    print_section("7. SAMPLE DATA")
    
    # Sample fund
    print("\n📊 Sample Fund:")
    query = """
    MATCH (f:Fund)
    RETURN f.ticker as ticker, f.name as name, f.netAssets as net_assets
    LIMIT 1
    """
    result = db._execute_query(query)
    if result:
        fund = result[0]
        print(f"  {fund['ticker']}: {fund['name']}")
        if fund['net_assets']:
            print(f"  Net Assets: ${fund['net_assets']:,.0f}")
    
    # Sample company
    print("\n🏢 Sample Company:")
    query = """
    MATCH (c:Company)
    WHERE c.ticker IS NOT NULL
    RETURN c.ticker as ticker, c.name as name, c.cik as cik
    LIMIT 1
    """
    result = db._execute_query(query)
    if result:
        company = result[0]
        print(f"  {company['ticker']}: {company['name']}")
        if company['cik']:
            print(f"  CIK: {company['cik']}")
    
    # Sample cross-domain relationship
    print("\n🔗 Sample Fund → Company Relationship:")
    query = """
    MATCH (f:Fund)-[:HAS_PORTFOLIO]->(:Portfolio)
          -[r:CONTAINS]->(h:Holding)
          -[:IS_EQUITY_OF]->(c:Company)
    RETURN f.ticker as fund,
           c.ticker as company,
           h.name as holding_name,
           r.marketValue as value
    LIMIT 1
    """
    result = db._execute_query(query)
    if result:
        rel = result[0]
        print(f"  Fund {rel['fund']} holds {rel['company']}")
        if rel['value']:
            print(f"  Market Value: ${rel['value']:,.0f}")


def run_validation():
    """Run all validation checks."""
    print("="*80)
    print("  NEO4J KNOWLEDGE GRAPH VALIDATION")
    print("="*80)
    
    # Connect
    db = check_connection()
    if not db:
        return
    
    try:
        # Run checks
        check_schema(db)
        check_fund_data(db)
        check_company_data(db)
        check_relationships(db)
        check_data_integrity(db)
        show_sample_data(db)
        
        # Summary
        print_section("VALIDATION COMPLETE")
        print("✅ All checks completed successfully")
        print("\nYour knowledge graph is ready for queries!")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()


if __name__ == "__main__":
    run_validation()
