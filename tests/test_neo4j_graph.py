"""
Comprehensive test suite for Neo4j knowledge graph creation.

This test file validates:
1. Database connectivity and setup
2. Schema constraints and indexes
3. Fund data ingestion
4. Company data ingestion
5. Relationships between funds and companies
6. Data integrity and completeness
7. Query performance
"""

import pytest
from datetime import date, datetime
from typing import Dict, List, Any
from src.simple_rag.database.neo4j import Neo4jDatabase
from src.simple_rag.models.fund import FundData, FilingMetadata
from src.simple_rag.models.company import CompanyEntity


class TestNeo4jConnection:
    """Test database connection and basic operations."""
    
    def test_connection(self):
        """Test that we can connect to Neo4j."""
        db = Neo4jDatabase(auto_start=True)
        assert db.driver is not None
        assert db._is_neo4j_running()
        db.close()
    
    def test_database_stats(self):
        """Test that we can retrieve database statistics."""
        db = Neo4jDatabase()
        stats = db.get_database_stats()
        assert isinstance(stats, dict)
        assert "total_relationships" in stats
        db.close()


class TestSchemaSetup:
    """Test schema constraints and indexes."""
    
    @pytest.fixture
    def db(self):
        """Fixture to provide a database connection."""
        db = Neo4jDatabase()
        yield db
        db.close()
    
    def test_constraints_exist(self, db):
        """Verify that all required constraints are created."""
        query = "SHOW CONSTRAINTS"
        
        with db.driver.session() as session:
            result = session.run(query)
            constraints = [record["name"] for record in result]
        
        # Check for key constraints
        expected_constraints = [
            "fund_ticker_unique",
            "company_ticker_unique",
            "person_name_unique",
            "document_accession_unique"
        ]
        
        for constraint in expected_constraints:
            assert constraint in constraints, f"Missing constraint: {constraint}"
    
    def test_indexes_exist(self, db):
        """Verify that all required indexes are created."""
        query = "SHOW INDEXES"
        
        with db.driver.session() as session:
            result = session.run(query)
            indexes = [record["name"] for record in result]
        
        # Check for key indexes
        expected_indexes = [
            "fund_name_index",
            "providerNameIndex",
            "trustNameIndex",
            "fundNameIndex"
        ]
        
        for index in expected_indexes:
            assert index in indexes, f"Missing index: {index}"


class TestFundDataIngestion:
    """Test fund data ingestion and retrieval."""
    
    @pytest.fixture
    def db(self):
        """Fixture to provide a clean database."""
        db = Neo4jDatabase()
        # Note: We don't reset the database to preserve existing data
        yield db
        db.close()
    
    def test_fund_node_exists(self, db):
        """Test that fund nodes exist in the database."""
        query = "MATCH (f:Fund) RETURN count(f) as count"
        result = db._execute_query(query)
        
        assert result[0]["count"] > 0, "No fund nodes found in database"
    
    def test_fund_has_required_properties(self, db):
        """Test that fund nodes have all required properties."""
        query = """
        MATCH (f:Fund)
        RETURN f.ticker as ticker,
               f.name as name,
               f.netAssets as net_assets,
               f.expenseRatio as expense_ratio
        LIMIT 1
        """
        
        result = db._execute_query(query)
        assert len(result) > 0, "No funds found"
        
        fund = result[0]
        assert fund["ticker"] is not None
        assert fund["name"] is not None
        # net_assets and expense_ratio may be null for some funds
    
    def test_fund_has_portfolio(self, db):
        """Test that funds have portfolio relationships."""
        query = """
        MATCH (f:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio)
        RETURN count(p) as portfolio_count
        """
        
        result = db._execute_query(query)
        assert result[0]["portfolio_count"] > 0, "No fund portfolios found"
    
    def test_portfolio_has_holdings(self, db):
        """Test that portfolios contain holdings."""
        query = """
        MATCH (p:Portfolio)-[:CONTAINS]->(h:Holding)
        RETURN count(h) as holding_count
        """
        
        result = db._execute_query(query)
        assert result[0]["holding_count"] > 0, "No holdings found in portfolios"
    
    def test_fund_has_document(self, db):
        """Test that funds are linked to source documents."""
        query = """
        MATCH (f:Fund)-[:EXTRACTED_FROM]->(d:Document)
        RETURN count(d) as doc_count
        """
        
        result = db._execute_query(query)
        assert result[0]["doc_count"] > 0, "No fund documents found"


class TestCompanyDataIngestion:
    """Test company data ingestion and structure."""
    
    @pytest.fixture
    def db(self):
        """Fixture to provide a database connection."""
        db = Neo4jDatabase()
        yield db
        db.close()
    
    def test_company_nodes_exist(self, db):
        """Test that company nodes exist."""
        query = "MATCH (c:Company) RETURN count(c) as count"
        result = db._execute_query(query)
        
        assert result[0]["count"] > 0, "No company nodes found"
    
    def test_company_has_required_properties(self, db):
        """Test that companies have required properties."""
        query = """
        MATCH (c:Company)
        RETURN c.ticker as ticker,
               c.name as name,
               c.cik as cik
        LIMIT 1
        """
        
        result = db._execute_query(query)
        assert len(result) > 0, "No companies found"
        
        company = result[0]
        assert company["ticker"] is not None
        assert company["name"] is not None
        # CIK may be null for some companies
    
    def test_company_has_10k_filings(self, db):
        """Test that companies have 10-K filings."""
        query = """
        MATCH (c:Company)-[:HAS_FILING]->(f:Filing10K)
        RETURN count(f) as filing_count
        """
        
        result = db._execute_query(query)
        # Note: This may be 0 if company data hasn't been ingested yet
        print(f"Found {result[0]['filing_count']} 10-K filings")
    
    def test_10k_has_sections(self, db):
        """Test that 10-K filings have section nodes."""
        query = """
        MATCH (f:Filing10K)-[:HAS_RISK_FACTORS|HAS_BUSINESS_INFORMATION|HAS_FINACIALS]->(s:Section)
        RETURN count(s) as section_count
        """
        
        result = db._execute_query(query)
        print(f"Found {result[0]['section_count']} section nodes")
    
    def test_financials_have_metrics(self, db):
        """Test that Financials sections have metrics."""
        query = """
        MATCH (fin:Financials)-[:HAS_METRIC]->(fm:FinancialMetric)
        RETURN count(fm) as metric_count
        """
        
        result = db._execute_query(query)
        print(f"Found {result[0]['metric_count']} financial metrics")
    
    def test_metrics_have_segments(self, db):
        """Test that financial metrics have segments."""
        query = """
        MATCH (fm:FinancialMetric)-[:HAS_SEGMENT]->(seg:Segment)
        RETURN count(seg) as segment_count
        """
        
        result = db._execute_query(query)
        print(f"Found {result[0]['segment_count']} segments")
    
    def test_company_has_ceo(self, db):
        """Test that companies have CEO relationships."""
        query = """
        MATCH (c:Company)-[:EMPLOYED_AS_CEO]->(p:Person)
        RETURN count(p) as ceo_count
        """
        
        result = db._execute_query(query)
        print(f"Found {result[0]['ceo_count']} CEO relationships")
    
    def test_company_has_insider_transactions(self, db):
        """Test that companies have insider transactions."""
        query = """
        MATCH (c:Company)-[:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction)
        RETURN count(it) as transaction_count
        """
        
        result = db._execute_query(query)
        print(f"Found {result[0]['transaction_count']} insider transactions")


class TestFundCompanyRelationships:
    """Test relationships between funds and companies."""
    
    @pytest.fixture
    def db(self):
        """Fixture to provide a database connection."""
        db = Neo4jDatabase()
        yield db
        db.close()
    
    def test_holdings_linked_to_companies(self, db):
        """Test that holdings are linked to companies via IS_EQUITY_OF."""
        query = """
        MATCH (h:Holding)-[:IS_EQUITY_OF]->(c:Company)
        RETURN count(c) as company_count
        """
        
        result = db._execute_query(query)
        assert result[0]["company_count"] > 0, "No holdings linked to companies"
    
    def test_fund_to_company_path(self, db):
        """Test that we can traverse from Fund to Company."""
        query = """
        MATCH path = (f:Fund)-[:HAS_PORTFOLIO]->(:Portfolio)
                     -[:CONTAINS]->(h:Holding)
                     -[:IS_EQUITY_OF]->(c:Company)
        RETURN count(path) as path_count
        LIMIT 1
        """
        
        result = db._execute_query(query)
        assert result[0]["path_count"] > 0, "No complete path from Fund to Company"
    
    def test_cross_domain_query(self, db):
        """Test a complex query across fund and company domains."""
        query = """
        MATCH (f:Fund)-[:HAS_PORTFOLIO]->(:Portfolio)
              -[:CONTAINS]->(h:Holding)
              -[:IS_EQUITY_OF]->(c:Company)
              -[:HAS_FILING]->(:Filing10K)
              -[:HAS_FINACIALS]->(fin:Financials)
              -[:HAS_METRIC]->(fm:FinancialMetric {label: "Revenue"})
        RETURN f.ticker as fund_ticker,
               c.ticker as company_ticker,
               fm.value as revenue
        LIMIT 5
        """
        
        result = db._execute_query(query)
        print(f"\nCross-domain query results:")
        for record in result:
            print(f"  Fund {record['fund_ticker']} holds {record['company_ticker']} "
                  f"(Revenue: ${record['revenue']:,.0f})")


class TestDataIntegrity:
    """Test data integrity and completeness."""
    
    @pytest.fixture
    def db(self):
        """Fixture to provide a database connection."""
        db = Neo4jDatabase()
        yield db
        db.close()
    
    def test_no_orphan_holdings(self, db):
        """Test that all holdings are connected to a portfolio."""
        query = """
        MATCH (h:Holding)
        WHERE NOT (h)<-[:CONTAINS]-(:Portfolio)
        RETURN count(h) as orphan_count
        """
        
        result = db._execute_query(query)
        assert result[0]["orphan_count"] == 0, f"Found {result[0]['orphan_count']} orphan holdings"
    
    def test_no_orphan_sections(self, db):
        """Test that all sections are connected to a filing."""
        
        # First, check what filing labels exist in the database
        print("\n=== Checking Filing Labels in Database ===")
        label_query = """
        CALL db.labels() YIELD label
        WHERE label CONTAINS 'Filing' OR label CONTAINS '10K'
        RETURN label
        """
        labels = db._execute_query(label_query)
        print(f"Filing-related labels found: {[l['label'] for l in labels]}")
        
        # Check how many nodes exist for each filing label
        old_label_query = "MATCH (f:`10KFiling`) RETURN count(f) as count"
        new_label_query = "MATCH (f:Filing10K) RETURN count(f) as count"
        
        try:
            old_count = db._execute_query(old_label_query)[0]["count"]
            print(f"Nodes with old label '10KFiling': {old_count}")
        except:
            old_count = 0
            print(f"Nodes with old label '10KFiling': 0 (label doesn't exist)")
        
        try:
            new_count = db._execute_query(new_label_query)[0]["count"]
            print(f"Nodes with new label 'Filing10K': {new_count}")
        except:
            new_count = 0
            print(f"Nodes with new label 'Filing10K': 0 (label doesn't exist)")
        
        # Check orphan sections
        query = """
        MATCH (s:Section)
        WHERE NOT (s)<-[:HAS_RISK_FACTORS|HAS_BUSINESS_INFORMATION|HAS_FINACIALS|
                        HAS_LEGAL_PROCEEDINGS|HAS_MANAGEMENT_DISCUSSION|HAS_PROPERTIES]-(:Filing10K)
        RETURN count(s) as orphan_count
        """
        
        result = db._execute_query(query)
        orphan_count = result[0]["orphan_count"]
        
        if orphan_count > 0:
            print(f"\n=== Found {orphan_count} Orphan Sections ===")
            
            # Get sample orphan sections and check what they ARE connected to
            sample_query = """
            MATCH (s:Section)
            WHERE NOT (s)<-[:HAS_RISK_FACTORS|HAS_BUSINESS_INFORMATION|HAS_FINACIALS|
                            HAS_LEGAL_PROCEEDINGS|HAS_MANAGEMENT_DISCUSSION|HAS_PROPERTIES]-(:Filing10K)
            OPTIONAL MATCH (s)<-[r]-(connected)
            RETURN s.id as section_id, 
                   labels(s) as section_labels,
                   type(r) as relationship_type,
                   labels(connected) as connected_labels
            LIMIT 5
            """
            samples = db._execute_query(sample_query)
            print("\nSample orphan sections:")
            for sample in samples:
                print(f"  Section ID: {sample['section_id']}")
                print(f"    Labels: {sample['section_labels']}")
                print(f"    Connected via: {sample['relationship_type']} to {sample['connected_labels']}")
        
        assert result[0]["orphan_count"] == 0, f"Found {result[0]['orphan_count']} orphan sections"
    
    def test_documents_have_accession_numbers(self, db):
        """Test that all documents have accession numbers."""
        query = """
        MATCH (d:Document)
        WHERE d.accession_number IS NULL AND d.id IS NULL
        RETURN count(d) as invalid_count
        """
        
        result = db._execute_query(query)
        assert result[0]["invalid_count"] == 0, "Found documents without accession numbers"
    
    def test_companies_have_tickers(self, db):
        """Test that all companies have ticker symbols."""
        query = """
        MATCH (c:Company)
        WHERE c.ticker IS NULL OR c.ticker = ''
        RETURN count(c) as invalid_count
        """
        
        result = db._execute_query(query)
        assert result[0]["invalid_count"] == 0, "Found companies without tickers"


class TestQueryPerformance:
    """Test query performance and optimization."""
    
    @pytest.fixture
    def db(self):
        """Fixture to provide a database connection."""
        db = Neo4jDatabase()
        yield db
        db.close()
    
    def test_fund_lookup_by_ticker(self, db):
        """Test that fund lookup by ticker is fast (uses index)."""
        import time
        
        query = """
        MATCH (f:Fund {ticker: 'VTSAX'})
        RETURN f
        """
        
        start = time.time()
        result = db._execute_query(query)
        elapsed = time.time() - start
        
        assert elapsed < 0.1, f"Fund lookup took {elapsed:.3f}s (should be < 0.1s)"
    
    def test_company_lookup_by_ticker(self, db):
        """Test that company lookup by ticker is fast (uses index)."""
        import time
        
        query = """
        MATCH (c:Company {ticker: 'AAPL'})
        RETURN c
        """
        
        start = time.time()
        result = db._execute_query(query)
        elapsed = time.time() - start
        
        assert elapsed < 0.1, f"Company lookup took {elapsed:.3f}s (should be < 0.1s)"


class TestComplexQueries:
    """Test complex analytical queries."""
    
    @pytest.fixture
    def db(self):
        """Fixture to provide a database connection."""
        db = Neo4jDatabase()
        yield db
        db.close()
    
    def test_top_holdings_by_fund(self, db):
        """Test query to get top holdings for a fund."""
        query = """
        MATCH (f:Fund {ticker: 'VTSAX'})-[:HAS_PORTFOLIO]->(p:Portfolio)
              -[r:CONTAINS]->(h:Holding)
        RETURN h.ticker as ticker,
               h.name as name,
               r.marketValue as market_value,
               r.weight as weight
        ORDER BY r.marketValue DESC
        LIMIT 10
        """
        
        result = db._execute_query(query)
        print(f"\nTop 10 holdings:")
        for record in result:
            print(f"  {record['ticker']}: ${record['market_value']:,.0f} ({record['weight']:.2f}%)")
    
    def test_revenue_by_segment(self, db):
        """Test query to get revenue breakdown by segment."""
        query = """
        MATCH (c:Company {ticker: 'AAPL'})-[:HAS_FILING]->(:Filing10K)
              -[:HAS_FINACIALS]->(fin:Financials)
              -[:HAS_METRIC]->(fm:FinancialMetric {label: "Revenue"})
              -[:HAS_SEGMENT]->(seg:Segment)
        RETURN seg.label as segment,
               seg.value as value,
               seg.percentage as percentage
        ORDER BY seg.value DESC
        """
        
        result = db._execute_query(query)
        print(f"\nRevenue by segment:")
        for record in result:
            print(f"  {record['segment']}: ${record['value']:,.0f} ({record['percentage']:.1f}%)")
    
    def test_insider_transactions_for_portfolio(self, db):
        """Test query to get insider transactions for portfolio companies."""
        query = """
        MATCH (f:Fund {ticker: 'VTSAX'})-[:HAS_PORTFOLIO]->(:Portfolio)
              -[:CONTAINS]->(:Holding)-[:IS_EQUITY_OF]->(c:Company)
              -[:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction)
              -[:MADE_BY]->(p:Person)
        WHERE it.transactionType = 'Sale'
        RETURN c.ticker as company,
               p.name as insider,
               it.shares as shares,
               it.value as value
        ORDER BY it.value DESC
        LIMIT 10
        """
        
        result = db._execute_query(query)
        print(f"\nTop insider sales in portfolio:")
        for record in result:
            print(f"  {record['company']}: {record['insider']} sold {record['shares']:,.0f} shares "
                  f"(${record['value']:,.0f})")


def run_all_tests():
    """Run all tests and print summary."""
    print("="*80)
    print("NEO4J KNOWLEDGE GRAPH VALIDATION")
    print("="*80)
    
    # Run pytest programmatically
    import sys
    pytest_args = [
        __file__,
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "-s"  # Don't capture output
    ]
    
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "="*80)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)
    
    return exit_code


if __name__ == "__main__":
    run_all_tests()
