import pytest
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from src.simple_rag.evaluation.entity_resolver import EntityResolver
from src.simple_rag.database.neo4j.config import settings

# Create reports directory
REPORTS_DIR = Path(__file__).parent / "entity_resolver_reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Module-level storage for test results (shared across all test classes)
ALL_TEST_RESULTS = []

# Temporary file to persist results between pytest run and report generation
TEMP_RESULTS_FILE = REPORTS_DIR / "temp_test_results.json"


def _save_result_to_file(result_data):
    """Save a single test result to the temp file (append mode)"""
    # Load existing results
    existing_results = []
    if TEMP_RESULTS_FILE.exists():
        try:
            with open(TEMP_RESULTS_FILE, 'r') as f:
                existing_results = json.load(f)
        except:
            existing_results = []
    
    # Append new result
    existing_results.append(result_data)
    
    # Save back to file
    with open(TEMP_RESULTS_FILE, 'w') as f:
        json.dump(existing_results, f, indent=2)


class TestEntityResolver:
    """Test suite for EntityResolver fuzzy matching capabilities"""
    
    @pytest.fixture(scope="class")
    def neo4j_driver(self):
        """Create Neo4j driver connection"""
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        yield driver
        driver.close()
    
    @pytest.fixture(scope="class")
    def resolver(self, neo4j_driver):
        """Create EntityResolver instance"""
        return EntityResolver(neo4j_driver, debug=False)
    
    def _log_result(self, test_name, query, result, expected=None, passed=True, notes=""):
        """Log test result for report generation"""
        result_data = {
            "test_name": test_name,
            "query": query,
            "result": result,
            "expected": expected,
            "passed": passed,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        ALL_TEST_RESULTS.append(result_data)
        _save_result_to_file(result_data)
    
    def test_cache_loading(self, resolver):
        """Test that cache loads entities from database"""
        assert len(resolver.funds) > 0, "Should load funds from database"
        assert len(resolver.providers) > 0, "Should load providers from database"
        assert len(resolver.tickers) > 0, "Should load tickers from database"
        print(f"✓ Loaded {len(resolver.funds)} funds, {len(resolver.providers)} providers")
    
    def test_exact_ticker_match(self, resolver):
        """Test exact ticker symbol matching"""
        query = "Show me information about VTI"
        result = resolver.extract_entities(query)
        
        assert "VTI" in result, "Should match exact ticker VTI"
        assert result["VTI"] == "Ticker"
        print(f"✓ Exact ticker match: {result}")
    
    def test_vanguard_provider_match(self, resolver):
        """Test exact provider name matching"""
        query = "Show me funds managed by Vanguard provider"
        result = resolver.extract_entities(query)
        
        # Should match "The Vanguard Group, Inc" or similar
        provider_matches = [k for k, v in result.items() if v == "Provider"]
        assert len(provider_matches) > 0, "Should match Vanguard provider"
        print(f"✓ Vanguard provider match: {provider_matches[0]}")
    
    def test_vangurd_typo_match(self, resolver):
        """Test provider matching with typo"""
        query = "Show me funds managed by Vangurd provider"
        result = resolver.extract_entities(query)
        
        # Should still match "The Vanguard Group, Inc" despite typo
        provider_matches = [k for k, v in result.items() if v == "Provider"]
        assert len(provider_matches) > 0, "Should match provider despite typo"
        print(f"✓ Vanguard typo match: {provider_matches[0]}")
    
    def test_fund_name_fuzzy_match(self, resolver):
        """Test fuzzy matching for fund names"""
        query = "What is the expense ratio of the Total Stock Market fund?"
        result = resolver.extract_entities(query)
        
        # Should match some variation of "Total Stock Market"
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        assert len(fund_matches) > 0, "Should find at least one fund match"
        print(f"✓ Fund fuzzy match: {fund_matches}")
    
    def test_provider_typo_match(self, resolver):
        """Test fuzzy matching with typos in provider name"""
        query = "Show me funds managed by Vanguard provider"
        result = resolver.extract_entities(query)
        
        # Should match "The Vanguard Group" despite typo
        provider_matches = [k for k, v in result.items() if v == "Provider"]
        assert len(provider_matches) > 0, "Should match provider despite typo"
        
        # Check if it's a Vanguard variant
        matched_provider = provider_matches[0]
        assert "Vanguard" in matched_provider, f"Should match Vanguard, got: {matched_provider}"
        print(f"✓ Provider typo match: {matched_provider}")
    
    def test_out_of_order_words(self, resolver):
        """Test matching with out-of-order words"""
        query = "Stock Market Vanguard Total fund"
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        assert len(fund_matches) > 0, "Should match despite word order"
        print(f"✓ Out-of-order match: {fund_matches}")
    
    def test_trust_matching(self, resolver):
        """Test trust name matching"""
        query = "Which funds are issued by Vanguard Index Funds?"
        result = resolver.extract_entities(query)
        
        trust_matches = [k for k, v in result.items() if v == "Trust"]
        assert len(trust_matches) > 0, "Should find trust match"
        print(f"✓ Trust match: {trust_matches}")
    
    def test_multiple_entity_extraction(self, resolver):
        """Test extracting multiple entities from one query"""
        query = "Show me VTI from Vanguard Total Stock Market"
        result = resolver.extract_entities(query)
        
        # Should find both ticker and fund/provider
        assert len(result) > 1, "Should extract multiple entities"
        
        entity_types = set(result.values())
        print(f"✓ Multiple entities: {result}")
        print(f"  Entity types found: {entity_types}")
    
    def test_no_match_low_similarity(self, resolver):
        """Test that low similarity queries don't match"""
        query = "What is the weather today?"
        result = resolver.extract_entities(query)
        
        # Should not match any financial entities
        assert len(result) == 0, "Should not match unrelated queries"
        print("✓ Correctly rejected unrelated query")
    
    def test_get_ticker_for_fund(self, resolver):
        """Test ticker lookup by fund name"""
        # Find a fund name from cache
        if resolver.funds:
            fund_name = list(resolver.funds.keys())[0]
            ticker = resolver.get_ticker_for_fund(fund_name)
            
            assert ticker is not None, "Should return ticker for valid fund"
            assert ticker == resolver.funds[fund_name], "Should return correct ticker"
            print(f"✓ Ticker lookup: {fund_name} -> {ticker}")
    
    def test_case_insensitive_ticker(self, resolver):
        """Test ticker matching with different cases"""
        query = "Show me information about vti"
        result = resolver.extract_entities(query)
        
        # Note: Current implementation is case-sensitive
        # This test documents the behavior
        print(f"  Case test result: {result}")
    
    def test_partial_fund_name(self, resolver):
        """Test matching with partial fund names"""
        query = "What about the S&P 500 fund?"
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        if len(fund_matches) > 0:
            print(f"✓ Partial name match: {fund_matches}")
        else:
            print("  No match for partial name (expected if no S&P 500 funds)")
    
    def test_total_international_stock(self, resolver):
        """Q50: How many holdings does the Total International Stock fund have in total?"""
        query = "How many holdings does the Total International Stock fund have in total?"
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        passed = len(fund_matches) > 0 and any("International Stock" in f for f in fund_matches)
        
        self._log_result(
            "Q50_Total_International_Stock",
            query,
            result,
            expected="Should match fund with CONTAINS 'Total International Stock'",
            passed=passed,
            notes="Tests partial name matching for Total International Stock"
        )
        
        print(f"Q50 - Total International Stock: {result}")
        if passed:
            print(f"  ✓ Found: {fund_matches}")
        else:
            print(f"  ✗ No match found")
        
        assert passed, f"Should match Total International Stock fund, got: {result}"
    
    def test_growth_index_fund(self, resolver):
        """Q53: Calculate the cost per 10k investment for the 'Growth Index' fund."""
        query = "Calculate the cost per 10k investment for the 'Growth Index' fund."
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        passed = len(fund_matches) > 0 and any("Growth Index" in f for f in fund_matches)
        
        self._log_result(
            "Q53_Growth_Index",
            query,
            result,
            expected="Should match fund with CONTAINS 'Growth Index'",
            passed=passed,
            notes="Tests partial name matching for Growth Index"
        )
        
        print(f"Q53 - Growth Index: {result}")
        if passed:
            print(f"  ✓ Found: {fund_matches}")
        else:
            print(f"  ✗ No match found")
        
        assert passed, f"Should match Growth Index fund, got: {result}"
    
    def test_high_dividend_yield_fund(self, resolver):
        """Q59: Return both the expense ratio and the turnover rate for the High Dividend Yield fund."""
        query = "Return both the expense ratio and the turnover rate for the High Dividend Yield fund."
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        passed = len(fund_matches) > 0 and any("High Dividend Yield" in f or "Dividend Yield" in f for f in fund_matches)
        
        self._log_result(
            "Q59_High_Dividend_Yield",
            query,
            result,
            expected="Should match fund with CONTAINS 'High Dividend Yield'",
            passed=passed,
            notes="Tests partial name matching for High Dividend Yield"
        )
        
        print(f"Q59 - High Dividend Yield: {result}")
        if passed:
            print(f"  ✓ Found: {fund_matches}")
        else:
            print(f"  ✗ No match found")
        
        assert passed, f"Should match High Dividend Yield fund, got: {result}"
    
    def test_wellington_fund(self, resolver):
        """Q64: Does the Wellington fund have a turnover rate higher than 20%?"""
        query = "Does the Wellington fund have a turnover rate higher than 20%?"
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        passed = len(fund_matches) > 0 and any("Wellington" in f for f in fund_matches)
        
        self._log_result(
            "Q64_Wellington",
            query,
            result,
            expected="Should match fund with CONTAINS 'Wellington'",
            passed=passed,
            notes="Tests partial name matching for Wellington"
        )
        
        print(f"Q64 - Wellington: {result}")
        if passed:
            print(f"  ✓ Found: {fund_matches}")
        else:
            print(f"  ✗ No match found")
        
        assert passed, f"Should match Wellington fund, got: {result}"


class TestHyphenationIssues:
    """Test hyphenation and formatting differences"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=False)
        yield resolver
        driver.close()
    
    def _log_result(self, test_name, query, result, expected=None, passed=True, notes=""):
        """Log test result for report generation"""
        result_data = {
            "test_name": test_name,
            "query": query,
            "result": result,
            "expected": expected,
            "passed": passed,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        ALL_TEST_RESULTS.append(result_data)
        _save_result_to_file(result_data)
    
    def test_small_cap_hyphenation(self, resolver):
        """Q21 & Q67: What is the total number of holdings for the Vanguard Small Cap Fund?"""
        query = "What is the total number of holdings for the Vanguard Small Cap Fund?"
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        # Should match "Vanguard Small-Cap Index Fund" even though query says "Small Cap"
        passed = len(fund_matches) > 0 and any("Small" in f and "Cap" in f for f in fund_matches)
        
        self._log_result(
            "Q21_Q67_Small_Cap_Hyphenation",
            query,
            result,
            expected="Should match 'Vanguard Small-Cap Index Fund' (with hyphen)",
            passed=passed,
            notes="Tests matching 'Small Cap' to 'Small-Cap' with hyphen"
        )
        
        print(f"Q21/Q67 - Small Cap: {result}")
        if passed:
            print(f"  ✓ Found: {fund_matches}")
        else:
            print(f"  ✗ No match found for Small Cap -> Small-Cap")
        
        assert passed, f"Should match Small-Cap fund, got: {result}"
    
    def test_mid_cap_hyphenation(self, resolver):
        """Q25 & Q71: Who are the managers of the Vanguard Mid Cap Growth fund?"""
        query = "Who are the managers of the Vanguard Mid Cap Growth fund?"
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        # Should match "Vanguard Mid-Cap Growth Index Fund" even though query says "Mid Cap"
        passed = len(fund_matches) > 0 and any("Mid" in f and "Cap" in f and "Growth" in f for f in fund_matches)
        
        self._log_result(
            "Q25_Q71_Mid_Cap_Hyphenation",
            query,
            result,
            expected="Should match 'Vanguard Mid-Cap Growth Index Fund' (with hyphen)",
            passed=passed,
            notes="Tests matching 'Mid Cap' to 'Mid-Cap' with hyphen"
        )
        
        print(f"Q25/Q71 - Mid Cap Growth: {result}")
        if passed:
            print(f"  ✓ Found: {fund_matches}")
        else:
            print(f"  ✗ No match found for Mid Cap -> Mid-Cap")
        
        assert passed, f"Should match Mid-Cap Growth fund, got: {result}"


class TestTickerHallucinations:
    """Test cases where resolver should find correct tickers to prevent hallucinations"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=False)
        yield resolver
        driver.close()
    
    def _log_result(self, test_name, query, result, expected=None, passed=True, notes=""):
        """Log test result for report generation"""
        result_data = {
            "test_name": test_name,
            "query": query,
            "result": result,
            "expected": expected,
            "passed": passed,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        ALL_TEST_RESULTS.append(result_data)
        _save_result_to_file(result_data)
    
    def test_information_technology_etf(self, resolver):
        """Q63: Information Technology ETF (should find VGT, not hallucinate IXIC)"""
        query = "What is the expense ratio for the Information Technology ETF?"
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        ticker_matches = [k for k, v in result.items() if v == "Ticker"]
        
        # Should find Information Technology fund/ticker (VGT), not IXIC
        passed = (len(fund_matches) > 0 and any("Information Technology" in f or "Technology" in f for f in fund_matches)) or \
                 (len(ticker_matches) > 0 and "VGT" in ticker_matches)
        
        self._log_result(
            "Q63_Information_Technology_ETF",
            query,
            result,
            expected="Should find VGT or Information Technology fund (not IXIC)",
            passed=passed,
            notes="Prevents hallucination of IXIC ticker"
        )
        
        print(f"Q63 - Information Technology ETF: {result}")
        if passed:
            print(f"  ✓ Found: Funds={fund_matches}, Tickers={ticker_matches}")
        else:
            print(f"  ✗ No match found - may cause IXIC hallucination")
    
    def test_health_care_etf(self, resolver):
        """Q78: Health Care ETF (should find VHT, not hallucinate VTI)"""
        query = "What is the 5-year return of the Health Care ETF?"
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        ticker_matches = [k for k, v in result.items() if v == "Ticker"]
        
        # Should find Health Care fund/ticker (VHT), not VTI
        passed = (len(fund_matches) > 0 and any("Health Care" in f or "Healthcare" in f for f in fund_matches)) or \
                 (len(ticker_matches) > 0 and "VHT" in ticker_matches)
        
        self._log_result(
            "Q78_Health_Care_ETF",
            query,
            result,
            expected="Should find VHT or Health Care fund (not VTI)",
            passed=passed,
            notes="Prevents hallucination of VTI ticker"
        )
        
        print(f"Q78 - Health Care ETF: {result}")
        if passed:
            print(f"  ✓ Found: Funds={fund_matches}, Tickers={ticker_matches}")
        else:
            print(f"  ✗ No match found - may cause VTI hallucination")
    
    def test_real_estate_etf(self, resolver):
        """Q87: Real Estate ETF (should find VNQ, not hallucinate REIT)"""
        query = "What are the risks for the Real Estate ETF?"
        result = resolver.extract_entities(query)
        
        fund_matches = [k for k, v in result.items() if v == "Fund"]
        ticker_matches = [k for k, v in result.items() if v == "Ticker"]
        
        # Should find Real Estate fund/ticker (VNQ), not REIT
        passed = (len(fund_matches) > 0 and any("Real Estate" in f for f in fund_matches)) or \
                 (len(ticker_matches) > 0 and "VNQ" in ticker_matches)
        
        self._log_result(
            "Q87_Real_Estate_ETF",
            query,
            result,
            expected="Should find VNQ or Real Estate fund (not REIT)",
            passed=passed,
            notes="Prevents hallucination of REIT ticker"
        )
        
        print(f"Q87 - Real Estate ETF: {result}")
        if passed:
            print(f"  ✓ Found: Funds={fund_matches}, Tickers={ticker_matches}")
        else:
            print(f"  ✗ No match found - may cause REIT hallucination")
    
    def test_nyse_arca_exchange(self, resolver):
        """Q68: NYSE Arca exchange (should preserve 'Arca' part)"""
        query = "Show me funds listed on NYSE Arca."
        result = resolver.extract_entities(query)
        
        # This test is more about documenting the query pattern
        # The resolver doesn't handle exchanges, but we log it
        self._log_result(
            "Q68_NYSE_Arca",
            query,
            result,
            expected="Query should preserve 'NYSE Arca' not just 'NYSE'",
            passed=True,
            notes="Documents exchange name handling - resolver doesn't extract exchanges"
        )
        
        print(f"Q68 - NYSE Arca: {result}")
        print(f"  Note: Resolver doesn't extract exchanges, but query should preserve 'Arca'")


class TestFuzzyMatching:
    """Test basic fuzzy matching capabilities"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=False)
        yield resolver
        driver.close()
    
    def _log_result(self, test_name, query, result, expected=None, passed=True, notes=""):
        """Log test result for report generation"""
        result_data = {
            "test_name": test_name,
            "query": query,
            "result": result,
            "expected": expected,
            "passed": passed,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        ALL_TEST_RESULTS.append(result_data)
        _save_result_to_file(result_data)
    
    def test_fund_name_variations(self, resolver):
        """Test: Fund name variations and partial names"""
        queries = [
            "Total Stock Market fund",
            "Vanguard Total Stock",
            "S&P 500 index",
            "total stock market index",
            "international stock fund"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Variation: {query} -> {result}")
    
    def test_word_order_insensitivity(self, resolver):
        """Test: Different word orderings"""
        queries = [
            "Stock Market Total Vanguard",
            "Market Stock Total fund",
            "Index S&P 500"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Word order: {query} -> {result}")


class TestMultiEntityExtraction:
    """Test extraction of multiple entities from single query"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_multiple_tickers(self, resolver):
        """Test: Multiple tickers in one query"""
        queries = [
            "Compare VTI and VOO",
            "What's the difference between VTI, VOO, and VXUS?",
            "Show me VTI vs VOO performance"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Multiple tickers: {query} -> {result}")
            ticker_count = sum(1 for v in result.values() if v == "Ticker")
            assert ticker_count >= 2, f"Should find multiple tickers in: {query}"
    
    def test_fund_and_provider(self, resolver):
        """Test: Fund and provider in same query"""
        queries = [
            "Vanguard Total Stock Market fund",
            "BlackRock's S&P 500 index fund",
            "Show me Fidelity total market fund"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Fund + Provider: {query} -> {result}")
    
    def test_ticker_and_fund_name(self, resolver):
        """Test: Ticker and fund name together"""
        queries = [
            "VTI Vanguard Total Stock Market",
            "Is VTI the Vanguard Total Stock fund?"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Ticker + Fund: {query} -> {result}")


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_no_match_unrelated(self, resolver):
        """Test: Completely unrelated queries"""
        queries = [
            "What is the weather today?",
            "How do I cook pasta?",
            "What is quantum mechanics?",
            "Tell me a joke"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            assert len(result) == 0, f"Should not match: {query}"
            print(f"✓ Correctly rejected: {query}")
    
    def test_partial_matches(self, resolver):
        """Test: Partial word matches"""
        queries = [
            "market",
            "stock",
            "fund",
            "index"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Partial: {query} -> {result}")
    
    def test_special_characters(self, resolver):
        """Test: Queries with special characters"""
        queries = [
            "S&P 500",
            "U.S. stock market",
            "What's VTI?",
            "VTI's performance"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Special chars: {query} -> {result}")
    
    def test_very_long_queries(self, resolver):
        """Test: Long, complex queries"""
        query = "I'm looking for information about the Vanguard Total Stock Market Index Fund, specifically its expense ratio compared to other index funds from Fidelity and BlackRock, and also want to know about VTI and VOO performance metrics"
        result = resolver.extract_entities(query)
        print(f"Long query: {result}")
        assert len(result) > 0, "Should extract entities from long query"
    
    def test_abbreviations(self, resolver):
        """Test: Common abbreviations"""
        queries = [
            "TSM fund",
            "intl stock fund",
            "em fund"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Abbreviation: {query} -> {result}")


class TestRealWorldScenarios:
    """Test real-world query scenarios"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_beginner_queries(self, resolver):
        """Test: Typical beginner investor queries"""
        queries = [
            "What is a good index fund for beginners?",
            "Tell me about VTI",
            "Should I invest in Vanguard or Fidelity?",
            "What's the cheapest S&P 500 fund?"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Beginner: {query} -> {result}")
    
    def test_comparison_queries(self, resolver):
        """Test: Comparison queries"""
        queries = [
            "VTI vs VOO which is better?",
            "Compare Vanguard Total Stock to Fidelity Total Market",
            "Difference between VTI and VXUS"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Comparison: {query} -> {result}")
    
    def test_performance_queries(self, resolver):
        """Test: Performance-related queries"""
        queries = [
            "How has VTI performed this year?",
            "What's the 10-year return of Vanguard Total Stock?",
            "VTI returns vs S&P 500"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Performance: {query} -> {result}")
    
    def test_portfolio_queries(self, resolver):
        """Test: Portfolio construction queries"""
        queries = [
            "Build a portfolio with VTI and VXUS",
            "I want 60% VTI, 30% VOO, 10% bonds",
            "Three fund portfolio Vanguard"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Portfolio: {query} -> {result}")


class TestCaseSensitivity:
    """Test case sensitivity and normalization"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_uppercase_tickers(self, resolver):
        """Test: Uppercase ticker variations"""
        queries = [
            "VTI",
            "vti",
            "Vti",
            "vTi"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Case test: {query} -> {result}")
    
    
    def test_mixed_case_fund_names(self, resolver):
        """Test: Mixed case fund names"""
        queries = [
            "TOTAL STOCK MARKET fund",
            "total stock market FUND",
            "ToTaL StOcK MaRkEt FuNd"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Mixed case: {query} -> {result}")
    
    def test_lowercase_provider_names(self, resolver):
        """Test: Lowercase provider names"""
        queries = [
            "vanguard provider",
            "blackrock provider",
            "fidelity provider"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Lowercase provider: {query} -> {result}")


class TestNumericAndSymbols:
    """Test handling of numbers and special symbols"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_numeric_fund_names(self, resolver):
        """Test: Fund names with numbers"""
        queries = [
            "S&P 500 fund",
            "Russell 2000 index",
            "FTSE 100 fund",
            "2060 target retirement fund"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Numeric: {query} -> {result}")
    
    def test_ampersand_handling(self, resolver):
        """Test: Ampersand and special characters"""
        queries = [
            "S&P 500",
            "S & P 500",
            "S and P 500"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Ampersand: {query} -> {result}")
    
    def test_hyphenated_names(self, resolver):
        """Test: Hyphenated fund names"""
        queries = [
            "mid-cap fund",
            "small-cap value",
            "tax-exempt bond fund"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Hyphenated: {query} -> {result}")


class TestContextualQueries:
    """Test queries with different contextual patterns"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_question_formats(self, resolver):
        """Test: Different question formats"""
        queries = [
            "What is VTI?",
            "Tell me about VTI",
            "Can you explain VTI?",
            "VTI information please",
            "I want to know about VTI"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            assert "VTI" in result, f"Should find VTI in: {query}"
            print(f"✓ Question format: {query} -> {result}")
    
    def test_negation_queries(self, resolver):
        """Test: Queries with negation"""
        queries = [
            "Not VTI but VOO",
            "Don't show me VTI, show VOO instead",
            "Anything except VTI"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Negation: {query} -> {result}")
    
    def test_conditional_queries(self, resolver):
        """Test: Conditional queries"""
        queries = [
            "If VTI is expensive, show me alternatives",
            "Should I buy VTI or VOO?",
            "VTI unless there's something better"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Conditional: {query} -> {result}")
    
    def test_temporal_queries(self, resolver):
        """Test: Time-based queries"""
        queries = [
            "VTI performance last year",
            "Historical data for Vanguard Total Stock",
            "VTI returns since 2020",
            "Future outlook for total stock market fund"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Temporal: {query} -> {result}")


class TestBoundaryConditions:
    """Test boundary conditions and limits"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_empty_query(self, resolver):
        """Test: Empty or whitespace-only queries"""
        queries = ["", "   ", "\t", "\n"]
        for query in queries:
            result = resolver.extract_entities(query)
            assert len(result) == 0, f"Should return empty for: '{query}'"
            print(f"✓ Empty query handled: '{query}'")
    
    def test_single_character(self, resolver):
        """Test: Single character queries"""
        queries = ["V", "T", "I", "S"]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Single char: {query} -> {result}")
    
    def test_very_short_queries(self, resolver):
        """Test: Very short queries"""
        queries = ["VT", "VO", "fund", "etf"]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Short: {query} -> {result}")
    
    def test_repeated_words(self, resolver):
        """Test: Queries with repeated words"""
        queries = [
            "VTI VTI VTI",
            "fund fund fund",
            "Vanguard Vanguard total stock"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Repeated: {query} -> {result}")


class TestStopwordHandling:
    """Test stopword removal and handling"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_with_stopwords(self, resolver):
        """Test: Queries with common stopwords"""
        queries = [
            "the Vanguard Total Stock Market index fund",
            "a total stock market fund from Vanguard",
            "an S&P 500 index fund"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"With stopwords: {query} -> {result}")
    
    def test_stopword_only_difference(self, resolver):
        """Test: Queries differing only by stopwords"""
        queries = [
            "Total Stock Market",
            "the Total Stock Market",
            "a Total Stock Market fund"
        ]
        results = [resolver.extract_entities(q) for q in queries]
        print(f"Stopword variations:")
        for q, r in zip(queries, results):
            print(f"  {q} -> {r}")


class TestTypoVariations:
    """Test various typo patterns"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_transposition_typos(self, resolver):
        """Test: Letter transposition typos"""
        queries = [
            "Vangaurd",  # au -> ua
            "Fideltiy",  # ti -> it
            "Balckrock"  # la -> al
        ]
        for query in queries:
            result = resolver.extract_entities(f"{query} provider")
            print(f"Transposition: {query} -> {result}")
    
    def test_missing_letter_typos(self, resolver):
        """Test: Missing letter typos"""
        queries = [
            "Vangard",   # missing u
            "Fidelty",   # missing i
            "Blckrock"   # missing a
        ]
        for query in queries:
            result = resolver.extract_entities(f"{query} provider")
            print(f"Missing letter: {query} -> {result}")
    
    def test_extra_letter_typos(self, resolver):
        """Test: Extra letter typos"""
        queries = [
            "Vanguuard",
            "Fidelitty",
            "Blackrrock"
        ]
        for query in queries:
            result = resolver.extract_entities(f"{query} provider")
            print(f"Extra letter: {query} -> {result}")
    
    def test_phonetic_typos(self, resolver):
        """Test: Phonetic typos"""
        queries = [
            "Vangard",
            "Fidelety",
            "Blakrok"
        ]
        for query in queries:
            result = resolver.extract_entities(f"{query} provider")
            print(f"Phonetic: {query} -> {result}")


class TestMultiWordMatching:
    """Test multi-word entity matching"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_scattered_words(self, resolver):
        """Test: Entity words scattered in query"""
        queries = [
            "I want a Total fund from Vanguard for Stock Market investing",
            "Looking for Stock investing in the Market with Total coverage",
            "Market Total Stock fund information"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Scattered: {query} -> {result}")
    
    def test_word_boundaries(self, resolver):
        """Test: Word boundary matching"""
        queries = [
            "totalstockmarket",  # No spaces
            "total-stock-market",  # Hyphens
            "total_stock_market"   # Underscores
        ]
        for query in queries:
            result = resolver.extract_entities(f"{query} fund")
            print(f"Boundaries: {query} -> {result}")


def test_integration_example():
    """Integration test showing typical usage"""
    print("\n" + "="*60)
    print("INTEGRATION TEST - Typical Usage Example")
    print("="*60)
    
    driver = GraphDatabase.driver(
        settings.NEO4J_URL,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )
    
    try:
        resolver = EntityResolver(driver)
        
        # Test queries
        test_queries = [
            "Show me the Vangurd Total Stock fund",
            "What is VTI's expense ratio?",
            "Which funds does Vanguard Index Funds manage?",
            "Compare VOO and VTI performance"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = resolver.extract_entities(query)
            print(f"Entities found: {result}")
            
            # Show tickers for fund matches
            for entity, entity_type in result.items():
                if entity_type == "Fund":
                    ticker = resolver.get_ticker_for_fund(entity)
                    print(f"  → {entity} = {ticker}")
    
    finally:
        driver.close()


def generate_test_report():
    """Generate comprehensive test report from all test results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"entity_resolver_test_report_{timestamp}.json"
    
    # Load results from temporary file
    all_results = []
    if TEMP_RESULTS_FILE.exists():
        with open(TEMP_RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        print(f"[DEBUG] Loaded {len(all_results)} test results from temp file")
    else:
        print("[WARNING] No temp results file found, using empty results")
        all_results = ALL_TEST_RESULTS
    
    # Calculate summary statistics
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r.get("passed", False))
    failed_tests = total_tests - passed_tests
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Group by test category
    by_category = {}
    for result in all_results:
        test_name = result["test_name"]
        category = test_name.split("_")[0] if "_" in test_name else "Other"
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result)
    
    # Create report structure
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": f"{pass_rate:.2f}%"
        },
        "summary": {
            "critical_issues": [
                {
                    "issue": "Partial Name Matching",
                    "tests": ["Q50", "Q53", "Q59", "Q64"],
                    "description": "Tests if resolver can match partial fund names using CONTAINS logic"
                },
                {
                    "issue": "Hyphenation Differences",
                    "tests": ["Q21", "Q67", "Q25", "Q71"],
                    "description": "Tests matching 'Small Cap' to 'Small-Cap' and 'Mid Cap' to 'Mid-Cap'"
                },
                {
                    "issue": "Ticker Hallucinations",
                    "tests": ["Q63", "Q78", "Q87"],
                    "description": "Tests that resolver finds correct tickers to prevent LLM hallucinations"
                }
            ]
        },
        "test_results_by_category": by_category,
        "all_test_results": all_results,
        "failed_tests_detail": [r for r in all_results if not r.get("passed", False)]
    }
    
    # Write report to file
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also create a human-readable text report
    text_report_file = REPORTS_DIR / f"entity_resolver_test_report_{timestamp}.txt"
    with open(text_report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENTITY RESOLVER TEST REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Passed: {passed_tests}\n")
        f.write(f"Failed: {failed_tests}\n")
        f.write(f"Pass Rate: {pass_rate:.2f}%\n\n")
        
        f.write("CRITICAL ISSUES TESTED\n")
        f.write("-"*80 + "\n")
        for issue in report["summary"]["critical_issues"]:
            f.write(f"\n{issue['issue']}\n")
            f.write(f"  Tests: {', '.join(issue['tests'])}\n")
            f.write(f"  Description: {issue['description']}\n")
        
        f.write("\n\nFAILED TESTS DETAIL\n")
        f.write("-"*80 + "\n")
        for result in report["failed_tests_detail"]:
            f.write(f"\n[FAILED] {result['test_name']}\n")
            f.write(f"  Query: {result['query']}\n")
            f.write(f"  Result: {result['result']}\n")
            f.write(f"  Expected: {result.get('expected', 'N/A')}\n")
            f.write(f"  Notes: {result.get('notes', 'N/A')}\n")
        
        f.write("\n\nALL TEST RESULTS\n")
        f.write("-"*80 + "\n")
        for result in all_results:
            status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
            f.write(f"\n{status} - {result['test_name']}\n")
            f.write(f"  Query: {result['query']}\n")
            f.write(f"  Result: {result['result']}\n")
    
    print(f"\n{'='*80}")
    print(f"TEST REPORT GENERATED")
    print(f"{'='*80}")
    print(f"JSON Report: {report_file}")
    print(f"Text Report: {text_report_file}")
    print(f"\nSummary: {passed_tests}/{total_tests} tests passed ({pass_rate:.2f}%)")
    print(f"{'='*80}\n")
    
    # Clean up temp file
    if TEMP_RESULTS_FILE.exists():
        TEMP_RESULTS_FILE.unlink()
    
    return report


if __name__ == "__main__":
    # Clear temp results file from previous runs
    if TEMP_RESULTS_FILE.exists():
        TEMP_RESULTS_FILE.unlink()
    
    # Run pytest tests
    print("\n" + "="*60)
    print("Running Entity Resolver Test Suite...")
    print("="*60)
    pytest.main([__file__, "-v", "-s"])
    
    # Generate report after tests complete
    print("\n" + "="*60)
    print("Generating Test Report...")
    print("="*60)
    generate_test_report()
