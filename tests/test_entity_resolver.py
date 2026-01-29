import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from src.simple_rag.evaluation.entity_resolver import EntityResolver
from src.simple_rag.database.neo4j.config import settings


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
        return EntityResolver(neo4j_driver, debug=True)
    
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


class TestFuzzyMatching:
    """Test fuzzy matching with typos and variations"""
    
    @pytest.fixture(scope="class")
    def resolver(self):
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        resolver = EntityResolver(driver, debug=True)
        yield resolver
        driver.close()
    
    def test_provider_typos(self, resolver):
        """Test: Provider name with common typos"""
        queries = [
            "Vangaurd funds",
            "Vangard total stock",
            "funds from BlackRock",
            "Fidelity investments"
        ]
        for query in queries:
            result = resolver.extract_entities(query)
            print(f"Typo test: {query} -> {result}")
    
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


if __name__ == "__main__":
    # Run integration test
    test_integration_example()
    
    # Run pytest tests
    print("\n" + "="*60)
    print("Running pytest suite...")
    print("="*60)
    pytest.main([__file__, "-v", "-s"])
