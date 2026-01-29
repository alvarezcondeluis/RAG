from neo4j import GraphDatabase
from rapidfuzz import process, fuzz, utils
from typing import Dict, List, Optional


class EntityResolver:
    def __init__(self, driver, debug: bool = False):
        self.driver = driver
        self.debug = debug
        # Cache structures
        self.providers: List[str] = []
        self.trusts: List[str] = []
        self.funds: Dict[str, str] = {}  # Map Name -> Ticker (or ID)
        self.tickers: List[str] = []
        
        # Load immediately on init
        self._refresh_cache()

    def _refresh_cache(self):
        """Loads all entities from Neo4j into memory (Fast for <10k items)"""
        print("Loading Entity Cache...")
        with self.driver.session() as session:
            # 1. Load Providers & Trusts (Simple Lists)
            self.providers = [r["name"] for r in session.run("MATCH (n:Provider) RETURN n.name as name")]
            self.trusts = [r["name"] for r in session.run("MATCH (n:Trust) RETURN n.name as name")]
            
            # 2. Load Funds (Map Name to Ticker for easy lookup later)
            fund_res = session.run("MATCH (f:Fund) RETURN f.name as name, f.ticker as ticker")
            for record in fund_res:
                self.funds[record["name"]] = record["ticker"]
                if record["ticker"]:
                    self.tickers.append(record["ticker"])
        print(f"Cache Loaded: {len(self.funds)} Funds, {len(self.providers)} Providers")

    def extract_entities(self, query: str) -> Dict[str, str]:
        """
        Scans the query against all caches and returns the BEST matches.
        Uses keyword detection to determine which entity types to search.
        Returns: {'original_text': 'entity_type'} or {'fund_name': 'Fund', 'ticker': 'Ticker'} for funds
        """
        resolved = {}
        query_lower = query.lower()
        
        # Step 1: Detect keywords to determine which entity types to search
        provider_keywords = ['provider']
        trust_keywords = ['trust', 'vanguard index funds']
        fund_keywords = ['fund', 'etf', 'index']
        ticker_keywords = ['ticker', 'symbol', 'stock']
        
        search_providers = any(keyword in query_lower for keyword in provider_keywords)
        search_trusts = any(keyword in query_lower for keyword in trust_keywords)
        search_funds = any(keyword in query_lower for keyword in fund_keywords)
        search_tickers = True  # Always check tickers (they're fast and specific)
        
        
        if self.debug:
            print(f"\n[DEBUG] Keyword Detection:")
            print(f"  Query: '{query}'")
            print(f"  Search Providers: {search_providers}")
            print(f"  Search Trusts: {search_trusts}")
            print(f"  Search Funds: {search_funds}")
        
        # Step 2: Search only the relevant entity types
        
        # A. Check Providers (Use token_sort_ratio for better typo tolerance)
        if search_providers:
            match = process.extractOne(query, self.providers, scorer=fuzz.WRatio, score_cutoff=25)
            if self.debug:
                print(f"\n[DEBUG] Provider Search:")
                print(f"  Match: {match}")
                if match:
                    print(f"  Matched: '{match[0]}' with score {match[1]}")
            if match:
                resolved[match[0]] = "Provider"

        # B. Check Trusts (Use partial_ratio for substring matching)
        if search_trusts:
            match = process.extractOne(query, self.trusts, scorer=fuzz.partial_ratio, score_cutoff=60)
            if self.debug and match:
                print(f"\n[DEBUG] Trust Search:")
                print(f"  Matched: '{match[0]}' with score {match[1]}")
            if match:
                resolved[match[0]] = "Trust"

        # C. Check Funds (The 500 items)
        if search_funds:
            match = process.extractOne(query, self.funds.keys(), scorer=fuzz.WRatio, score_cutoff=60)
            if self.debug and match:
                print(f"\n[DEBUG] Fund Search:")
                print(f"  Matched: '{match[0]}' with score {match[1]}")
            if match:
                fund_name = match[0]
                ticker = self.funds[fund_name]
                resolved[fund_name] = "Fund"
                # Also add the ticker if it exists
                if ticker:
                    resolved[ticker] = "Ticker"
            
        # D. Check Tickers (Always check - they're fast and specific)
        if search_tickers:
            # Exact word match
            query_words = set(query.split())
            for ticker in self.tickers:
                if ticker in query_words:
                    resolved[ticker] = "Ticker"
            
            # Case-insensitive ticker check
            query_upper = query.upper()
            for ticker in self.tickers:
                if ticker in query_upper:
                    resolved[ticker] = "Ticker"

        return resolved

    def get_ticker_for_fund(self, fund_name: str) -> Optional[str]:
        """Get ticker symbol for a fund name"""
        return self.funds.get(fund_name)
