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


    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching - handle hyphens and case"""
        # Replace hyphens with spaces for better matching (Small-Cap -> Small Cap)
        text = text.replace("-", " ").lower()
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

    def extract_entities(self, query: str) -> Dict[str, Dict[str, any]]:
        """
        Scans the query against all caches and returns the BEST matches with scores.
        Uses keyword detection to determine which entity types to search.
        Returns: {
            'entity_name': {
                'type': 'Fund'|'Provider'|'Trust'|'Ticker',
                'score': float,
                'ticker': str (only for funds)
            }
        }
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
            print(f"\nKeyword Detection:")
            print(f"  Query: '{query}'")
            print(f"  Search Providers: {search_providers}")
            print(f"  Search Trusts: {search_trusts}")
            print(f"  Search Funds: {search_funds}")
        
        
        
        # A. Check Providers (Use token_sort_ratio for better typo tolerance)
        if search_providers:
            match = process.extractOne(query, self.providers, scorer=fuzz.WRatio, score_cutoff=25)
            if self.debug:
                print(f"\nProvider Search:")
                print(f"  Match: {match}")
                if match:
                    print(f"  Matched: '{match[0]}' with score {match[1]}")
            if match:
                resolved[match[0]] = {
                    "type": "Provider",
                    "score": match[1]
                }

        # B. Check Trusts (Use partial_ratio for substring matching)
        if search_trusts:
            match = process.extractOne(query, self.trusts, scorer=fuzz.partial_ratio, score_cutoff=60)
            if self.debug and match:
                print(f"\nTrust Search:")
                print(f"  Matched: '{match[0]}' with score {match[1]}")
            if match:
                resolved[match[0]] = {
                    "type": "Trust",
                    "score": match[1]
                }

        # C. Check Funds (The 500 items)
        if search_funds:
            # Normalize query for better matching
            normalized_query = self._normalize_text(query)
            
            # Create normalized fund names for matching
            normalized_funds = {self._normalize_text(name): name for name in self.funds.keys()}
            
            # Try multiple scoring strategies and combine results
            # 1. Token set ratio - good for word order independence
            token_matches = process.extract(
                normalized_query, 
                normalized_funds.keys(), 
                scorer=fuzz.token_set_ratio, 
                limit=10
            )
            
            # 2. Partial ratio - good for substring matching
            partial_matches = process.extract(
                normalized_query,
                normalized_funds.keys(),
                scorer=fuzz.partial_ratio,
                limit=10
            )
            
            # Combine and deduplicate matches
            all_matches = {}
            for match, score, idx in token_matches:
                original_name = normalized_funds[match]
                all_matches[original_name] = max(all_matches.get(original_name, 0), score)
            
            for match, score, idx in partial_matches:
                original_name = normalized_funds[match]
                # Boost partial ratio scores slightly as they're more precise for substrings
                boosted_score = min(100, score * 1.1)
                all_matches[original_name] = max(all_matches.get(original_name, 0), boosted_score)
            
            # Sort by score and get top matches
            sorted_matches = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Show top matches for debugging
            print(f"\nFund Search - Top 5 matches:")
            for i, (fund_name, score) in enumerate(sorted_matches, 1):
                ticker = self.funds[fund_name]
                print(f"  {i}. '{fund_name}' ({ticker}) - Score: {score:.1f}")
            
            # Get matches within 5 points of top score (up to 5) that meet threshold
            if sorted_matches:
                top_score = sorted_matches[0][1]
                score_threshold = 60  # Minimum score to consider
                
                if top_score >= score_threshold:
                    # Include all matches within 5 points of the top score
                    matches_within_range = [
                        (name, score) for name, score in sorted_matches 
                        if score >= top_score - 5
                    ][:5]
                    
                    print(f"\nFound {len(matches_within_range)} match(es) within 5 points of top score {top_score:.1f}")
                    
                    for fund_name, score in matches_within_range:
                        ticker = self.funds[fund_name]
                        resolved[fund_name] = {
                            "type": "Fund",
                            "score": score,
                            "ticker": ticker
                        }
                        # Also add the ticker if it exists
                        if ticker:
                            resolved[ticker] = {
                                "type": "Ticker",
                                "score": score,
                                "fund": fund_name
                            }
                else:
                    print(f"\nTop score {top_score:.1f} below threshold {score_threshold}, no matches returned")
            
        # D. Check Tickers (Always check - they're fast and specific)
        if search_tickers:
            # Exact word match
            query_words = set(query.split())
            for ticker in self.tickers:
                if ticker in query_words:
                    # Don't overwrite if already added from fund matching
                    if ticker not in resolved:
                        resolved[ticker] = {
                            "type": "Ticker",
                            "score": 100.0  # Exact match
                        }
            
            # Case-insensitive ticker check
            query_upper = query.upper()
            for ticker in self.tickers:
                if ticker in query_upper and ticker not in resolved:
                    resolved[ticker] = {
                        "type": "Ticker",
                        "score": 100.0  # Exact match
                    }

        return resolved

    def get_ticker_for_fund(self, fund_name: str) -> Optional[str]:
        """Get ticker symbol for a fund name"""
        return self.funds.get(fund_name)
