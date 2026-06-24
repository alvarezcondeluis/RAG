
import re
from rapidfuzz import process, fuzz
from typing import Dict, List, Optional


class EntityResolver:

    # Words stripped from both query tokens and entity name tokens before
    # token-level fuzzy matching. Keeps only distinctive name fragments.
    # NOTE: 'fund', 'etf', 'index', 'trust' are intentionally NOT removed — they
    # distinguish between fund types (e.g., "Vanguard 500 Fund" vs "Vanguard 500 ETF",
    # or "Total Stock Market Fund" vs "Total Stock Market Index Fund")
    _STOPWORDS: frozenset = frozenset({
        # Common English
        'what', 'does', 'show', 'tell', 'give', 'list', 'find', 'which',
        'have', 'that', 'this', 'from', 'with', 'about', 'annual', 'report',
        'filing', 'the', 'are', 'for', 'and', 'how', 'who', 'when', 'say',
        'its', 'any', 'all', 'also', 'into', 'some', 'been', 'were', 'their',
        'will', 'would', 'could', 'should', 'more', 'most', 'than', 'your',
        'make', 'where', 'there', 'they', 'them', 'these', 'those', 'such',
        'each', 'very', 'just', 'over', 'like', 'only', 'here', 'both',
        'between', 'through', 'during', 'before', 'after', 'above', 'below',
        'since', 'while', 'says', 'said', 'using', 'used',
        # Company legal suffixes
        'inc', 'corp', 'company', 'llc', 'limited', 'plc', 'group',
    })

    def __init__(self, driver, debug: bool = False):
        self.driver = driver
        self.debug = debug
        # Cache structures
        self.providers: List[str] = []
        self.trusts: List[str] = []
        self.funds: Dict[str, str] = {}      # Fund name -> ticker
        self.companies: Dict[str, str] = {}  # Company name -> ticker (Filing10K only)
        self.tickers: List[str] = []
        
        # Load immediately on init
        self._refresh_cache()

    def _refresh_cache(self):
        """Loads all entities from Neo4j into memory (Fast for <10k items)"""
        print("Loading Entity Cache...")
        # Reset before reload so a refresh doesn't accumulate duplicates
        self.providers = []
        self.trusts = []
        self.funds = {}
        self.companies = {}
        self.tickers = []

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

            # 3. Load Company names + tickers (only those with a Filing10K relationship)
            company_res = session.run(
                "MATCH (c:Company)-[:REPORTS_IN]->(:Filing10K) "
                "RETURN DISTINCT c.ticker AS ticker, c.name AS companyName "
                "ORDER BY ticker ASC"
            )
            company_count = 0
            for record in company_res:
                ticker = record["ticker"]
                name = record["companyName"]
                if ticker and ticker not in self.tickers:
                    self.tickers.append(ticker)
                    company_count += 1
                if name and ticker:
                    self.companies[name] = ticker

        print(f"✓ Entity cache loaded:")
        print(f"  Funds     : {len(self.funds):>5}  ({len(self.tickers) - company_count} with tickers)")
        print(f"  Companies : {company_count:>5}  (with Filing10K, {len(self.companies)} with names)")
        print(f"  Providers : {len(self.providers):>5}")
        print(f"  Trusts    : {len(self.trusts):>5}")
        if not self.funds:
            print("  ⚠️  WARNING: No funds loaded — entity resolver will not resolve fund/ticker queries.")


    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching - handle hyphens and case"""
        text = text.replace("-", " ").lower()
        text = ' '.join(text.split())
        return text

    def _normalize_for_matching(self, text: str) -> str:
        """
        Normalize + strip stopwords before full-string fuzzy matching.
        Removes only truly generic English words — preserves fund classification
        keywords ('fund', 'etf', 'index', 'trust') since they distinguish fund types.
        e.g. "typical advisory fees for index funds" → "typical advisory fees index funds"
             "Vanguard Total Stock Market Index Fund" → "vanguard total stock market index fund"
        """
        tokens = [t for t in self._normalize_text(text).split() if t not in self._STOPWORDS]
        return " ".join(tokens)

    def _extract_candidate_tokens(self, query: str) -> List[str]:
        """
        Extract significant words from a query for token-level typo matching.
        Strips possessives, punctuation, stopwords, and tokens shorter than 3 chars.
        Includes 3-char tickers (VTI, BND, AGG, etc.) for token-level matching.
        e.g. "What does Amazoon's annual report say?" → ["amazoon"]
        e.g. "risks of VTI fund" → ["vti", "risks", "fund"]
        """
        text = re.sub(r"'s\b", "", query.lower())
        text = re.sub(r"[^\w\s]", " ", text)
        return [w for w in text.split() if len(w) >= 3 and w not in self._STOPWORDS]

    def _entity_tokens(self, name: str) -> List[str]:
        """
        Extract significant tokens from an entity name.
        Filters generic suffixes so 'Vanguard' rather than 'Fund' drives matching.
        """
        return [t for t in self._normalize_text(name).split()
                if len(t) > 3 and t not in self._STOPWORDS]

    def _token_level_score(self, query_tokens: List[str], entity_name: str) -> float:
        """
        Best single-token fuzzy match between query tokens and entity name tokens.
        Catches typos like 'Amazoon'→'Amazon' (fuzz.ratio ≈ 84) that full-string
        matching misses because the typo is diluted across the whole sentence.
        Returns the highest fuzz.ratio across all (query_token, entity_token) pairs.
        """
        entity_toks = self._entity_tokens(entity_name)
        if not entity_toks or not query_tokens:
            return 0.0
        best = 0.0
        for qt in query_tokens:
            for et in entity_toks:
                s = fuzz.ratio(qt, et)
                if s > best:
                    best = s
        return best

    def _calculate_secondary_score(self, fuzzy_score: float, query: str, entity_name: str) -> float:
        """
        Apply secondary scoring to increase differentiation between matches.
        Uses multiple tiebreakers: token overlap ratio, length, token density.

        E.g. for query "500 index fund vanguard" matching:
          - "Vanguard 500 Index Fund": overlap 4/4, compact (4 tokens) → boost
          - "Vanguard Extended Market Index Fund": overlap 3/4, longer (6 tokens) → slight penalty
          - "iShares S&P 500 Index Fund": overlap 2/4, moderate length (5 tokens) → more penalty
        """
        # Extract significant tokens from normalized query and entity name
        query_tokens_set = set(self._normalize_for_matching(query).split())
        entity_tokens_list = self._normalize_for_matching(entity_name).split()
        entity_tokens_set = set(entity_tokens_list)

        if not query_tokens_set or not entity_tokens_set:
            return fuzzy_score

        # 1. Token overlap ratio: proportion of query tokens present in the entity name
        overlap_count = len(query_tokens_set & entity_tokens_set)
        token_overlap_ratio = overlap_count / len(query_tokens_set) if query_tokens_set else 0.0

        # Multiplicative adjustment from token overlap (creates main score spread)
        overlap_factor = 0.8 + (0.2 * token_overlap_ratio)  # Range: 0.8 to 1.0

        # 2. Token density tiebreaker: prefer matches where matched tokens are more concentrated
        # E.g., "Vanguard 500 Index Fund" is dense (4 tokens, 4 matched)
        # "Vanguard Extended Market Index Fund" is less dense (6 tokens, 3-4 matched)
        entity_token_count = len(entity_tokens_list)
        query_token_count = len(query_tokens_set)

        # Token density = proportion of entity tokens that matched (higher = better)
        # Then adjust for query length: matches shorter entity names than query length
        density_ratio = overlap_count / max(entity_token_count, 1)
        length_ratio = entity_token_count / max(query_token_count, 1)

        # Density factor: prefer compact matches
        # - Perfect compact match: 1.0 bonus
        # - Dilute match: penalty up to 0.10
        density_bonus = density_ratio * 0.1  # Up to +0.10 for perfect density

        # Length factor: penalize names significantly longer than query
        if length_ratio > 1.2:
            length_penalty = (length_ratio - 1.2) * 0.05  # 5% penalty per 20% over query length
            length_penalty = min(0.10, length_penalty)  # Cap penalty at -0.10
        else:
            length_penalty = 0.0

        # Net adjustment from tiebreakers (can be -0.10 to +0.10)
        tiebreaker_adjustment = density_bonus - length_penalty

        # Combine: fuzzy score × overlap factor × (1 + tiebreaker adjustment)
        final_score = fuzzy_score * overlap_factor * (1.0 + tiebreaker_adjustment)
        return min(100, final_score)

    _AGGREGATE_INDICATORS: frozenset = frozenset({
        'highest', 'lowest', 'best', 'worst', 'most', 'least', 'greatest',
        'average', 'across all', 'among all', 'every fund', 'all funds',
        'compare all', 'rank all', 'top fund', 'bottom fund',
    })

    def _is_aggregate_query(self, query: str) -> bool:
        """
        Detect superlative/aggregate queries that should search globally,
        not be filtered to specific entities.
        e.g. "Which fund has the highest expense ratio?" → True
             "What is VTI's expense ratio?" → False
        """
        query_lower = query.lower()
        # If query contains an explicit ticker, it's not aggregate
        import re as _re
        if _re.search(r'\b[A-Z]{2,5}\b', query):
            return False
        return any(indicator in query_lower for indicator in self._AGGREGATE_INDICATORS)

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
        is_aggregate = self._is_aggregate_query(query)
        
        # Step 1: Detect keywords to determine which entity types to search
        # 'manage'/'manages' are added because the schema has Provider -[:MANAGES]-> Trust,
        # so queries about what something "manages" are asking about a Provider entity.
        provider_keywords = ['provider', 'manage', 'manages', 'managed']
        trust_keywords = ['trust', 'vanguard index funds']
        fund_keywords = [
            'fund', 'etf', 'index', 'portfolio', 'holdings', 'strategy', 'prospectus',
            'stock', 'market', 'ratio', 'performance', 'return', 'allocation',
            'equity', 'bond', 'sector', 'objective', 'risk',
        ]
        company_keywords = [
            'company', '10-k', '10k', 'annual', 'filing', 'revenue', 'earnings',
            'ceo', 'executive', 'insider', 'acquisition', 'merger', 'fiscal',
            'inc', 'corp', 'business', 'report', 'financials', 'financial',
            'highlights', 'quarterly', 'balance sheet', 'income statement', 'portfolio',
        ]

        search_providers = any(keyword in query_lower for keyword in provider_keywords)
        search_trusts = any(keyword in query_lower for keyword in trust_keywords)
        search_funds = any(keyword in query_lower for keyword in fund_keywords)
        search_companies = any(keyword in query_lower for keyword in company_keywords)

        # Step 1.5: Pre-flight exact ticker detection
        # Run BEFORE fuzzy matching so exact tickers always get score 100.0 and skip
        # the noisy fund fuzzy search (which scores ~87 for any fund containing "fund").
        # Only consider words that are ALREADY all-caps in the original query: tickers
        # like "VTI" or "AAPL" are written in caps by users; lowercase words like "made"
        # must never be uppercased and mistaken for the MADE ticker.
        import re as _re
        _query_upper_words: set = set()
        for _w in query.split():
            _w_stripped = _re.sub(r"'[Ss]$", "", _w)          # remove possessive
            _w_clean = _re.sub(r"[^A-Za-z0-9]", "", _w_stripped)  # strip punctuation
            if _w_clean and _w_clean == _w_clean.upper() and len(_w_clean) >= 2:
                _query_upper_words.add(_w_clean)
        _query_upper_words.discard("")

        found_exact_tickers: set = set()
        for ticker in self.tickers:
            if ticker in _query_upper_words:
                found_exact_tickers.add(ticker)
                resolved[ticker] = {"type": "Ticker", "score": 100.0}

        # When query contains an exact ticker, skip the fuzzy fund search.
        # "fund VTI" → every fund with "fund" in its name scores ~87 via partial_ratio*1.1 — pure noise.
        if found_exact_tickers:
            search_funds = False

        if self.debug:
            print(f"\nKeyword Detection:")
            print(f"  Query: '{query}'")
            print(f"  Search Providers: {search_providers}")
            print(f"  Search Trusts: {search_trusts}")
            print(f"  Search Funds: {search_funds}")
            print(f"  Search Companies: {search_companies}")
        
        
        
        # A. Check Providers (Use token_sort_ratio for better typo tolerance)
        if search_providers:
            match = process.extractOne(query, self.providers, scorer=fuzz.WRatio, score_cutoff=60)
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
        # Cutoff raised to 75: generic uses of the word "trust" (e.g. "how many trusts")
        # produce false-positive matches around 60. Specific trust names score 90+.
        if search_trusts:
            match = process.extractOne(query, self.trusts, scorer=fuzz.partial_ratio, score_cutoff=75)
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
            # Stopword-filtered normalization: strips generic domain words so
            # "typical advisory fees for index funds" doesn't match fund names
            # on the word "index" alone.
            normalized_query = self._normalize_for_matching(query)

            # Create stopword-filtered fund names for matching
            normalized_funds = {self._normalize_for_matching(name): name for name in self.funds.keys()}
            # Drop entries that became empty after stopword removal
            normalized_funds = {k: v for k, v in normalized_funds.items() if k}

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
                boosted_score = min(100, score * 1.1)
                all_matches[original_name] = max(all_matches.get(original_name, 0), boosted_score)

            # Token-level pass: catches single-word typos that full-string matching
            # misses because the misspelled word is diluted across the full query.
            # e.g. "Vangaurd Total Stock Market" → "Vanguard" scores 94 token-level.
            query_tokens = self._extract_candidate_tokens(query)
            if query_tokens:
                for name in self.funds:
                    tok_score = self._token_level_score(query_tokens, name)
                    if tok_score >= 80:
                        all_matches[name] = max(all_matches.get(name, 0), tok_score)

            # Apply secondary scoring to increase differentiation
            # Token overlap + length penalty separates best matches from generic ones
            refined_matches = {}
            for name, fuzzy_score in all_matches.items():
                secondary_score = self._calculate_secondary_score(fuzzy_score, query, name)
                refined_matches[name] = secondary_score

            # Sort by refined score and get top matches
            sorted_matches = sorted(refined_matches.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Show top matches for debugging
            print(f"\nFund Search - Top 5 matches:")
            for i, (fund_name, score) in enumerate(sorted_matches, 1):
                ticker = self.funds[fund_name]
                print(f"  {i}. '{fund_name}' ({ticker}) - Score: {score:.1f}")
            
            # Get matches within 5 points of top score (up to 5) that meet threshold
            if sorted_matches:
                top_score = sorted_matches[0][1]
                # Raise threshold for aggregate queries to avoid injecting irrelevant entities
                score_threshold = 87 if is_aggregate else 82
                
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
            
        # D. Check Company names (fuzzy — same approach as fund matching)
        if search_companies and self.companies:
            normalized_query = self._normalize_for_matching(query)
            normalized_companies = {self._normalize_for_matching(name): name for name in self.companies.keys()}
            normalized_companies = {k: v for k, v in normalized_companies.items() if k}

            token_matches = process.extract(
                normalized_query,
                normalized_companies.keys(),
                scorer=fuzz.token_set_ratio,
                limit=10,
            )
            partial_matches = process.extract(
                normalized_query,
                normalized_companies.keys(),
                scorer=fuzz.partial_ratio,
                limit=10,
            )

            all_company_matches: Dict[str, float] = {}
            for match, score, _ in token_matches:
                original_name = normalized_companies[match]
                all_company_matches[original_name] = max(all_company_matches.get(original_name, 0), score)
            for match, score, _ in partial_matches:
                original_name = normalized_companies[match]
                boosted = min(100, score * 1.1)
                all_company_matches[original_name] = max(all_company_matches.get(original_name, 0), boosted)

            # Token-level pass for company typos
            # e.g. "Amazoon" → "Amazon" (fuzz.ratio ≈ 84), "Microsft" → "Microsoft" (≈ 93)
            query_tokens = self._extract_candidate_tokens(query)
            if query_tokens:
                for name in self.companies:
                    tok_score = self._token_level_score(query_tokens, name)
                    if tok_score >= 80:
                        all_company_matches[name] = max(all_company_matches.get(name, 0), tok_score)

            # Apply secondary scoring for companies as well
            refined_company_matches = {}
            for name, fuzzy_score in all_company_matches.items():
                secondary_score = self._calculate_secondary_score(fuzzy_score, query, name)
                refined_company_matches[name] = secondary_score

            sorted_company_matches = sorted(refined_company_matches.items(), key=lambda x: x[1], reverse=True)[:5]

            if self.debug:
                print(f"\nCompany Search - Top 5 matches:")
                for i, (company_name, score) in enumerate(sorted_company_matches, 1):
                    ticker = self.companies[company_name]
                    print(f"  {i}. '{company_name}' ({ticker}) - Score: {score:.1f}")

            company_score_threshold = 70
            if sorted_company_matches:
                top_score = sorted_company_matches[0][1]
                if top_score >= company_score_threshold:
                    matches_within_range = [
                        (name, score) for name, score in sorted_company_matches
                        if score >= top_score - 5
                    ][:3]

                    for company_name, score in matches_within_range:
                        ticker = self.companies[company_name]
                        if company_name not in resolved:
                            resolved[company_name] = {
                                "type": "Company",
                                "score": score,
                                "ticker": ticker,
                            }
                        if ticker and ticker not in resolved:
                            resolved[ticker] = {
                                "type": "Ticker",
                                "score": score,
                                "company": company_name,
                            }

        # E. Ticker detection handled in Step 1.5 (pre-flight, before fuzzy matching)
        # This ensures exact tickers always get 100.0 score and skip the noisy fund fuzzy search.

        return resolved

    def get_ticker_for_fund(self, fund_name: str) -> Optional[str]:
        """Get ticker symbol for a fund name"""
        return self.funds.get(fund_name)
