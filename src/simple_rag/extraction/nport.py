import json
import re
from dataclasses import asdict, is_dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from ..models.fund import PortfolioHolding
from rapidfuzz import fuzz, process


# Comprehensive asset category to sector mapping
ASSET_TO_SECTOR_MAP = {
    # Equities
    "EC": "Equity - Common Stock",
    "EP": "Equity - Preferred Stock",
    "EF": "Equity - Fund",
    
    # Fixed Income
    "DBT": "Fixed Income - Corporate Debt",
    "DBTS": "Fixed Income - Structured Debt",
    "UST": "Fixed Income - US Treasury",
    "USTN": "Fixed Income - US Treasury Note",
    "ABS": "Fixed Income - Asset Backed Security",
    "MBS": "Fixed Income - Mortgage Backed Security",
    "MUNI": "Fixed Income - Municipal Bond",
    "FORDEBT": "Fixed Income - Foreign Debt",
    
    # Cash & Equivalents
    "MMF": "Cash - Money Market Fund",
    "CASH": "Cash - Cash Equivalent",
    "STIV": "Cash - Short Term Investment",
    
    # Derivatives
    "DE": "Derivative - Equity",
    "DF": "Derivative - Fixed Income",
    "DC": "Derivative - Commodity",
    "DFX": "Derivative - Foreign Exchange",
    "DI": "Derivative - Interest Rate",
    "DO": "Derivative - Other",
    "SWPS": "Derivative - Swap",
    "FUT": "Derivative - Future",
    "OPT": "Derivative - Option",
    "WAR": "Derivative - Warrant",
    
    # Other
    "REPO": "Repurchase Agreement",
    "RVRPD": "Reverse Repurchase Agreement",
    "LLC": "Limited Liability Company",
    "LP": "Limited Partnership",
    "OTHER": "Other Investment",
}

# Issuer category descriptions
ISSUER_CATEGORY_MAP = {
    "CORP": "Corporate",
    "USGOVT": "US Government",
    "FORGOV": "Foreign Government",
    "MUNI": "Municipal",
    "SUPRA": "Supranational",
    "OTHER": "Other",
}


def process_portfolio_holdings(investments: List) -> List[PortfolioHolding]:
    """
    Extract and enrich portfolio information from InvestmentOrSecurity objects
    
    Args:
        investments: List of InvestmentOrSecurity objects from NPORT filing
        
    Returns:
        List of enriched PortfolioHolding objects
    """
    holdings = []
    
    for inv in investments:
        # Extract ISIN safely
        isin = ''
        if inv.identifiers:
            if hasattr(inv.identifiers, 'isin') and inv.identifiers.isin:
                isin = inv.identifiers.isin
        
        # Extract ticker safely
        ticker = None
        if inv.identifiers and hasattr(inv.identifiers, 'ticker'):
            ticker = inv.identifiers.ticker
        
        # Map asset category to readable sector
        sector = ASSET_TO_SECTOR_MAP.get(inv.asset_category, inv.asset_category)
        
        # Map issuer category
        issuer_type = ISSUER_CATEGORY_MAP.get(inv.issuer_category, inv.issuer_category)
        
        # Create enriched holding
        holding = PortfolioHolding(
            name=inv.name,
            ticker=ticker,
            cusip=inv.cusip,
            isin=isin,
            lei=inv.lei if hasattr(inv, 'lei') else None,
            shares=float(inv.balance) if inv.balance else 0,
            market_value=float(inv.value_usd) if inv.value_usd else 0,
            weight_pct=float(inv.pct_value),
            asset_category=inv.asset_category,
            asset_category_desc=sector,
            issuer_category=inv.issuer_category,
            issuer_category_desc=issuer_type,
            country=inv.investment_country,
            currency=inv.currency_code if hasattr(inv, 'currency_code') else 'USD',
            payoff_profile=inv.payoff_profile if hasattr(inv, 'payoff_profile') else None,
            is_restricted=inv.is_restricted_security if hasattr(inv, 'is_restricted_security') else False,
            fair_value_level=inv.fair_value_level if hasattr(inv, 'fair_value_level') else None,
        )
        holdings.append(holding)
    
    return holdings


def holdings_to_df(holdings: List[PortfolioHolding]) -> pd.DataFrame:
    def _get(d: Dict[str, Any], key: str) -> Any:
        return d.get(key)

    def _row_from_mapping(m: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": _get(m, "name"),
            "ticker": _get(m, "ticker"),
            "cusip": _get(m, "cusip"),
            "isin": _get(m, "isin"),
            "lei": _get(m, "lei"),
            "shares": _get(m, "shares"),
            "market_value": _get(m, "market_value"),
            "weight_pct": _get(m, "weight_pct"),
            "currency": _get(m, "currency"),
            "asset_category": _get(m, "asset_category"),
            "asset_category_desc": _get(m, "asset_category_desc"),
            "issuer_category": _get(m, "issuer_category"),
            "issuer_category_desc": _get(m, "issuer_category_desc"),
            "country": _get(m, "country"),
            "payoff_profile": _get(m, "payoff_profile"),
            "is_restricted": _get(m, "is_restricted"),
            "fair_value_level": _get(m, "fair_value_level"),
            "sector": _get(m, "sector"),
        }

    rows: List[Dict[str, Any]] = []
    for i, h in enumerate(holdings):
        if isinstance(h, str):
            rows.append(_row_from_mapping({"name": h}))
            continue
        if isinstance(h, dict):
            rows.append(_row_from_mapping(h))
            continue
        if is_dataclass(h):
            rows.append(_row_from_mapping(asdict(h)))
            continue
        if hasattr(h, "name"):
            rows.append(
                {
                    "name": getattr(h, "name", None),
                    "ticker": getattr(h, "ticker", None),
                    "cusip": getattr(h, "cusip", None),
                    "isin": getattr(h, "isin", None),
                    "lei": getattr(h, "lei", None),
                    "shares": getattr(h, "shares", None),
                    "market_value": getattr(h, "market_value", None),
                    "weight_pct": getattr(h, "weight_pct", None),
                    "currency": getattr(h, "currency", None),
                    "asset_category": getattr(h, "asset_category", None),
                    "asset_category_desc": getattr(h, "asset_category_desc", None),
                    "issuer_category": getattr(h, "issuer_category", None),
                    "issuer_category_desc": getattr(h, "issuer_category_desc", None),
                    "country": getattr(h, "country", None),
                    "payoff_profile": getattr(h, "payoff_profile", None),
                    "is_restricted": getattr(h, "is_restricted", None),
                    "fair_value_level": getattr(h, "fair_value_level", None),
                    "sector": getattr(h, "sector", None),
                }
            )
            continue

        raise TypeError(
            f"holdings_to_df expected each element to be a PortfolioHolding/dict/dataclass/str; got {type(h)} at index {i}"
        )

    return pd.DataFrame(
        rows,
        columns=[
            "name",
            "ticker",
            "cusip",
            "isin",
            "lei",
            "shares",
            "market_value",
            "weight_pct",
            "currency",
            "asset_category",
            "asset_category_desc",
            "issuer_category",
            "issuer_category_desc",
            "country",
            "payoff_profile",
            "is_restricted",
            "fair_value_level",
            "sector",
        ],
    )


def _normalize_company_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(the|and)\b", " ", s)
    s = re.sub(
        r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|sa|nv|ag|holdings|holding|group|class)\b",
        " ",
        s,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_company_tickers_json(path: str | Path) -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, Dict[str, Any]] = {}
    for _, rec in raw.items():
        title = rec.get("title")
        ticker = rec.get("ticker")
        if not title or not ticker:
            continue
        norm = _normalize_company_name(title)
        if not norm:
            continue
        out[norm] = {
            "ticker": ticker,
            "title": title,
            "cik_str": rec.get("cik_str"),
        }

    return out


def _build_company_prefix_index(company_map: Dict[str, Dict[str, Any]], prefix_len: int = 6) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for norm_name in company_map.keys():
        key = norm_name[:prefix_len]
        if not key:
            continue
        idx.setdefault(key, []).append(norm_name)
    return idx


def _best_fuzzy_match(
    query_norm: str,
    candidates: List[str],
) -> Tuple[Optional[str], float]:
    best_name: Optional[str] = None
    best_score = 0.0

    for cand in candidates:
        score = SequenceMatcher(None, query_norm, cand).ratio()
        if score > best_score:
            best_score = score
            best_name = cand

    return best_name, best_score


def match_company_ticker(
    holding_name: str,
    company_map: Dict[str, Dict[str, Any]],
    company_prefix_index: Optional[Dict[str, List[str]]] = None,
    min_similarity: float = 0.92,
    prefix_len: int = 6,
    stats: Optional[Dict[str, int]] = None, # <--- NEW OPTIONAL ARGUMENT
) -> Tuple[Optional[str], float, Optional[str]]:
    
    query_norm = _normalize_company_name(holding_name)
    if not query_norm:
        return None, 0.0, None

    # 1. Exact Match
    if query_norm in company_map:
        rec = company_map[query_norm]
        if stats is not None: stats['exact'] += 1
        return rec.get("ticker"), 1.0, rec.get("title")

    # 2. Prefix Match (Existing Logic)
    if company_prefix_index is None:
        company_prefix_index = _build_company_prefix_index(company_map, prefix_len=prefix_len)

    key = query_norm[:prefix_len]
    candidates = company_prefix_index.get(key, [])
    
    # Fallback to scanning keys if index misses but key exists
    if not candidates and key:
        candidates = [n for n in company_map.keys() if n.startswith(key[:3])]
        candidates = candidates[:5000]

    best_name, best_score = _best_fuzzy_match(query_norm, candidates)
    
    if best_name and best_score >= min_similarity:
        rec = company_map[best_name]
        if stats is not None: stats['prefix'] += 1
        return rec.get("ticker"), best_score, rec.get("title")

    
    extraction = process.extractOne(
        query_norm, 
        company_map.keys(), 
        scorer=fuzz.token_set_ratio, 
        score_cutoff=min_similarity * 100
    )

    if extraction:
        match_name, score, _ = extraction
        rec = company_map[match_name]
        if stats is not None: stats['rapidfuzz'] += 1
        return rec.get("ticker"), score / 100.0, rec.get("title")

    if stats is not None: stats['none'] += 1
    return None, best_score, None


def enrich_holdings_with_company_tickers(
    holdings: List[PortfolioHolding],
    company_map: Dict[str, Dict[str, Any]],
    min_similarity: float = 0.75,
    verbose: bool = False,
) -> pd.DataFrame:
    company_prefix_index = _build_company_prefix_index(company_map)

    rows: List[Dict[str, Any]] = []
    match_count = 0
    for h in holdings:
        ticker_before = h.ticker
        matched_ticker, score, matched_title = match_company_ticker(
            holding_name=h.name,
            company_map=company_map,
            company_prefix_index=company_prefix_index,
            min_similarity=min_similarity,
        )

        updated = False
        if (ticker_before is None or str(ticker_before).strip() == "") and matched_ticker:
            h.ticker = matched_ticker
            updated = True
            match_count += 1

        rows.append(
            {
                "holding_name": h.name,
                "ticker_before": ticker_before,
                "ticker_after": h.ticker,
                "matched_ticker": matched_ticker,
                "matched_title": matched_title,
                "similarity": score,
                "updated": updated,
            }
        )

    df = pd.DataFrame(rows)

    if verbose:
        found = df[df["matched_ticker"].notna() & (df["matched_ticker"] != "")].copy()
        unique_pairs = (
            found[["matched_ticker", "matched_title"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["matched_ticker", "matched_title"], na_position="last")
        )

        print(f"Number of matched holdings: {match_count}")

    return df


class NPortProcessor:
    def __init__(
        self,
        company_tickers_json_path: Optional[str | Path] = None,
        min_similarity: float = 0.92,
    ):
        self.min_similarity = min_similarity
        self.company_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._company_prefix_index: Optional[Dict[str, List[str]]] = None

        if company_tickers_json_path is not None:
            self.load_company_tickers(company_tickers_json_path)

    def load_company_tickers(self, path: str | Path) -> Dict[str, Dict[str, Any]]:
        self.company_map = load_company_tickers_json(path)
        self._company_prefix_index = _build_company_prefix_index(self.company_map)
        return self.company_map

    def process_holdings(self, investments: List) -> List[PortfolioHolding]:
        return process_portfolio_holdings(investments)

    def to_df(self, holdings: List[PortfolioHolding]) -> pd.DataFrame:
        return holdings_to_df(holdings)

    def match_ticker(self, holding_name: str) -> Tuple[Optional[str], float, Optional[str]]:
        if not self.company_map:
            raise ValueError("company_map is not loaded. Call load_company_tickers(...) first.")

        return match_company_ticker(
            holding_name=holding_name,
            company_map=self.company_map,
            company_prefix_index=self._company_prefix_index,
            min_similarity=self.min_similarity,
        )

    def enrich_tickers(self, holdings: List[PortfolioHolding], verbose: bool = False) -> pd.DataFrame:
        if not self.company_map:
            raise ValueError("company_map is not loaded. Call load_company_tickers(...) first.")

        return enrich_holdings_with_company_tickers(
            holdings=holdings,
            company_map=self.company_map,
            min_similarity=self.min_similarity,
            verbose=verbose,
        )