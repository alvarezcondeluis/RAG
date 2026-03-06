import logging
from typing import List, Optional
import pandas as pd
import re

from src.simple_rag.models.fund import FundData, FinancialHighlights

def clean_financial_number(val):
    """
    Parses financial strings like '23.19 %(b)' or '(24.82 )%'.
    - Extracts the numerical value.
    - Handles (12.34) as negative -12.34.
    - Ignores footnote markers like (a), (b).
    - Removes %, $, and commas.
    """
    if pd.isna(val) or val is None:
        return 0.0
    
    s = str(val).strip()
    match = re.search(r'(\d{1,3}(?:,\d{3})*\.?\d*|\d*\.?\d+)', s)
    if not match:
        return 0.0
        
    num_str = match.group(0)
    is_negative = s.startswith('(') or s.startswith('-')
    
    try:
        clean_num = float(num_str.replace(',', ''))
        return -clean_num if is_negative else clean_num
    except ValueError:
        return 0.0


def normalize_finance_string(s: str) -> str:
    """
    Normalizes a string for robust matching:
    - Lowercases.
    - Removes common prefixes (Vanguard, iShares).
    - Removes all non-alphanumeric characters.
    - Collapses multiple spaces.
    """
    if not s or pd.isna(s):
        return ""
    s = str(s).lower()
    # Remove provider names and common symbols
    s = s.replace("vanguard", "").replace("ishares", "").replace("™", "").replace("®", "")
    # Remove all non-alphanumeric characters (handles hyphens, commas, etc.)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return " ".join(s.split())


logger = logging.getLogger(__name__)

def process_funds_financial_highlights(
    funds_list: List[FundData], 
    returns_dataframe: pd.DataFrame,
) -> None:
    """
    Enriches a list of FundData objects with financial highlights.
    
    Args:
        funds_list (List[FundData]): A list of FundData objects to process.
        returns_dataframe (pd.DataFrame): The DataFrame containing financial highlights data.
    """
    if not funds_list or returns_dataframe.empty:
        logger.warning("Either funds_list is empty or returns_dataframe is empty. Nothing to enrich.")
        return

    returns_lookup = returns_dataframe.copy()

    # Pre-normalize fund names in the lookup table for faster matching
    if 'fund_name' in returns_lookup.columns:
        returns_lookup['normalized_fund_name'] = returns_lookup['fund_name'].apply(normalize_finance_string)

    # Cleaning the numeric columns
    numeric_columns = [
        'portfolio_turnover', 'expense_ratio', 'net_assets', 
        'nav_beginning', 'nav_end', 'net_income_ratio', 'distribution_shares', 'total_return'
    ]
    
    for col in numeric_columns:
        if col in returns_lookup.columns:
            returns_lookup[f'{col}_clean'] = returns_lookup[col].apply(clean_financial_number)

    count = 0
    for fund_obj in funds_list:
        logger.debug(f"Processing fund object: {fund_obj.name} - {fund_obj.share_class}")
        
        if not fund_obj.name:
            continue
            
        # Standardize search name
        safe_name = normalize_finance_string(fund_obj.name)
        logger.debug(f"Normalized fund name for matching: '{safe_name}'")
        
        # Find matching rows based on fund name
        if 'fund_name' not in returns_lookup.columns:
            logger.error("'fund_name' column missing in DataFrame. Cannot match funds by name.")
            return
            
        # Try exact normalized match first
        name_matches = returns_lookup[returns_lookup['normalized_fund_name'] == safe_name]
        
        # Fallback: if no match, try substring match (important for names with tickers in them)
        if len(name_matches) == 0:
            name_matches = returns_lookup[
                returns_lookup['normalized_fund_name'].str.contains(safe_name, case=False, na=False) |
                returns_lookup['fund_name'].astype(str).str.lower().str.contains(safe_name, case=False, na=False)
            ]
        
        if len(name_matches) == 0:
            logger.info(f"  No name matches found for ticker: {fund_obj.ticker} {fund_obj.name}")
            continue
        
        logger.debug(f"  Found {len(name_matches)} name matches")
        
        share_class_obj = fund_obj.share_class
        share_class_str = share_class_obj.value if hasattr(share_class_obj, 'value') else str(share_class_obj)
        norm_share_class = normalize_finance_string(share_class_str)
        
        logger.info(f"Share class: '{share_class_str}' (normalized: '{norm_share_class}')")
        
        if 'share_class' in name_matches.columns:
            # 1. Try match by normalized share class
            def is_share_class_match(row_val):
                row_norm = normalize_finance_string(str(row_val))
                if not norm_share_class or norm_share_class in ['none', 'nan', 'other', 'unknown']:
                    return not row_norm or row_norm in ['none', 'nan', 'other', 'unknown']
                
                # Check for substring match in either direction
                if norm_share_class in row_norm or row_norm in norm_share_class:
                    return True
                # Special handling for ETFs
                if 'etf' in norm_share_class and ('etf' in row_norm or 'shares' in row_norm):
                    return True
                return False

            share_class_matches = name_matches[name_matches['share_class'].apply(is_share_class_match)]
            
            # 2. Fallback: if no specific match, look for rows with empty share class
            if len(share_class_matches) == 0:
                fallback_matches = name_matches[
                    name_matches['share_class'].isna() | 
                    name_matches['share_class'].astype(str).str.lower().isin(['none', 'nan', '<na>', '', 'unknown'])
                ]
                # Only use fallback if it represents the majority of records or we have no other choice
                if len(fallback_matches) > 0:
                    share_class_matches = fallback_matches

            # 3. Last Resort: if still no match, take all name matches (better to show something than nothing)
            if len(share_class_matches) == 0:
                logger.info(f"  No specific share class matches found for '{share_class_str}'. Falling back to all name matches.")
                logger.info(f"  Available share classes in data: {name_matches['share_class'].unique()}")
                share_class_matches = name_matches
        else:
            share_class_matches = name_matches
        
        logger.debug(f"  Found {len(share_class_matches)} matching records")
        count += 1
        
        if fund_obj.financial_highlights is None:
            fund_obj.financial_highlights = {}
            
        fund_header_logged = False
            
        # Remove duplicate years from the dataframe before processing
        if 'year' in share_class_matches.columns:
            share_class_matches = share_class_matches.drop_duplicates(subset=['year'], keep='first')
            
        # Process and assign matching returns
        for _, row in share_class_matches.iterrows():
            if 'year' not in row:
                continue
            
            year = str(row['year']).strip()
            
            def _get_val(key, default=0.0):
                val = row.get(key, default)
                return default if pd.isna(val) else val
            
            try:
                highlights = FinancialHighlights(
                    turnover=float(_get_val('portfolio_turnover_clean', 0.0)),
                    expense_ratio=float(_get_val('expense_ratio_clean', 0.0)),
                    total_return=float(_get_val('total_return_clean', 0.0)),
                    net_assets=float(_get_val('net_assets_clean', 0.0)),
                    net_assets_value_beginning=float(_get_val('nav_beginning_clean', 0.0)),
                    net_assets_value_end=float(_get_val('nav_end_clean', 0.0)),
                    net_income_ratio=float(_get_val('net_income_ratio_clean', 0.0))
                )
                fund_obj.financial_highlights[year] = highlights
                
                if not fund_header_logged:
                    logger.info(f"[{fund_obj.ticker}] {fund_obj.name} - {share_class_str}")
                    fund_header_logged = True
                    
                logger.info(
                    f"  ├─ {year}: Total Return = {highlights.total_return}%, "
                    f"Expense Ratio = {highlights.expense_ratio}%, "
                    f"Net Assets = {highlights.net_assets}, "
                    f"Net Income Ratio = {highlights.net_income_ratio}, "
                    f"Turnover = {highlights.turnover}, "
                    f"NAV Beginning = {highlights.net_assets_value_beginning}, "
                    f"NAV End = {highlights.net_assets_value_end}"
                )
            except Exception as e:
                logger.error(f"Error mapping highlights for {fund_obj.ticker} year {year}: {e}")

    logger.info(f"count: {count}")
    logger.info(f"Total funds: {len(funds_list)}")
