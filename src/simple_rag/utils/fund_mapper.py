import logging
from typing import List
import pandas as pd
import re
from src.simple_rag.models.fund import FundData, FinancialHighlights

logger = logging.getLogger(__name__)


def compute_annual_returns(df: pd.DataFrame):
    """Compute annual returns for the first fund column in a performance DataFrame.

    Automatically detects format:
    - Text Month Year (e.g., "Jan 23") → Parses as Month 2023
    - Quarterly/Monthly (MM/DD/YY format) → uses year-end values
    - Annual data (YYYY format) → uses consecutive years

    Args:
        df: DataFrame with columns [date/year column, fund values...]

    Returns:
        Dictionary: {year: return_percentage}
    """
    annual_returns = {}

    # Get the date column name (first column)
    date_col = df.columns[0]

    # Get the first fund column (second column in the dataframe)
    if len(df.columns) < 2:
        logger.debug("DataFrame doesn't have enough columns")
        return annual_returns

    fund_column = df.columns[1]

    # Make a copy to avoid warnings
    df = df.copy()

    # --- 1. DETECT FORMAT ---
    first_value = str(df[date_col].iloc[0]).strip()

    # Check for "Jan 23" format
    is_text_month_year = bool(re.match(r'^[A-Za-z]{3}\s+\d{2}$', first_value))

    # Check for Standard Date format (MM/DD/YY)
    is_standard_date = '/' in first_value or '-' in first_value

    is_month_hyphen_year = bool(re.match(r'^[A-Za-z]{3}-\d{2}$', first_value))

    # Group them as "Date Logic" vs "Year Integer Logic"
    use_date_logic = is_text_month_year or is_standard_date

    logger.debug(
        f"Detected format: "
        f"{'Text Month-Year' if is_text_month_year else ('Standard Date' if is_standard_date else 'Year (YYYY)')}"
    )

    if use_date_logic:
        # --- 2. HANDLE DATES ---
        try:
            if is_text_month_year:
                df['parsed_date'] = pd.to_datetime(df[date_col], format='%b %y', errors='coerce')
            elif is_month_hyphen_year:
                df['parsed_date'] = pd.to_datetime(df[date_col], format='%b-%y', errors='coerce')
            else:
                df['parsed_date'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')

            if df['parsed_date'].isna().all():
                logger.debug("Failed to parse dates")
                return annual_returns

            df['year'] = df['parsed_date'].dt.year
            df['month'] = df['parsed_date'].dt.month

            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)

            year_end_df = df.groupby('year').last().reset_index()
            years = sorted(year_end_df['year'].unique())

            if len(years) < 2:
                logger.debug(f"Not enough years to calculate returns. Found: {years}")
                return annual_returns

            logger.debug(f"Found year-end data for years: {years}")

            clean_vals = year_end_df[fund_column].astype(str).str.replace(r'[\$,]', '', regex=True)
            year_end_df[fund_column] = pd.to_numeric(clean_vals, errors='coerce')
            year_end_df['return_pct'] = year_end_df[fund_column].pct_change() * 100

            valid_returns = year_end_df.dropna(subset=['return_pct'])
            for _, row in valid_returns.iterrows():
                ret = row['return_pct']
                if pd.notna(ret) and ret not in (float('inf'), float('-inf')):
                    annual_returns[str(int(row['year']))] = round(ret, 2)
                    logger.debug(f"  {int(row['year'])} Return: {ret:.2f}%")

        except Exception as e:
            logger.debug(f"Error parsing date format: {e}")
            return annual_returns

    else:
        # --- 3. HANDLE YEAR INTEGERS (YYYY) ---
        try:
            df['year_extracted'] = df[date_col].astype(str).str.extract(r'(\d{4})')[0]
            df = df.dropna(subset=['year_extracted'])

            if len(df) == 0:
                logger.debug("No valid year data found")
                return annual_returns

            df['year_extracted'] = df['year_extracted'].astype(int)

            year_end_df = df.groupby('year_extracted').last().reset_index()
            years = sorted(year_end_df['year_extracted'].unique())

            if len(years) < 2:
                logger.debug(f"Not enough years to calculate returns. Found: {years}")
                return annual_returns

            logger.debug(f"Found years: {years}")

            clean_vals = year_end_df[fund_column].astype(str).str.replace(r'[\$,]', '', regex=True)
            year_end_df[fund_column] = pd.to_numeric(clean_vals, errors='coerce')
            year_end_df['return_pct'] = year_end_df[fund_column].pct_change() * 100

            valid_returns = year_end_df.dropna(subset=['return_pct'])
            for _, row in valid_returns.iterrows():
                ret = row['return_pct']
                if pd.notna(ret) and ret not in (float('inf'), float('-inf')):
                    annual_returns[str(int(row['year_extracted']))] = round(ret, 2)
                    logger.debug(f"  {int(row['year_extracted'])} Return: {ret:.2f}%")

        except Exception as e:
            logger.debug(f"Error parsing year format: {e}")
            return annual_returns

    return annual_returns


def enrich_funds_with_annual_returns(funds: list, financial_highlights_dfs: list, debug: bool = False) -> None:
    """Compute annual returns from performance tables and merge financial highlights.

    This is the post-processing step that was previously done in notebook code:
    1. Computes annual returns from each fund's performance table.
    2. Creates stub ``FinancialHighlights`` for years not already present.
    3. Merges all financial-highlights DataFrames and calls
       :func:`process_funds_financial_highlights` to enrich the funds.

    Args:
        funds: List of :class:`FundData` objects (modified in place).
        financial_highlights_dfs: List of DataFrames from
            ``NCSRExtractor.get_financial_highlights()``.
        debug: If True, prints verbose processing logs via logger.info.
    """
    performance_tickers = set()

    for fund in funds:
        if fund.performance_table is not None:
            performance_tickers.add(fund.ticker)

    # Step 1: compute annual returns + fill missing highlights
    for fund in funds:
        if fund.ticker in performance_tickers and fund.performance_table is not None:
            returns = compute_annual_returns(fund.performance_table)
            fund.annual_returns = returns
            if debug:
                logger.info(f"[{fund.ticker}] Annual returns: {returns}")

            if fund.financial_highlights is None:
                fund.financial_highlights = {}

            for year, return_val in returns.items():
                if year not in fund.financial_highlights:
                    fund.financial_highlights[year] = FinancialHighlights(
                        total_return=return_val,
                        turnover=0.0,
                        expense_ratio=0.0,
                        net_assets=0.0,
                        net_assets_value_beginning=0.0,
                        net_assets_value_end=0.0,
                        net_income_ratio=0.0,
                    )

    # Step 2: merge highlights DataFrames and enrich
    valid_dfs = [df for df in financial_highlights_dfs if df is not None and not df.empty]
    if valid_dfs:
        combined_df = pd.concat(valid_dfs, ignore_index=True)
        process_funds_financial_highlights(funds_list=funds, returns_dataframe=combined_df, debug=debug)


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

def process_funds_financial_highlights(
    funds_list: List[FundData], 
    returns_dataframe: pd.DataFrame,
    debug: bool = False,
) -> None:
    """
    Enriches a list of FundData objects with financial highlights.
    
    Args:
        funds_list (List[FundData]): A list of FundData objects to process.
        returns_dataframe (pd.DataFrame): The DataFrame containing financial highlights data.
        debug (bool): If True, enables verbose logger.info output.
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
            if debug:
                logger.info(f"  No name matches found for ticker: {fund_obj.ticker} {fund_obj.name}")
            continue
        
        logger.debug(f"  Found {len(name_matches)} name matches")
        
        share_class_obj = fund_obj.share_class
        share_class_str = share_class_obj.value if hasattr(share_class_obj, 'value') else str(share_class_obj)
        norm_share_class = normalize_finance_string(share_class_str)
        
        if debug:
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
                if debug:
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
                
                if debug:
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

    if debug:
        logger.info(f"count: {count}")
        logger.info(f"Total funds: {len(funds_list)}")
