import io
from typing import List, Dict
from bs4 import BeautifulSoup
from ..utils.utils import XBRLUtils
from ..models.fund import FundData, ShareClassType
import pandas as pd
import re
from IPython.display import display, Markdown

def compute_annual_returns(df: pd.DataFrame):
        """
        Compute annual returns for the first fund column in the performance DataFrame.
        Automatically detects format:
        - Text Month Year (e.g., "Jan 23") -> Parses as Month 2023
        - Quarterly/Monthly (MM/DD/YY format) -> uses year-end values
        - Annual data (YYYY format) -> uses consecutive years
        
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
            print("DataFrame doesn't have enough columns")
            return annual_returns
        
        fund_column = df.columns[1]
        
        # Make a copy to avoid warnings
        df = df.copy()
        
        # --- 1. DETECT FORMAT ---
        first_value = str(df[date_col].iloc[0]).strip()
        
        # Check for "Jan 23" format: Starts with 3 letters + space + 2 digits
        # Regex explanation: ^[A-Za-z]{3} (Month) \s+ (Space) \d{2}$ (2-digit Year)
        is_text_month_year = bool(re.match(r'^[A-Za-z]{3}\s+\d{2}$', first_value))
        
        # Check for Standard Date format (MM/DD/YY)
        is_standard_date = '/' in first_value or '-' in first_value
        
        is_month_hyphen_year = bool(re.match(r'^[A-Za-z]{3}-\d{2}$', first_value))
        

        # Group them as "Date Logic" vs "Year Integer Logic"
        use_date_logic = is_text_month_year or is_standard_date
        
        print(f"Detected format: {'Text Month-Year' if is_text_month_year else ('Standard Date' if is_standard_date else 'Year (YYYY)')}")
        
        if use_date_logic:
            # --- 2. HANDLE DATES ---
            try:
                # Explicitly handle "Jan 23" to ensure it parses as Year 2023, not Day 23
                if is_text_month_year:
                    # %b = Abbreviated month (Jan), %y = 2-digit year (23)
                    df['parsed_date'] = pd.to_datetime(df[date_col], format='%b %y', errors='coerce')
                elif is_month_hyphen_year:
                    # %b = Abbreviated month (Jan), %y = 2-digit year (23)
                    df['parsed_date'] = pd.to_datetime(df[date_col], format='%b-%y', errors='coerce')
                else:
                    # Let pandas guess for standard MM/DD/YYYY
                    df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
                
                if df['parsed_date'].isna().all():
                    print("Failed to parse dates")
                    return annual_returns
                
                df['year'] = df['parsed_date'].dt.year
                df['month'] = df['parsed_date'].dt.month
                
                # Drop rows where year couldn't be extracted
                df = df.dropna(subset=['year'])
                df['year'] = df['year'].astype(int)
                
                # Always use the last available data point for each year
                # This handles both complete years (Dec data) and incomplete years (e.g., Nov-24)
                year_end_df = df.groupby('year').last().reset_index()
                
                # Get unique years
                years = sorted(year_end_df['year'].unique())
                
                if len(years) < 2:
                    print(f"Not enough years to calculate returns. Found: {years}")
                    return annual_returns
                
                print(f"Found year-end data for years: {years}")
                
                # Calculate returns using the last available data point for each year
                # Formula: (Year T Value - Year T-1 Value) / Year T-1 Value
                # Note: This now works for incomplete years (e.g., Jan-24 to Nov-24)
                for i in range(len(years) - 1):
                    prev_year = years[i]
                    current_year = years[i + 1]
                    
                    start_data = year_end_df[year_end_df['year'] == prev_year].iloc[0]
                    end_data = year_end_df[year_end_df['year'] == current_year].iloc[0]
                    
                    try:
                        start_val_str = str(start_data[fund_column])
                        end_val_str = str(end_data[fund_column])
                        
                        if start_val_str in ['nan', 'None', ''] or end_val_str in ['nan', 'None', '']:
                            continue
                        
                        start_val = float(start_val_str.replace('$', '').replace(',', ''))
                        end_val = float(end_val_str.replace('$', '').replace(',', ''))
                        
                        if start_val == 0:
                            continue
                        
                        return_pct = ((end_val - start_val) / start_val) * 100
                        annual_returns[str(current_year)] = round(return_pct, 2)
                        
                        print(f"  {current_year} Return: ${start_val:,.2f} -> ${end_val:,.2f} = {return_pct:.2f}%")
                        
                    except (ValueError, ZeroDivisionError) as e:
                        print(f"Error calculating return in {current_year}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error parsing date format: {e}")
                return annual_returns
        
        else:
            # --- 3. HANDLE YEAR INTEGERS (YYYY) ---
            # (This block remains largely the same as your original logic)
            try:
                df['year_extracted'] = df[date_col].astype(str).str.extract(r'(\d{4})')[0]
                df = df.dropna(subset=['year_extracted'])
                
                if len(df) == 0:
                    print("No valid year data found")
                    return annual_returns
                
                df['year_extracted'] = df['year_extracted'].astype(int)
                
                # Use the last available row for each year
                # (Assumes data is sorted chronologically or the last entry is year-end)
                year_end_df = df.groupby('year_extracted').last().reset_index()
                years = sorted(year_end_df['year_extracted'].unique())
                
                if len(years) < 2:
                    print(f"Not enough years to calculate returns. Found: {years}")
                    return annual_returns
                
                print(f"Found years: {years}")
                
                for i in range(len(years) - 1):
                    prev_year = years[i]
                    current_year = years[i + 1]
                    
                    start_data = year_end_df[year_end_df['year_extracted'] == prev_year].iloc[0]
                    end_data = year_end_df[year_end_df['year_extracted'] == current_year].iloc[0]
                    
                    try:
                        start_val = float(str(start_data[fund_column]).replace('$', '').replace(',', ''))
                        end_val = float(str(end_data[fund_column]).replace('$', '').replace(',', ''))
                        
                        if start_val == 0: continue
                        
                        return_pct = ((end_val - start_val) / start_val) * 100
                        annual_returns[str(current_year)] = round(return_pct, 2)
                        print(f"  {current_year} Return: ${start_val:,.2f} -> ${end_val:,.2f} = {return_pct:.2f}%")
                        
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"Error parsing year format: {e}")
                return annual_returns
        
        return annual_returns

class BlackRockFiling:
    def __init__(self, html_content: str):
        self.soup = BeautifulSoup(html_content, 'lxml')
        self._clean_soup()
        self.report_date = self._get_report_date()
        self.registrant = self._get_registrant()

    def _clean_soup(self):
        """Remove hidden content once at init."""
        for hidden in self.soup.find_all(class_="ix_hid_content"):
            hidden.decompose()

    def _get_report_date(self):
        tag = self.soup.find(["ix:nonnumeric", "ix:nonfraction"], attrs={"name": "dei:DocumentPeriodEndDate"})
        return XBRLUtils.clean_text(tag)

    def _get_registrant(self):
        tag = self.soup.find(["ix:nonnumeric", "ix:nonfraction"], attrs={"name": "dei:EntityRegistrantName"})
        return XBRLUtils.clean_text(tag)

    def get_funds(self, vanguard: bool = False) -> List[FundData]:
        """Main entry point to get all funds."""
        funds = []
        # Find all unique funds based on FundName
        tags = self.soup.find_all("ix:nonnumeric", attrs={"name": "oef:FundName"})
        seen_contexts = set()
        for tag in tags:
            c_id = tag.get("contextref")

            if c_id and c_id not in seen_contexts:
                seen_contexts.add(c_id)
                fund_name = tag.text.strip()
                if vanguard and "Vanguard" not in fund_name: 
                    fund_name = "Vanguard " + fund_name
                print(f"Processing: {fund_name}")
                
                # Extract Data for this specific fund context
                fund = self._extract_single_fund(c_id, fund_name)
                funds.append(fund)
        
        return funds

    def get_cid(self, contextref: str) -> str:
        """Extracts CID from contextref"""
        return re.search(r'(C\d+)', contextref).group(1) if re.search(r'(C\d+)', contextref) else ""
    
    def _map_share_class(self, share_class_str: str) -> ShareClassType:
        """Map string share class to ShareClassType enum."""
        if not share_class_str:
            return ShareClassType.OTHER
        
        share_class_str = share_class_str.strip().lower()
        
        # Map common variations to enum values
        if "admiral" in share_class_str:
            return ShareClassType.ADMIRAL
        elif "institutional plus" in share_class_str:
            return ShareClassType.INSTITUTIONAL_PLUS
        elif "institutional select" in share_class_str:
            return ShareClassType.INSTITUTIONAL_SELECT
        elif "institutional" in share_class_str:
            return ShareClassType.INSTITUTIONAL
        elif "investor" in share_class_str:
            return ShareClassType.INVESTOR
        elif "etf" in share_class_str:
            return ShareClassType.ETF
        else:
            return ShareClassType.OTHER
        
    def _extract_single_fund(self, c_id: str, name: str) -> FundData:
        """Extracts all data for a specific Context ID."""
        
        # Initialize the data object
        
        fund = FundData(name=name, context_id=c_id, report_date=self.report_date, registrant=self.registrant)
        
        # 1. Basic Tags (Scoped to this context)
        print("Extracting context: ", c_id)
        fund.ticker = self._get_value("dei:TradingSymbol", c_id)
        fund.expense_ratio = self._get_value("oef:ExpenseRatioPct", c_id)
        fund.costs_per_10k = self._get_value("oef:ExpensesPaidAmt", c_id)
        share_class_str = self._get_value("oef:ClassName", c_id)
        
        # Map string to enum
        if share_class_str:
            fund.share_class = self._map_share_class(share_class_str)
        elif "ETF" in name:
            fund.share_class = ShareClassType.ETF
        else:
            fund.share_class = ShareClassType.OTHER
        
        fund.report_date = self.report_date

        # 2. Embedded Values (Inside larger blocks)
        stats_block = "oef:AddlFundStatisticsTextBlock"
        fund.net_assets = self._get_embedded_value(stats_block, "us-gaap:AssetsNet", c_id)
        fund.turnover_rate = self._get_embedded_value(stats_block, "us-gaap:InvestmentCompanyPortfolioTurnover", c_id)
        fund.advisory_fees = self._get_embedded_value(stats_block, "oef:AdvisoryFeesPaidAmt", c_id)
        fund.n_holdings = self._get_embedded_value(stats_block, "oef:HoldingsCount", c_id)
        fund.security_exchange = self._get_value("dei:SecurityExchangeName", c_id)
        fund.performance_commentary = self._get_value("oef:FactorsAffectingPerfTextBlock", c_id)
        
        # 3. Tables (Holdings/Sectors)
        holdings_block = "oef:HoldingsTableTextBlock"
        
        tables = self._extract_tables(c_id, holdings_block)
        
        fund.top_holdings = tables.get("Top 10 Holdings")
        if fund.top_holdings is None:
            table = self._extract_tables(c_id, "oef:LargestHoldingsTableTextBlock")
            fund.top_holdings = table.get("Top 10 Holdings")
        
        fund.sector_allocation = tables.get("Sector Allocation")
        fund.portfolio_composition = tables.get("Portfolio Composition")
        fund.geographic_allocation = tables.get("Geographic Allocation")
        fund.industry_allocation = tables.get("Industry Allocation")
        fund.maturity_allocation = tables.get("Maturity Allocation")
        fund.issuer_allocation = tables.get("Issuer Allocation")
        fund.credit_rating = tables.get("Credit Rating")

        # 4. Performance Table
        performance_block = "oef:LineGraphTableTextBlock"
        tables = self._extract_tables(c_id, performance_block)
        fund.performance_table = tables.get("Performance Table")
        
        # 5. Average Annual Returns
        returns_block = "oef:AvgAnnlRtrTableTextBlock"
        tables = self._extract_tables(c_id, returns_block)
        fund.avg_annual_returns = tables.get("Average Annual Returns")
            
        return fund

    
    def _get_block(self, tag_name:str, c_id: str) -> str:
        """Finds a simple tag restricted by context."""
        tag = self.soup.find(attrs={"name": tag_name, "contextref": c_id})
        return tag

    def _get_value(self, tag_name: str, c_id: str) -> str:
        """Finds a simple tag restricted by context."""
        tag = self.soup.find(attrs={"name": tag_name, "contextref": c_id})
        if not tag: print("Tag not found: ", tag_name, c_id)
        return XBRLUtils.clean_text(tag)
    

    def get_financial_highlights(self) -> pd.DataFrame:
        """
        Extract Financial Highlights tables from HTML filing.
        Returns a DataFrame with performance data for each fund and share class.
        """
        results = []
        
        # Find all sections with "Financial Highlights" heading
        for heading in self.soup.find_all(['div', 'p'], string=re.compile(r'Financial Highlights', re.IGNORECASE)):
            # Navigate to find the fund name (usually appears before Financial Highlights)
            fund_name = None
            current = heading
            
            # Look backwards for fund name
            for _ in range(10):
                current = current.find_previous(['div', 'p', 'td'])
                if current and current.get_text(strip=True):
                    text = current.get_text(strip=True)
                    
                    # Fund names typically end with "Fund" or "Index Fund"
                    if ('Fund' in text and len(text) < 100) or 'Stock' in text or 'Bond' in text and 'Financial' not in text:
                        fund_name = text
                        
                        break
            
            # Find the table following this heading
            table = heading.find_next('table')
            if not table:
                continue
            
            # Extract share class name from table
            share_class = None
            share_class_row = table.find('td', string=re.compile(r'Shares|Share Class', re.IGNORECASE))
            if share_class_row:
                share_class = share_class_row.get_text(strip=True)
            
            if fund_name is not None and "ETF" in fund_name :
                share_class = "ETF Shares"

            # Parse the table
            try:
                df = pd.read_html(io.StringIO(str(table)))[0]
                df = df.fillna('')
               
               
                # Extract key metrics
                metrics = {
                    'fund_name': fund_name,
                    'share_class': share_class,
                    'nav_beginning': [],
                    'nav_end': [],
                    'total_return': [],
                    'expense_ratio': [],
                    'net_income_ratio': [],
                    'portfolio_turnover': [],
                    'years': []
                }
                
                # Extract specific rows
                for idx, row in df.iterrows():
                    first_col = str(row.iloc[0]).lower()
                    row_values = [str(x).strip() for x in row.iloc[1:].tolist() if str(x).strip()]
                    # Check if this row contains year headers
                    if row_values and all(re.search(r'\b2\d{3}\b', val) for val in row_values):
                        years = []
                        for val in row_values:
                            year_match = re.search(r'\b\d{4}\b', val)
                            if year_match:
                                years.append(year_match.group())
                        metrics['years'] = years
                    
                    if 'net asset value, beginning' in first_col:
                        metrics['nav_beginning'] = row.iloc[1:].tolist()
                    elif 'net assets,' in first_col:
                        metrics['net_assets'] = row.iloc[1:].tolist()
                    elif 'net asset value, end' in first_col:
                        metrics['nav_end'] = row.iloc[1:].tolist()
                    elif 'total return' in first_col and 'ratio' not in first_col:
                        metrics['total_return'] = row.iloc[1:].tolist()
                    elif 'ratio of total expenses' in first_col:
                        metrics['expense_ratio'] = row.iloc[1:].tolist()
                    elif 'ratio of net investment income' in first_col:
                        metrics['net_income_ratio'] = row.iloc[1:].tolist()
                    elif 'portfolio turnover' in first_col:
                        metrics['portfolio_turnover'] = row.iloc[1:].tolist()
                
                results.append(metrics)
                
            except Exception as e:
                print(f"Error parsing table for {fund_name} - {share_class}: {e}")
                continue
        
        # Convert to DataFrame
        return self._create_performance_dataframe(results)

    def get_financial_highlights2(self) -> pd.DataFrame:
        """
        Extracts Financial Highlights, handling tables where years are in columns 
        and metrics are in rows (transposed format).
        """
        # Dictionary to accumulate data by fund/share class
        fund_data = {}
        
        # 1. Find potential Financial Highlight tables
        text_nodes = self.soup.find_all(string=re.compile(r'Financial Highlights', re.IGNORECASE))
        seen_tables = set()
        
        print(f"Found {len(text_nodes)} potential Financial Highlights sections")
        
        for text_node in text_nodes:
            if text_node is None or len(str(text_node)) > 100:
                continue
            
            # Find the next table after this heading
            heading = text_node.parent
            table = heading.find_next('table')
            
            if not table or id(table) in seen_tables:
                continue
            seen_tables.add(id(table))

            try:
                # Read table without a header to preserve the exact layout
                df = pd.read_html(io.StringIO(str(table)), header=None)[0]
                df = df.fillna('')
                if len(df) < 3:
                    continue
                # 2. Identify the Structure
                label_col_idx = 0
                
                # Temporary storage for this table's columns
                table_columns = []
                
                # 3. Iterate through DATA columns (start from label_col_idx + 1)
                for col_idx in range(label_col_idx + 1, df.shape[1]):
                    col_data = df.iloc[:, col_idx]
                    
                    # --- A. Detect Year ---
                    year = None
                    year_row_idx = -1
                    
                    for i in range(min(5, len(col_data))):
                        val = str(col_data.iloc[i]).strip()
                        # Look for date pattern like MM/DD/YY and extract the year
                        match = re.search(r'(?:20\d{2})|\/(\d{2})$', val)
                        if match:
                            y = match.group(1) if match.group(1) else match.group(0)
                            year = "20" + y if len(y) == 2 else y
                            year_row_idx = i
                            break
                    
                    if not year:
                        continue

                    # --- B. Detect Fund Name ---
                    fund_name = None
                    
                    if year_row_idx >= 0:
                        fund_val = str(col_data.iloc[year_row_idx]).strip()
                        fund_match = re.search(r'(.+?)(?=\s*Year Ended)', fund_val, re.IGNORECASE)
                        if fund_match:
                            fund_name = fund_match.group(1).strip()
                    if year_row_idx > 0:
                        prev_row_val = str(col_data.iloc[year_row_idx - 1]).strip()
                        if prev_row_val and prev_row_val.lower() != 'nan':
                            fund_name = prev_row_val

                    if not fund_name or fund_name.lower() == 'nan' or not fund_name:
                        row_0_val = str(col_data.iloc[0]).strip()
                        if row_0_val and row_0_val.lower() != 'nan':
                            fund_name = row_0_val

                    if not fund_name or fund_name.lower() == 'nan' or not fund_name:
                        row_1_val = str(col_data.iloc[1]).strip()
                       
                        if row_1_val and row_1_val.lower() != 'nan':
                            fund_name = row_1_val
                    
                    if not fund_name or fund_name.lower() == 'nan' or not fund_name:
                        fund_name = "Unknown Fund"
                        

                    # --- C. Detect Share Class ---
                    share_class = "ETF Shares" if "ETF" in fund_name else "Unknown"
                    
                    if share_class is None or "Unknown" in share_class:
                        share_class = "Unknown"
                        
                    # --- D. Extract Metrics for this column ---
                    column_metrics = {
                        'year': year,
                        'nav_beginning': None,
                        'nav_end': None,
                        'total_return': None,
                        'expense_ratio': None,
                        'net_income_ratio': None,
                        'portfolio_turnover': None
                    }
                    
                    for row_idx in range(len(df)):
                        label = str(df.iloc[row_idx, label_col_idx]).strip().lower()
                        val = str(df.iloc[row_idx, col_idx]).strip()
                        
                        if not label or label == 'nan' or label == '':
                            continue
                        if not val or val == 'nan' or val == '':
                            continue
                        
                        # Skip footnote rows (they start with parentheses and contain long text)
                        if label.startswith('(') and len(label) > 50:
                            continue
                                  
                        # Value Extraction
                        if 'net asset value' in label and 'beginning' in label:
                            column_metrics['nav_beginning'] = val
                        elif 'net assets, end' in label:
                            clean_val = val.replace('$', '').replace(',', '').strip()
                            
                            try:
                                numeric_val = int(clean_val) * 1000
                                column_metrics['net_assets'] = str(numeric_val)  # Store as string or keep as float
                            except ValueError as e:
                                print("ValueError:", e)
                                column_metrics['net_assets'] = val  # Fallback to original if parsing fails
                        elif 'net asset value' in label and 'end' in label:
                            column_metrics['nav_end'] = val
                            
                        elif 'based on net asset value' in label:
                            column_metrics['total_return'] = val
                            
                        elif 'total return' in label and not column_metrics['total_return']:
                            column_metrics['total_return'] = val
                            
                        elif ('total expenses' in label or 'ratio of expenses' in label or 'ratio of total expenses' in label):
                            column_metrics['expense_ratio'] = val
                            
                        elif 'net investment income' in label:
                            column_metrics['net_income_ratio'] = val
                            
                        elif 'portfolio turnover' in label:
                            column_metrics['portfolio_turnover'] = val

                        elif 'distribution from net' in label:
                            column_metrics['distribution_shares'] = val
                    # Store this column's data
                    table_columns.append({
                        'fund_name': fund_name,
                        'share_class': share_class,
                        'metrics': column_metrics
                    })
                
                # --- E. Group columns by fund/share class ---
                for col_info in table_columns:
                    key = (col_info['fund_name'], col_info['share_class'])
                    
                    if key not in fund_data:
                        fund_data[key] = {
                            'fund_name': col_info['fund_name'],
                            'share_class': col_info['share_class'],
                            'years': [],
                            'net_assets': [],
                            'nav_beginning': [],
                            'nav_end': [],
                            'total_return': [],
                            'expense_ratio': [],
                            'net_income_ratio': [],
                            'portfolio_turnover': [],
                            'distribution_shares': []
                        }
                    
                    # Append this column's data to the lists
                    metrics = col_info['metrics']
                    fund_data[key]['years'].append(metrics['year'])
                    fund_data[key]['net_assets'].append(metrics.get('net_assets', None))
                    fund_data[key]['nav_beginning'].append(metrics['nav_beginning'])
                    fund_data[key]['nav_end'].append(metrics['nav_end'])
                    fund_data[key]['total_return'].append(metrics['total_return'])
                    fund_data[key]['expense_ratio'].append(metrics['expense_ratio'])
                    fund_data[key]['net_income_ratio'].append(metrics['net_income_ratio'])
                    fund_data[key]['portfolio_turnover'].append(metrics['portfolio_turnover'])
                    fund_data[key]['distribution_shares'].append(metrics.get('distribution_shares', None))

            except Exception as e:
                print(f"Error parsing table: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Convert to list format expected by _create_performance_dataframe
        results = list(fund_data.values())
        
        print(f"Total funds extracted: {len(results)}")
        for result in results:
            print(f"  {result['fund_name']} - {result['share_class']}: {len(result['years'])} years")
        
        return self._create_performance_dataframe(results)
    
     

    def _clean_numeric_value(self, value, column_name: str = None) -> float:
        """
        Clean numeric values from financial data.
        Handles: $, commas, %, parentheses (negative), em-dash, None, etc.
        
        Examples:
            '$31,195' -> 31195.0
            '(0.02%)' -> -0.02
            '0.40%3' -> 0.40
            '$—' -> 0.0
            None -> 0.0
        """
        if value is None or value == '' or pd.isna(value):
            return 0.0
        
        str_val = str(value).strip()
        
        # Handle em-dash and common non-numeric values
        if str_val in ['—', '-', 'N/A', 'n/a', 'NA', 'None', 'null']:
            return 0.0
        
        # Check for negative (parentheses)
        is_negative = str_val.startswith('(') and str_val.endswith(')')
        if is_negative:
            str_val = str_val[1:-1]  # Remove parentheses
        
        # Remove $, %, commas, spaces
        str_val = str_val.replace('$', '').replace('%', '').replace(',', '').replace(' ', '')
        
        # Remove any trailing non-numeric characters (like '3' in '0.40%3')
        import re
        match = re.match(r'^-?\d+\.?\d*', str_val)
        if match:
            str_val = match.group()
        
        try:
            result = float(str_val)
            result = -result if is_negative else result
            
            # Special handling for expense_ratio - format to 2 decimal places
            if column_name == 'expense_ratio' or column_name == 'net_income_ratio':
                result = round(result, 2)
            
            return result
        except (ValueError, TypeError):
            return 0.0
    
    def _create_performance_dataframe(self, performance_data: List[Dict]) -> pd.DataFrame:
        """
        Convert extracted performance data into a structured DataFrame.
        Helper method for get_financial_highlights.
        """
        rows = []
        
        for entry in performance_data:
            fund_name = entry.get('fund_name', 'Unknown')
            share_class = entry.get('share_class', 'Unknown')
            years = entry.get('years', [])
            net_assets = entry.get('net_assets', [])
            
            # Get metrics
            nav_beg = entry.get('nav_beginning', [])
            nav_end = entry.get('nav_end', [])
            total_ret = entry.get('total_return', [])
            exp_ratio = entry.get('expense_ratio', [])
            net_inc = entry.get('net_income_ratio', [])
            turnover = entry.get('portfolio_turnover', [])
            distribution_shares_list = entry.get('distribution_shares', [None])
            # Create a row for each year
            max_len = max(len(years), len(nav_beg), len(nav_end), len(total_ret))
            
            for i in range(max_len):
                row = {
                    'fund_name': fund_name,
                    'share_class': share_class,
                    'year': years[i] if i < len(years) else None,
                    'net_assets': net_assets[i] if i < len(net_assets) else None,
                    'nav_beginning': nav_beg[i] if i < len(nav_beg) else None,
                    'nav_end': nav_end[i] if i < len(nav_end) else None,
                    'total_return': total_ret[i] if i < len(total_ret) else None,
                    'expense_ratio': exp_ratio[i] if i < len(exp_ratio) else None,
                    'net_income_ratio': net_inc[i] if i < len(net_inc) else None,
                    'portfolio_turnover': turnover[i] if i < len(turnover) else None,
                    'distribution_shares': distribution_shares_list[i] if i < len(distribution_shares_list) else None
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Clean numeric columns
        numeric_columns = ['net_assets', 'nav_beginning', 'nav_end', 'total_return', 
                          'expense_ratio', 'net_income_ratio', 'portfolio_turnover']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._clean_numeric_value(x, col))
        
        return df

    def _get_embedded_value(self, block_name: str, target_name: str, c_id: str) -> str:
        """Finds a block, parses it, finds a tag inside."""
        # Find outer block - try both nonnumeric and nonfraction
        block = self.soup.find(["ix:nonnumeric", "ix:nonfraction"], attrs={"name": block_name, "contextref": c_id})
        
        if not block: 
            
            c_id = self.get_cid(c_id)
            
            tags = self.soup.find_all(attrs={"name": target_name})
            
            for tag in tags:
                if tag.has_attr('contextref') and c_id in tag['contextref']:
                    block = tag
                    break
        
        # Stitch if necessary (optimization: only stitch if continuedAt exists)
        html = XBRLUtils.stitch_block(block, self.soup)
        
        # Quick parse of just this snippet
        mini_soup = BeautifulSoup(html, 'lxml')
        target = mini_soup.find(attrs={"name": target_name})
        return XBRLUtils.clean_text(target)

    def _extract_tables(self, c_id: str, block_name: str = "oef:HoldingsTableTextBlock", table_name=None) -> Dict[str, pd.DataFrame]:
        """
        Complex logic to find and classify tables from a specified XBRL block.
        
        Args:
            c_id: Context ID to scope the search
            block_name: Name of the XBRL block to extract tables from (default: "oef:HoldingsTableTextBlock")
        
        Returns:
            Dictionary mapping table types to DataFrames
        """
        found_tables = {}
        data_rows = []
        # Try Strategy A: Extract from specified block
        block = self.soup.find("ix:nonnumeric", attrs={"name": block_name, "contextref": c_id})
        
        if not block and block_name != "oef:LargestHoldingsTableTextBlock":
            
            c_id2 = self.get_cid(c_id)
            tags = self.soup.find_all(attrs={"name": "oef:PctOfTotalInv"})
            target_tags = []
            
            for tag in tags:
                if tag.has_attr('contextref') and c_id2 in tag['contextref']:
                    target_tags.append(tag)
            for tag in target_tags:
                
                context_ref = tag.get('contextref', '')
                value_txt = tag.text.strip()
                try:
                    value_float = float(value_txt)
                except ValueError:
                    continue

                parts = context_ref.split('_')
        
                raw_asset_name = parts[-1] 

                clean_name = raw_asset_name.replace('Member', '').replace('CTI', '').replace('NVS', '')
                
                clean_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', clean_name)
               

               
                row_type = "Sector" if "Sector" in raw_asset_name else "Holding"
                

                data_rows.append({
                    "Category": row_type,
                    "Name": clean_name.strip(),
                    "Value": value_float
                })

            # 2. AFTER the loop: Create the DataFrame
            if data_rows:
                
                df = pd.DataFrame(data_rows)
                
                # Remove duplicates (XBRL sometimes has duplicate tags)
                df = df.drop_duplicates()
                
                # Sort by Value (Highest % first)
                df = df.sort_values(by='Value', ascending=False)

                df_sector = df[df['Category'] == 'Sector'][['Name', 'Value']].copy()
                df_holding = df[df['Category'] == 'Holding'][['Name', 'Value']].copy()

                found_tables["Sector Allocation"] = df_sector
                found_tables["Top 10 Holdings"] = df_holding
                
                return found_tables
            else:
                print("No data obtained")

        if block:
            full_html = XBRLUtils.stitch_block(block, self.soup)
            try:
                dfs = pd.read_html(io.StringIO(full_html))
                
                for df in dfs:
                    
                    t_type = XBRLUtils.classify_table(df)
                    
                    if t_type != "Unknown":
                        found_tables[t_type] = df.copy()
                    else:
                        print(f"Unknown table type: {df}")
            except Exception as e:
                print("Failed to extract tables from block: ", block_name)
                if "tables found" in str(e):
                    print("No tables found for block: ", block_name)
                else:
                    print("Tables not found.")

        return found_tables

    def fund_summary(self, fund_list: List[FundData]) -> str:
        

        print(f"Total Funds: {len(fund_list)}\n")
        print("=" * 80)

        # Define all possible field names and their types
        STRING_FIELDS = [
            "name", "context_id", "ticker", "share_class", "report_date",
            "security_exchange", "net_assets", "expense_ratio", "turnover_rate",
            "costs_per_10k", "advisory_fees", "n_holdings", "performance_commentary",
        ]

        TABLE_FIELDS = [
            "performance_table", "industry_allocation", "avg_annual_returns",
            "top_holdings", "sector_allocation", "portfolio_composition",
            "geographic_allocation", "maturity_allocation", "credit_rating", "issuer_allocation"
        ]

        for idx, fund in enumerate(fund_list, 1):
            # Header
            print(f"\n🏦 Fund {idx}/{len(fund_list)}: {fund.name}")
            print(f"🆔 {fund.context_id} | 🎫 {fund.ticker}")
        
            # --- NULL/MISSING FIELDS ---
            null_fields = []
            
            # Check string fields for None, "N/A", or empty strings
            for field_name in STRING_FIELDS:
                value = getattr(fund, field_name, None)
                if value is None or value == "N/A" or (isinstance(value, str) and value.strip() == ""):
                    null_fields.append(field_name)
            
            # Check table fields for None
            for field_name in TABLE_FIELDS:
                value = getattr(fund, field_name, None)
                if value is None:
                    null_fields.append(field_name)
            
            print(f"❌ Null Fields ({len(null_fields)}): {', '.join(null_fields) if null_fields else '✅ All populated!'}")
            
            # --- FOUND TABLES ---
            found_tables = []
            for field_name in TABLE_FIELDS:
                value = getattr(fund, field_name, None)
                if value is not None and isinstance(value, pd.DataFrame):
                    found_tables.append(f"{field_name}({value.shape[0]}×{value.shape[1]})")
            
            print(f"📊 Tables ({len(found_tables)}): {', '.join(found_tables) if found_tables else '⚠️ None'}")
            print("-" * 80)
        

    def print_fund_info(self, fund_list: List[FundData]) -> None:
        

        print(f"Showing information of {len(fund_list)} funds")
        # Assuming 'funds' is your list of FundData objects
        for fund in fund_list:
            # 1. HEADER: Use Markdown for a nice visual separation
            display(Markdown(f"### 🏦 {fund.name}"))
            
            # 2. BASIC INFO GRID
            print(f"🆔 Context ID:      {fund.context_id}")
            print(f"🎫 Ticker:          {fund.ticker}")
            print(f"🏷️ Share Class:     {fund.share_class}")
            print(f"📅 Report Date:     {fund.report_date}")
            print(f"🏛️ Sec Exchange:    {fund.security_exchange}")

            # 3. FINANCIALS
            print("\n--- 💰 Costs & Financials ---")
            # Using f-string alignment (<20) to make columns line up perfectly
            print(f"{'Net Assets':<20}: {fund.net_assets}")
            print(f"{'Expense Ratio':<20}: {fund.expense_ratio}")
            print(f"{'Turnover Rate':<20}: {fund.turnover_rate}")
            print(f"{'Costs per $10k':<20}: {fund.costs_per_10k}")
            print(f"{'Advisory Fees':<20}: {fund.advisory_fees}")
            print(f"{'Number of Holdings':<20}: {fund.n_holdings}")

            # 4. COMMENTARY (Truncated)
            # We strip newlines and limit it to 200 chars so it doesn't clutter the screen
            if fund.performance_commentary and fund.performance_commentary != "N/A":
                clean_commentary = fund.performance_commentary.replace('\n', ' ').strip()
                print(f"\n📝 Commentary: \"{clean_commentary[:250]}...\"")

            # 5. DATA TABLES
            # We check each dataframe to see if it exists before displaying
            if fund.performance_table is not None:
                display(Markdown("**📈 Performance History**"))
                display(fund.performance_table)

            if fund.industry_allocation is not None:
                display(Markdown("**📊 Industry Allocation**"))
                display(fund.industry_allocation)
            
            if fund.avg_annual_returns is not None:
                display(Markdown("**📊 Average Annual Returns**"))
                display(fund.avg_annual_returns)

            if fund.top_holdings is not None:
                display(Markdown("**🏆 Top Holdings**"))
                display(fund.top_holdings)

            if fund.sector_allocation is not None:
                display(Markdown("**🏗️ Sector Allocation**"))
                display(fund.sector_allocation)

            if fund.portfolio_composition is not None:
                display(Markdown("**🍰 Portfolio Composition**"))
                display(fund.portfolio_composition)

            if fund.geographic_allocation is not None:
                display(Markdown("**🌍 Geographic Allocation**"))
                display(fund.geographic_allocation)
            if fund.maturity_allocation is not None:
                display(Markdown("**📊 Maturity Allocation**"))
                display(fund.maturity_allocation)

            if fund.credit_rating is not None:
                display(Markdown("**📊 Credit Rating**"))
                display(fund.credit_rating)

            if fund.issuer_allocation is not None:
                display(Markdown("**📊 Issuer Allocation**"))
                display(fund.issuer_allocation)

            # 6. END OF FUND SEPARATOR
            print("\n" + "="*80 + "\n")
                