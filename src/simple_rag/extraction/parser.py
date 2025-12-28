import io
from typing import List, Dict
from bs4 import BeautifulSoup
from ..utils.utils import XBRLUtils
from ..models.fund import FundData
import pandas as pd
import re
from IPython.display import display, Markdown

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
        
    def _extract_single_fund(self, c_id: str, name: str) -> FundData:
        """Extracts all data for a specific Context ID."""
        
        # Initialize the data object
        
        fund = FundData(name=name, context_id=c_id, report_date=self.report_date, registrant=self.registrant)
        if "ETF" in name: fund.share_class = "ETF"
        # 1. Basic Tags (Scoped to this context)
        print("Extracting context: ", c_id)
        fund.ticker = self._get_value("dei:TradingSymbol", c_id)
        fund.expense_ratio = self._get_value("oef:ExpenseRatioPct", c_id)
        fund.costs_per_10k = self._get_value("oef:ExpensesPaidAmt", c_id)
        share_class = self._get_value("oef:ClassName", c_id)
        if share_class:
            fund.share_class = share_class
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
                if fund_name == "ESG U.S. Corporate Bond ETF":
                    print(df)
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
            
            # Get metrics
            nav_beg = entry.get('nav_beginning', [])
            nav_end = entry.get('nav_end', [])
            total_ret = entry.get('total_return', [])
            exp_ratio = entry.get('expense_ratio', [])
            net_inc = entry.get('net_income_ratio', [])
            turnover = entry.get('portfolio_turnover', [])
            
            # Create a row for each year
            max_len = max(len(years), len(nav_beg), len(nav_end), len(total_ret))
            
            for i in range(max_len):
                row = {
                    'fund_name': fund_name,
                    'share_class': share_class,
                    'year': years[i] if i < len(years) else None,
                    'nav_beginning': nav_beg[i] if i < len(nav_beg) else None,
                    'nav_end': nav_end[i] if i < len(nav_end) else None,
                    'total_return': total_ret[i] if i < len(total_ret) else None,
                    'expense_ratio': exp_ratio[i] if i < len(exp_ratio) else None,
                    'net_income_ratio': net_inc[i] if i < len(net_inc) else None,
                    'portfolio_turnover': turnover[i] if i < len(turnover) else None,
                }
                rows.append(row)
        
        return pd.DataFrame(rows)

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
                