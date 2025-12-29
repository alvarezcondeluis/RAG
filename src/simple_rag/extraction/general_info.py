import re

class FundInfoExtractor:
    def __init__(self, raw_text, ticker="UNKNOWN"):
        self.raw_text = raw_text
        self.ticker = ticker
        # [VERIFIED] Clean text: remove multiple spaces/newlines for easier regex matching
        self.clean_text = re.sub(r'\s+', ' ', raw_text).strip()

    def _extract_between(self, start_pattern, end_pattern):
        """Helper to extract text block between two headers."""
        try:
            # We use \s* to handle potential spaces/newlines around the headers
            pattern = f"{start_pattern}\s*(.*?)\s*{end_pattern}"
            match = re.search(pattern, self.clean_text, re.IGNORECASE)
            return match.group(1).strip() if match else None
        except Exception:
            return None


    def extract_ticker(self):
        """
        Robust extraction of Ticker Symbols (2-5 uppercase letters).
        
        Upgrade Notes:
        1. PRIORITIZED iShares 'Pipe' style to avoid capturing 'Ticker: IVV' (reference asset) 
        often found in the footer of Buffer/Outcome ETFs.
        2. RELAXED character limit from {3,5} to {2,5} to capture tickers like 'XT', 'VV', 'VO'.
        """
        
        # 1. Expand Noise Words
        # Added common false positives found in financial headers/footers
        noise_words = {
            "SEC", "CFA", "USA", "BID", "LLC", "INC", "IRA", "NAV", "ETF", 
            "NYSE", "ARCA", "CBOE", "BZX", "NASDAQ", "USD", "EUR", "GBP", 
            "SUM", "PRO", "S&P", "MSCI", "FTSE", "RUSSELL", "ICE", "FAX", "TEL"
        }

        # 2. Pattern A: Pipe Delimiter (iShares/BlackRock Header Style) -- MOVED TO PRIORITY #1
        # Context: These documents put the real ticker in the header: "Fund Name | Ticker | Exchange"
        # Captures: "| SMAX |", "| XT |", "ETF | IVV |"
        # We look for a standalone token between pipes, or between a pipe and a newline.
        pipe_match = re.search(r"\|\s*([A-Z]{2,5})\s*\|", self.clean_text)
        if pipe_match:
            candidate = pipe_match.group(1).upper()
            if candidate not in noise_words:
                return candidate

        # 3. Pattern B: Vanguard Style Parentheses -- MOVED TO PRIORITY #2
        # Context: "Vanguard 500 Index Fund (VOO)" or "Admiral Shares (VFIAX)"
        # This is extremely high signal for Vanguard docs.
        share_match = re.search(r"(?:Shares|Fund|ETF)\s*\(([A-Z]{2,5})\)", self.clean_text, re.IGNORECASE)
        if share_match:
            candidate = share_match.group(1).upper()
            if candidate not in noise_words:
                return candidate

        # 4. Pattern C: Explicit Label
        # Context: "Ticker: IVV" or "Ticker Symbol: AAPL"
        # DOWNGRADED: In Buffer ETFs, this label sometimes points to the 'Underlying Reference Asset' (IVV)
        # instead of the fund itself. We accept this ONLY if the logic above failed.
        explicit_match = re.search(r"Ticker(?:\sSymbol)?[:\s]+([A-Z]{2,5})\b", self.clean_text, re.IGNORECASE)
        if explicit_match:
            candidate = explicit_match.group(1).upper()
            if candidate not in noise_words:
                return candidate

        # 5. Pattern D: Loose "ETF" Predecessor (Last Resort)
        # Context: "Global Tech ETF IXN"
        etf_match = re.search(r"\bETF\s+([A-Z]{2,5})\b", self.clean_text)
        if etf_match:
            candidate = etf_match.group(1).upper()
            if candidate not in noise_words:
                return candidate

        return "UNKNOWN"

    def get_objective(self):
        return self._extract_between("Investment Objective", "Fees and Expenses")

    def get_strategies(self):
        # [VERIFIED] Vanguard/iShares usually follow Strategies with "Principal Risks"
        return self._extract_between("Principal Investment Strategies", "Principal Risks")

    def get_risks(self):
        """Extracts risks while preserving original formatting."""
        pattern = r"Principal Risks\s*(.*?)\s*Annual Total Returns"
        match = re.search(pattern, self.raw_text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None
    def get_benchmark(self):
        """
        Extracts the exact name of the Index the fund tracks.
        
        Handles multiple formats:
        - Vanguard: "track the performance of the CRSP US Mid Cap Index"
        - iShares: "track the investment results of the MSCI EAFE Value Index"
        """
        strategies = self.get_strategies()
        if not strategies:
            return None
        
        # Pattern 1: Vanguard style - "track the performance of"
        match = re.search(r"track the performance of (?:the )?([A-Z][a-zA-Z\s\&\-\.]+Index)", strategies, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: iShares style - "track the investment results of"
        match = re.search(r"track the investment results of (?:the |an )?([A-Z][a-zA-Z\s\&\-\.]+Index)", strategies, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return "Unknown Benchmark"

    def get_managers(self):
        """
        Extracts list of manager names from Portfolio Managers section.
        
        Handles multiple formats:
        - Vanguard: "Kenny Narzikul, CFA" or "Walter Nejman, Portfolio Manager"
        - iShares: "Jennifer Hsui, Matt Waldron, Peter Sietsema and Steven White"
        """
        section = self._extract_between("Portfolio Managers", "(Purchase and Sale|Tax Information)")
        if not section:
            return []
        
        names = []
        
        # Pattern 1: Vanguard style - names followed by ", CFA" or ", Portfolio Manager"
        pattern_vanguard = r"([A-ZÀ-ÿ][a-zÀ-ÿ]+(?:\s[A-ZÀ-ÿ][a-zÀ-ÿ]+)+)(?:,\s+CFA|,\s+Portfolio Manager)"
        names.extend(re.findall(pattern_vanguard, section))
        
        # Pattern 2: iShares style - sentence format with "and" conjunction
        # Example: "Jennifer Hsui, Matt Waldron, Peter Sietsema and Steven White (the "Portfolio Managers")"
        # Look for pattern: Name, Name, Name and Name before "(the" or "are primarily"
        pattern_ishares = r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)(?:\s*,\s*|\s+and\s+)(?=[A-Z][a-z]+|\(the|are\s+primarily)"
        ishares_matches = re.findall(pattern_ishares, section)
        names.extend(ishares_matches)
        
        # Also capture the last name in an "and" sequence
        pattern_last = r"and\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)(?=\s*\(the|\s+are\s+primarily)"
        last_match = re.search(pattern_last, section)
        if last_match:
            names.append(last_match.group(1))
        
        # Filter out common false positives
        false_positives = {"Portfolio Manager", "Portfolio Managers", "Fund Manager", "Fund Managers"}
        names = [name for name in names if name not in false_positives]
        
        # Remove duplicates and return
        return list(set(names)) if names else []

    def get_expenses(self):
        """Extracts the specific Total Annual Fund Operating Expense."""
        pattern = r"Total Annual Fund Operating Expenses\s*([0-9]+\.[0-9]+)%"
        match = re.search(pattern, self.clean_text)
        return float(match.group(1)) if match else None

    def get_min_investment(self):
        """Extracts the dollar amount for minimum investment."""
        pattern = r"minimum investment.*?is\s*\$([0-9,]+)"
        match = re.search(pattern, self.clean_text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
        return None

    def get_investment_advisor(self):
        """
        Extracts the investment advisor name from the Management section.
        
        Handles formats like:
        - "Investment Adviser. BlackRock Fund Advisors."
        - "Investment Adviser: The Vanguard Group, Inc."
        """
        # Look for "Investment Adviser" or "Investment Advisor" followed by the name
        pattern = r"Investment Advis[eo]r[:\.]?\s+([A-Z][a-zA-Z\s,\.&]+?)(?:\.|$|\n)"
        match = re.search(pattern, self.clean_text, re.IGNORECASE)
        
        if match:
            advisor = match.group(1).strip()
            # Clean up common trailing characters
            advisor = re.sub(r'[,\.]$', '', advisor).strip()
            return advisor
        
        return None

    def extract_report_date(self):
        """
        Extracts the report date from Form 497K prospectus documents.
        
        Handles formats like:
        - "November 29, 2024"
        - "November 28, 2025"
        - "(as revised September 4, 2025)" - extracts the revision date
        
        Returns the most recent date found near Form 497K.
        """
        # Pattern 1: Standard date after Form 497K (Month Day, Year)
        # Looks for: "Form 497K" followed by date within reasonable proximity
        pattern_standard = r"Form 497K.*?([A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
        match = re.search(pattern_standard, self.clean_text, re.IGNORECASE)
        
        if match:
            primary_date = match.group(1)
            
            # Pattern 2: Check for revision date "(as revised Month Day, Year)"
            # This takes precedence as it's more recent
            revision_pattern = r"\(as revised\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\)"
            revision_match = re.search(revision_pattern, self.clean_text, re.IGNORECASE)
            
            if revision_match:
                return revision_match.group(1)
            
            return primary_date
        
        return None

    # --- OUTPUT GENERATORS ---

    def get_structured_data(self):
        """Returns the Dictionary for your Graph/SQL DB."""
        self.ticker = self.extract_ticker()
        return {
            "ticker": self.ticker,
            "report_date": self.extract_report_date(),
            "objective": self.get_objective(),
            "strategies": self.get_strategies(),
            "risks": self.get_risks(),
            "benchmark_index": self.get_benchmark(),
            "investment_advisor": self.get_investment_advisor(),
            "managers": self.get_managers(),
            "expense_ratio": self.get_expenses(),
            "min_investment": self.get_min_investment()
        }

    def get_clean_markdown(self):
        """Returns the string for your Vector Store (Embedding)."""
        data = self.get_structured_data()
        
        # [VERIFIED] We format this specifically for RAG.
        # Headers are clear, and noise (legal disclaimers) is stripped.
        markdown = f"""# FUND PROFILE({self.ticker}):

## Investment Objective
{data['objective'] or "Data not available"}

## Principal Investment Strategies
{data['strategies'] or "Data not available"}

## Principal Risks
{data['risks'] or "Data not available"}

## Report Date
{data['report_date'] or "Data not available"}
## Benchmark Index
{data['benchmark_index'] or "Data not available"}

## Fund Details
* **Investment Advisor**: {data['investment_advisor'] or "Unknown"}
* **Managers**: {', '.join(data['managers']) if data['managers'] else "Unknown"}
* **Minimum Investment**: ${data['min_investment']}
"""
        return markdown

