import re
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

class ProspectusExtractor:
    """
    Parser for SEC Form 497K (Summary Prospectus) documents.
    Optimized for Vanguard and iShares naming conventions and structures.
    """
    
    # Common false positives found in financial headers/footers
    TICKER_NOISE_WORDS: set[str] = {
        "SEC", "CFA", "USA", "BID", "LLC", "INC", "IRA", "NAV", "ETF", 
        "NYSE", "ARCA", "CBOE", "BZX", "NASDAQ", "USD", "EUR", "GBP", 
        "SUM", "PRO", "S&P", "MSCI", "FTSE", "RUSSELL", "ICE", "FAX", "TEL"
    }

    def __init__(self, raw_text: str, ticker: str = "UNKNOWN"):
        self.raw_text = raw_text
        self.ticker = ticker
        # Normalize text: remove multiple spaces and newlines to simplify regex matching
        self.normalized_content = re.sub(r'\s+', ' ', raw_text).strip()

    def _extract_between(self, start_pattern: str, end_pattern: str) -> Optional[str]:
        """Extracts text block located between two specified headers."""
        try:
            pattern = f"{start_pattern}\s*(.*?)\s*{end_pattern}"
            match = re.search(pattern, self.normalized_content, re.IGNORECASE)
            return match.group(1).strip() if match else None
        except Exception as e:
            logger.error(f"Error extracting between {start_pattern} and {end_pattern}: {e}")
            return None

    def get_ticker(self) -> str:
        """
        Extracts the fund's ticker symbol using prioritized heuristic patterns.
        """
        # Pattern 1: iShares 'Pipe' style (e.g., | TICKER |)
        pipe_match = re.search(r"\|\s*([A-Z]{2,5})\s*\|", self.normalized_content)
        if pipe_match:
            candidate = pipe_match.group(1).upper()
            if candidate not in self.TICKER_NOISE_WORDS:
                return candidate

        # Pattern 2: Standard parentheses style (e.g., Fund Name (TICKER))
        share_match = re.search(r"(?:Shares|Fund|ETF)\s*\(([A-Z]{2,5})\)", self.normalized_content, re.IGNORECASE)
        if share_match:
            candidate = share_match.group(1).upper()
            if candidate not in self.TICKER_NOISE_WORDS:
                return candidate

        # Pattern 3: Explicit 'Ticker' label
        explicit_match = re.search(r"Ticker(?:\sSymbol)?[:\s]+([A-Z]{2,5})\b", self.normalized_content, re.IGNORECASE)
        if explicit_match:
            candidate = explicit_match.group(1).upper()
            if candidate not in self.TICKER_NOISE_WORDS:
                return candidate

        # Pattern 4: Loose 'ETF' predecessor
        etf_match = re.search(r"\bETF\s+([A-Z]{2,5})\b", self.normalized_content)
        if etf_match:
            candidate = etf_match.group(1).upper()
            if candidate not in self.TICKER_NOISE_WORDS:
                return candidate
                
        return "UNKNOWN"

    def get_objective(self) -> Optional[str]:
        """Extracts the Investment Objective section."""
        return self._extract_between("Investment Objective", "Fees and Expenses")

    def get_strategies(self) -> Optional[str]:
        """Extracts the Principal Investment Strategies section."""
        return self._extract_between("Principal Investment Strategies", "Principal Risks")

    def get_risks(self) -> Optional[str]:
        """Extracts the Principal Risks section while preserving original formatting."""
        pattern = r"Principal Risks\s*(.*?)\s*Annual Total Returns"
        match = re.search(pattern, self.raw_text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None
    
    def get_benchmark(self) -> Optional[str]:
        """
        Extracts the name of the benchmark index the fund tracks.
        """
        strategies = self.get_strategies()
        if not strategies:
            return None
        
        # Vanguard/Common: "track the performance of..."
        match = re.search(r"track the performance of (?:the )?([A-Z][a-zA-Z\s\&\-\.]+Index)", strategies, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # iShares: "track the investment results of..."
        match = re.search(r"track the investment results of (?:the |an )?([A-Z][a-zA-Z\s\&\-\.]+Index)", strategies, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        return "Unknown Benchmark"

    def get_managers(self) -> List[str]:
        """
        Extracts portfolio manager names from the Management section.
        """
        section = self._extract_between("Portfolio Managers", "(Purchase and Sale|Tax Information)")
        if not section:
            return []
        
        names = []
        
        # Vanguard style: Name followed by designation
        pattern_vanguard = r"([A-ZÀ-ÿ][a-zÀ-ÿ]+(?:\s[A-ZÀ-ÿ][a-zÀ-ÿ]+)+)(?:,\s+CFA|,\s+Portfolio Manager)"
        names.extend(re.findall(pattern_vanguard, section))
        
        # iShares style: Sentence format with 'and' conjunction
        pattern_ishares = r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)(?:\s*,\s*|\s+and\s+)(?=[A-Z][a-z]+|\(the|are\s+primarily)"
        ishares_matches = re.findall(pattern_ishares, section)
        names.extend(ishares_matches)
        
        # Capture the final name in a sequence
        pattern_last = r"and\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)(?=\s*\(the|\s+are\s+primarily)"
        last_match = re.search(pattern_last, section)
        if last_match:
            names.append(last_match.group(1))
        
        # Clean results
        false_positives = {"Portfolio Manager", "Portfolio Managers", "Fund Manager", "Fund Managers"}
        names = [n.strip() for n in names if n not in false_positives]
        
        return list(set(names))

    def get_expense_ratio(self) -> Optional[float]:
        """Extracts the Total Annual Fund Operating Expense percentage."""
        pattern = r"Total Annual Fund Operating Expenses\s*([0-9]+\.[0-9]+)%"
        match = re.search(pattern, self.normalized_content)
        return float(match.group(1)) if match else None

    def get_min_investment(self) -> Optional[float]:
        """Extracts the minimum investment dollar amount."""
        pattern = r"minimum investment.*?is\s*\$([0-9,]+)"
        match = re.search(pattern, self.normalized_content, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
        return None

    def get_investment_advisor(self) -> Optional[str]:
        """Extracts the investment advisor entity."""
        pattern = r"Investment Advis[eo]r[:\.]?\s+([A-Z][a-zA-Z\s,\.&]+?)(?:\.|$|\n)"
        match = re.search(pattern, self.normalized_content, re.IGNORECASE)
        
        if match:
            advisor = match.group(1).strip()
            return re.sub(r'[,\.]$', '', advisor).strip()
        
        return None

    def get_report_date(self) -> Optional[str]:
        """
        Extracts the document date, prioritizing the most recent revision date.
        """
        # Primary document date
        pattern_standard = r"Form 497K.*?([A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
        match = re.search(pattern_standard, self.normalized_content, re.IGNORECASE)
        
        if match:
            primary_date = match.group(1)
            
            # Check for revision date which takes precedence
            rev_pattern = r"\(as revised\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\)"
            rev_match = re.search(rev_pattern, self.normalized_content, re.IGNORECASE)
            
            return rev_match.group(1) if rev_match else primary_date
        
        return None

    def get_structured_data(self) -> Dict[str, Any]:
        """Returns a structured dictionary of all extracted fund metrics."""
        self.ticker = self.get_ticker()
        return {
            "ticker": self.ticker,
            "report_date": self.get_report_date(),
            "objective": self.get_objective(),
            "strategies": self.get_strategies(),
            "risks": self.get_risks(),
            "benchmark_index": self.get_benchmark(),
            "investment_advisor": self.get_investment_advisor(),
            "managers": self.get_managers(),
            "expense_ratio": self.get_expense_ratio(),
            "min_investment": self.get_min_investment()
            
        }

    def get_clean_markdown(self) -> str:
        """Generates a clean markdown profile suitable for RAG context."""
        data = self.get_structured_data()
        
        return f"""# FUND PROFILE ({self.ticker}):

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
* **Minimum Investment**: ${data['min_investment'] if data['min_investment'] is not None else "Unknown"}
"""

