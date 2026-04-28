import re
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

class ProspectusExtractor:
    """
    Parser for SEC Form 497K (Summary Prospectus) documents.

    Use :meth:`from_text` to auto-detect the provider and get the most
    appropriate subclass (``VanguardProspectusExtractor``,
    ``iSharesProspectusExtractor``, or ``FidelityProspectusExtractor``).
    Falls back to the base class for unknown providers.
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
        full_normalized = re.sub(r'\s+', ' ', raw_text).strip()
        # Scope to the first fund section only to avoid multi-fund bleed.
        # A new fund section starts with a second '497K' document header.
        section_split = re.split(r'497K\s+\d+\s+\S+\.htm', full_normalized)
        self.normalized_content = section_split[0].strip() if len(section_split) > 1 else full_normalized
        # Similarly scope raw_text for DOTALL extractions
        raw_split = re.split(r'497K\s+\d+\s+\S+\.htm', raw_text)
        self._scoped_raw = raw_split[0].strip() if len(raw_split) > 1 else raw_text

    def _extract_between(self, start_pattern: str, end_pattern: str) -> Optional[str]:
        """Extracts text block located between two specified headers (normalized, single-line)."""
        try:
            pattern = f"{start_pattern}\s*(.*?)\s*{end_pattern}"
            match = re.search(pattern, self.normalized_content, re.IGNORECASE)
            return match.group(1).strip() if match else None
        except Exception as e:
            logger.error(f"Error extracting between {start_pattern} and {end_pattern}: {e}")
            return None

    def _extract_between_raw(self, start_pattern: str, end_pattern: str) -> Optional[str]:
        """Extracts text block using _scoped_raw, preserving newlines (for paragraph-based parsing)."""
        try:
            pattern = f"{start_pattern}\s*(.*?)\s*{end_pattern}"
            match = re.search(pattern, self._scoped_raw, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else None
        except Exception as e:
            logger.error(f"Error extracting between {start_pattern} and {end_pattern}: {e}")
            return None

    def get_ticker(self) -> str:
        """
        Extracts the fund's ticker symbol using prioritized heuristic patterns.
        """
        # Pattern 0: Fidelity 'Fund/Ticker' slash style (e.g., Fund Name ETF/FEMV)
        slash_match = re.search(r"(?:Fund|ETF)/([A-Z]{2,6})\b", self.normalized_content)
        if slash_match:
            candidate = slash_match.group(1).upper()
            if candidate not in self.TICKER_NOISE_WORDS:
                return candidate

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

    def get_fund_name(self) -> Optional[str]:
        """Extracts the fund name from the document header."""
        fund_name_re = r"([A-Z][A-Za-z®™\s\-&,\.]+(?:Fund|ETF|Trust|Portfolio))"

        # Structure A: fund name is on its own line BEFORE 'Summary Prospectus'
        # e.g. "Vanguard Real Estate Index Fund\nSummary Prospectus\nMay 29, 2025"
        match = re.search(
            rf"{fund_name_re}\s*\n\s*Summary Prospectus",
            self._scoped_raw, re.IGNORECASE
        )
        if match:
            name = match.group(1).strip()
            if len(name) > 4:
                return name

        # Structure B: fund name appears AFTER 'Summary Prospectus' (date-first layout)
        # e.g. "April 10, 2026\nSummary Prospectus\nVanguard Developed Markets ex-US Growth\nIndex ETF"
        # Fund name may span multiple lines, so collect lines until we hit Fund/ETF/Trust/Portfolio
        sp_match = re.search(r"Summary Prospectus\s*\n", self._scoped_raw, re.IGNORECASE)
        if sp_match:
            after = self._scoped_raw[sp_match.end():]
            # Grab up to 5 non-empty lines after Summary Prospectus
            lines = [l.strip() for l in after.splitlines() if l.strip()][:5]
            # Join lines until we hit one ending with Fund/ETF/Trust/Portfolio
            name_parts = []
            for line in lines:
                # Skip date-like lines
                if re.match(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$", line):
                    continue
                name_parts.append(line)
                if re.search(r"(?:Fund|ETF|Trust|Portfolio)$", line):
                    break
            if name_parts:
                name = " ".join(name_parts).strip()
                if re.search(r"(?:Fund|ETF|Trust|Portfolio)", name) and len(name) > 4:
                    return name

        # Fallback: name anywhere adjacent to 'Summary Prospectus' in normalized text
        match = re.search(
            rf"{fund_name_re}\s+Summary Prospectus|Summary Prospectus\s+{fund_name_re}",
            self.normalized_content, re.IGNORECASE
        )
        if match:
            name = (match.group(1) or match.group(2) or "").strip()
            if name and len(name) > 4:
                return name
        return None

    def get_objective(self) -> Optional[str]:
        """Extracts the Investment Objective section."""
        return self._extract_between("Investment Objective", "Fees and Expenses")

    def get_strategies(self) -> Optional[str]:
        """Extracts the Principal Investment Strategies section."""
        return self._extract_between("Principal Investment Strategies", "Principal Risks")

    def get_risks(self) -> Optional[str]:
        """Extracts the Principal Risks section while preserving original formatting."""
        # End anchors are section headers that follow risks — use line-start-like anchors
        # to avoid matching the same words that appear inside risk descriptions.
        end_anchors = r"(?:Annual Total Returns|\nPerformance\n|\nPerformance History|\nInvestment Advis|\nInvestment Adviser)"
        pattern = rf"Principal (?:Investment )?Risks\s*(.*?)\s*{end_anchors}"
        match = re.search(pattern, self._scoped_raw, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: normalized content without newline anchors
        end_anchors_norm = r"(?:Annual Total Returns|Performance History|Investment Advis(?:er|or)\b)"
        pattern_norm = rf"Principal (?:Investment )?Risks\s*(.*?)\s*{end_anchors_norm}"
        match = re.search(pattern_norm, self.normalized_content, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def get_benchmark(self) -> Optional[str]:
        """
        Extracts the name of the benchmark index the fund tracks.
        """
        strategies = self.get_strategies()
        if not strategies:
            return None

        # Broad index pattern: greedy but stops before parenthesis, comma, period-space, or end.
        # Allows digits, slash, %, spaces, &, hyphen, dot (handles "MSCI US 25/50 Index", "S& P U. S. Index")
        index_pattern = r"([A-Z®][a-zA-Z0-9®\s\&\-\./%]+?Index)(?=[\s,\.\n\(]|$)"

        candidates = []

        # Vanguard/Common: "track the performance of..." (handle PDF artifact 'theperformance')
        for m in re.finditer(rf"track\s+the\s*performance\s+of\s+(?:the\s+)?{index_pattern}", strategies, re.IGNORECASE):
            candidates.append(m.group(1).strip())

        # iShares: "track the investment results of..."
        for m in re.finditer(rf"track\s+the\s*investment\s+results\s+of\s+(?:the\s+|an\s+)?{index_pattern}", strategies, re.IGNORECASE):
            candidates.append(m.group(1).strip())

        # Fidelity: "investing at least N% of assets in ... the XYZ Index"
        for m in re.finditer(rf"investing at least \d+%[^.]*?\bthe\s+{index_pattern}", strategies, re.IGNORECASE):
            candidates.append(m.group(1).strip())

        # Return the shortest match (most specific), strip trailing whitespace
        if candidates:
            return min(candidates, key=len).strip()

        return "Unknown Benchmark"

    def get_managers(self) -> List[str]:
        """
        Extracts portfolio manager names from the Management section.
        Uses raw text (preserving newlines) and splits on blank-line paragraph boundaries.
        """
        # Prefer raw extraction to preserve newlines between manager paragraphs
        section = self._extract_between_raw(
            r"Portfolio Manager(?:s|\(s\))?",
            r"(?:Purchase and Sale|Tax Information)"
        )
        if not section:
            # Fallback to normalized if raw extraction fails
            section = self._extract_between(r"Portfolio Manager(?:s|\(s\))?", "(Purchase and Sale|Tax Information)")
        if not section:
            return []
        return self._parse_manager_names(section)

    @staticmethod
    def _parse_manager_names(section: str) -> List[str]:
        """
        Shared helper: splits manager section on blank lines (paragraph boundaries),
        then extracts the name as everything before the first role-keyword comma.
        Falls back to sentence-splitting if no blank lines found.
        """
        # Fix PDF artifact: "O' Reilly" or "O'\u2019 Reilly" -> "O'Reilly"
        # Normalize Unicode right single quote to ASCII apostrophe first
        section = section.replace("\u2019", "'").replace("\u2018", "'")
        # Then collapse any whitespace between letter+apostrophe and following capital
        section = re.sub(r"([A-Za-z]')\s+([A-Z][a-z])", r"\1\2", section)

        role_keywords = re.compile(
            r",\s+(?:CFA|Portfolio Manager|Co-Portfolio Manager|Senior Managing Director"
            r"|Managing Director|Principal|Equity Portfolio Manager)",
            re.IGNORECASE
        )

        # Split on blank lines (paragraph boundaries) — works on raw text
        paragraphs = re.split(r"\n\s*\n", section)

        # If only one paragraph (normalized text), fall back to sentence splitting
        if len(paragraphs) <= 1:
            paragraphs = re.split(r"(?<=[a-z0-9\)])\.\s+(?=[A-Z])", section)

        names = []
        for para in paragraphs:
            # Collapse internal newlines and extra whitespace within each paragraph
            chunk = re.sub(r"\s+", " ", para).strip()
            # Strip leading page numbers e.g. "5 Gerard C. O'Reilly..."
            chunk = re.sub(r"^\d+\s+", "", chunk)
            if not chunk:
                continue
            role_match = role_keywords.search(chunk)
            if role_match:
                name = chunk[:role_match.start()].strip()
                words = name.split()
                # Sanity: 2–5 words, first word starts with a capital
                if 2 <= len(words) <= 5 and words[0][0].isupper():
                    names.append(name)

        false_positives = {"Portfolio Manager", "Portfolio Managers", "Fund Manager", "Fund Managers"}
        names = [n for n in names if n not in false_positives]
        return list(dict.fromkeys(names))

    def get_expense_ratio(self) -> Optional[float]:
        """Extracts the Total Annual Fund Operating Expense percentage."""
        pattern = r"Total Annual Fund Operating Expenses\s*([0-9]+\.[0-9]+)%"
        match = re.search(pattern, self.normalized_content)
        return float(match.group(1)) if match else None

    def get_min_investment(self) -> Optional[float]:
        """Extracts the minimum investment dollar amount (open/maintain account).
        Returns None if the fund is an ETF with no minimum or if not found.
        """
        # ETFs explicitly state no minimum — detect and return None
        no_min = re.search(
            r"no minimum dollar amount you must invest",
            self.normalized_content, re.IGNORECASE
        )
        if no_min:
            return None
        # Match dollar amounts with optional word-form multiplier (million, billion)
        pattern = r"minimum investment.*?open and maintain.*?is\s*\$([0-9,]+)\s*(million|billion)?"
        match = re.search(pattern, self.normalized_content, re.IGNORECASE)
        if match:
            amount = float(match.group(1).replace(',', ''))
            multiplier = (match.group(2) or "").lower()
            if multiplier == "million":
                amount *= 1_000_000
            elif multiplier == "billion":
                amount *= 1_000_000_000
            return amount
        # Fallback: any minimum investment dollar amount
        pattern_fallback = r"minimum investment.*?is\s*\$([0-9,]+)\s*(million|billion)?"
        match = re.search(pattern_fallback, self.normalized_content, re.IGNORECASE)
        if match:
            amount = float(match.group(1).replace(',', ''))
            multiplier = (match.group(2) or "").lower()
            if multiplier == "million":
                amount *= 1_000_000
            elif multiplier == "billion":
                amount *= 1_000_000_000
            return amount
        return None

    def _format_managers_section(self, section: Optional[str]) -> str:
        """Formats the raw manager section into one paragraph per manager."""
        if not section:
            return "Data not available"
        # Already split into paragraphs by get_managers_section; join with double newline
        paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return "\n\n".join(paragraphs)
        # Single block: split on sentence boundary after year (e.g. "2023. Aurélie")
        chunks = re.split(r"(?<=\d{4})\.\s+", paragraphs[0])
        cleaned = [c.strip().rstrip(".") + "." for c in chunks if c.strip()]
        return "\n\n".join(cleaned)

    def get_managers_section(self) -> Optional[str]:
        """Returns the full raw manager section text (names + roles + tenure)."""
        section = self._extract_between_raw(
            r"Portfolio Manager(?:s|\(s\))?",
            r"(?:Purchase and Sale|Tax Information)"
        )
        if not section:
            section = self._extract_between(
                r"Portfolio Manager(?:s|\(s\))?",
                "(Purchase and Sale|Tax Information)"
            )
        if not section:
            return None
        # Collapse internal whitespace for readability, preserve paragraph breaks
        paragraphs = re.split(r"\n\s*\n", section)
        cleaned = []
        for para in paragraphs:
            para = re.sub(r"\s+", " ", para).strip()
            para = re.sub(r"^\d+\s+", "", para)  # strip page numbers
            if para:
                cleaned.append(para)
        return "\n\n".join(cleaned) if cleaned else None

    def get_investment_advisor(self) -> Optional[str]:
        """Extracts the investment advisor entity."""
        # Match the value on the same line as the section header only.
        # Uses raw text to leverage newlines as a hard boundary.
        match = re.search(
            r"Investment Advis[eo]r\s*\n\s*\n?([^\n]+)",
            self._scoped_raw, re.IGNORECASE
        )
        if match:
            advisor = match.group(1).strip()
            # Stop at first sentence boundary to avoid subsidiary bleed
            # e.g. "The Vanguard Group, Inc. (Vanguard) through its wholly owned subsidiary..."
            advisor = re.split(r"\.\s+[A-Z]", advisor)[0]
            # Strip trailing parenthetical abbreviation e.g. "(Vanguard)" or "(FMR)"
            advisor = re.sub(r'\s*\([^)]+\)\s*$', '', advisor).strip()
            return re.sub(r'[,\.]$', '', advisor).strip()

        # Fallback: normalized single-line match, stop at sentence end or parenthesis
        match = re.search(
            r"Investment Advis[eo]r[:\.]?\s+([A-Z][a-zA-Z\s,\.&]+?)(?:\s+\(|\.|$)",
            self.normalized_content, re.IGNORECASE
        )
        if match:
            advisor = match.group(1).strip()
            return re.sub(r'[,\.]$', '', advisor).strip()

        return None

    def get_report_date(self) -> Optional[str]:
        """
        Extracts the document date, prioritizing the most recent revision date.
        """
        date_re = r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})"

        # Check for revision date first — always takes precedence
        rev_match = re.search(rf"\(as revised\s+{date_re}\)", self.normalized_content, re.IGNORECASE)
        if rev_match:
            return rev_match.group(1)

        # Primary: date immediately after "Form 497K"
        match = re.search(rf"Form 497K.*?{date_re}", self.normalized_content, re.IGNORECASE)
        if match:
            return match.group(1)

        # Structure A: date AFTER "Summary Prospectus"
        # e.g. "Summary Prospectus\nMay 29, 2025"
        match = re.search(rf"Summary Prospectus\s+{date_re}", self.normalized_content, re.IGNORECASE)
        if match:
            return match.group(1)

        # Structure B: date BEFORE "Summary Prospectus" (date-first layout)
        # e.g. "April 10, 2026\nSummary Prospectus"
        match = re.search(rf"{date_re}\s+Summary Prospectus", self.normalized_content, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def get_structured_data(self) -> Dict[str, Any]:
        """Returns a structured dictionary of all extracted fund metrics."""
        self.ticker = self.get_ticker()
        return {
            "ticker": self.ticker,
            "fund_name": self.get_fund_name(),
            "report_date": self.get_report_date(),
            "objective": self.get_objective(),
            "strategies": self.get_strategies(),
            "risks": self.get_risks(),
            "benchmark_index": self.get_benchmark(),
            "investment_advisor": self.get_investment_advisor(),
            "managers": self.get_managers(),
            "managers_section": self.get_managers_section(),
            "expense_ratio": self.get_expense_ratio(),
            "min_investment": self.get_min_investment(),
        }

    def get_clean_markdown(self) -> str:
        """Generates a clean markdown profile suitable for RAG context."""
        data = self.get_structured_data()
        
        return f"""# FUND PROFILE ({self.ticker}):

## Fund Name
{data['fund_name'] or "Data not available"}

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
* **Minimum Investment**: {("$" + str(data['min_investment'])) if data['min_investment'] is not None else "None (ETF — no minimum)"}

## Portfolio Managers
{chr(10).join(f"* {name}" for name in data['managers']) if data['managers'] else "Unknown"}

### Manager Details
{self._format_managers_section(data['managers_section'])}
"""

    @classmethod
    def from_text(cls, raw_text: str, ticker: str = "UNKNOWN") -> "ProspectusExtractor":
        """Factory: auto-detect provider and return the right extractor subclass.

        Args:
            raw_text: Raw 497K filing text.
            ticker: Optional seed ticker (passed through to the extractor).

        Returns:
            The most specific :class:`ProspectusExtractor` subclass for the
            detected provider, or the base class if unknown.
        """
        sample = raw_text[:3000].lower()
        if "vanguard" in sample:
            return VanguardProspectusExtractor(raw_text, ticker=ticker)
        if "ishares" in sample or "blackrock" in sample:
            return iSharesProspectusExtractor(raw_text, ticker=ticker)
        if "fidelity" in sample:
            return FidelityProspectusExtractor(raw_text, ticker=ticker)
        return cls(raw_text, ticker=ticker)


# ---------------------------------------------------------------------------
# Provider-specific subclasses
# ---------------------------------------------------------------------------

class VanguardProspectusExtractor(ProspectusExtractor):
    """497K extractor tuned for Vanguard filings."""

    def get_objective(self) -> Optional[str]:
        return self._extract_between("Investment Objective", "Fees and Expenses")

    def get_strategies(self) -> Optional[str]:
        return self._extract_between("Principal Investment Strategies", "Principal Risks")

    def get_risks(self) -> Optional[str]:
        end_anchors = r"(?:Annual Total Returns|\nPerformance\n|\nInvestment Advis)"
        pattern = rf"Principal Risks\s*(.*?)\s*{end_anchors}"
        match = re.search(pattern, self._scoped_raw, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        end_norm = r"(?:Annual Total Returns|Investment Advis(?:er|or)\b)"
        match = re.search(rf"Principal Risks\s*(.*?)\s*{end_norm}", self.normalized_content, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def get_managers(self) -> List[str]:
        section = self._extract_between_raw(
            r"Portfolio Manager(?:s|\(s\))?",
            r"(?:Purchase and Sale|Tax Information)"
        )
        if not section:
            section = self._extract_between(r"Portfolio Manager(?:s|\(s\))?", "(Purchase and Sale|Tax Information)")
        if not section:
            return []
        return self._parse_manager_names(section)


class iSharesProspectusExtractor(ProspectusExtractor):
    """497K extractor tuned for iShares / BlackRock filings."""

    def get_fund_name(self) -> Optional[str]:
        """iShares fund name appears as 'iShares ... ETF' on the summary prospectus header line."""
        # Pattern: "● iShares ... ETF* | TICKER | Exchange" or "iSHARES® ... ETF"
        # Try the bullet-line format first (Summary Prospectus header)
        match = re.search(
            r"[●•]?\s*(iShares[^|\n]+?(?:ETF|Fund|Trust))[*\s]*\|",
            self.normalized_content, re.IGNORECASE
        )
        if match:
            return match.group(1).strip().rstrip("*").strip()
        # Fallback: ALL-CAPS section header e.g. "iSHARES® iBONDS® DEC 2035 TERM MUNI BOND ETF"
        match = re.search(
            r"(iSHARES[®\s\w]+(?:ETF|FUND|TRUST))",
            self.normalized_content
        )
        if match:
            return match.group(1).strip()
        return super().get_fund_name()

    def get_report_date(self) -> Optional[str]:
        """iShares date appears at the very top of the document, before Summary Prospectus."""
        date_re = r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
        # Date at the very top (first 500 chars of normalized content)
        match = re.search(date_re, self.normalized_content[:500])
        if match:
            return match.group(1)
        return super().get_report_date()

    def get_objective(self) -> Optional[str]:
        return self._extract_between("Investment Objective", "Fees and Expenses")

    def get_strategies(self) -> Optional[str]:
        # Use raw extraction so the end anchor works across line breaks
        # iShares uses "Summary of Principal Risks" (may span two lines in raw)
        result = self._extract_between_raw(
            r"Principal Investment Strategies",
            r"Summary of Principal Risks"
        )
        if not result:
            result = self._extract_between_raw(
                r"Principal Investment Strategies",
                r"Principal Risks"
            )
        if result:
            # Normalize whitespace and strip inline page markers e.g. "S-1", "S-2"
            result = re.sub(r"\s+", " ", result).strip()
            result = re.sub(r"\bS-\d+\b", "", result).strip()
        return result

    def get_risks(self) -> Optional[str]:
        # iShares uses "Summary of Principal Risks" or "Principal Risks"
        end_anchors = r"(?:Performance Information|\nManagement\n|\nInvestment Advis)"
        for header in [r"Summary of Principal Risks", r"Principal Risks"]:
            pattern = rf"{header}\s*(.*?)\s*{end_anchors}"
            match = re.search(pattern, self._scoped_raw, re.IGNORECASE | re.DOTALL)
            if match:
                result = match.group(1).strip()
                return re.sub(r"\bS-\d+\b", "", result).strip()
        end_norm = r"(?:Performance Information|Management|Investment Advis(?:er|or)\b)"
        for header in ["Summary of Principal Risks", "Principal Risks"]:
            match = re.search(rf"{header}\s*(.*?)\s*{end_norm}", self.normalized_content, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                return re.sub(r"\bS-\d+\b", "", result).strip()
        return None

    def get_benchmark(self) -> Optional[str]:
        """iShares names the index in strategies; avoid returning 'Underlying Index' alias."""
        strategies = self.get_strategies()
        if not strategies:
            return None
        index_pattern = r"([A-Z®][a-zA-Z0-9®\s\&\-\./%TM]+?Index(?:TM)?)"
        # iShares: "track the investment results of the S&P ... Index"
        for m in re.finditer(
            rf"track\s+the\s*investment\s+results\s+of\s+(?:the\s+|an\s+)?{index_pattern}",
            strategies, re.IGNORECASE
        ):
            candidate = m.group(1).strip().rstrip(",")
            # Skip the generic alias
            if re.search(r"^Underlying Index", candidate, re.IGNORECASE):
                continue
            return candidate
        return super().get_benchmark()

    def get_managers(self) -> List[str]:
        """iShares lists managers under 'Management' section as 'Portfolio Managers. Name1, Name2 and Name3'"""
        # Use the scoped raw to find the Management *section header* (standalone line, not 'Management Fees')
        section = self._extract_between_raw(
            r"(?m)^Management$",
            r"(?:Purchase and Sale|Tax Information)"
        )
        if not section:
            # Fallback: find 'Portfolio Managers.' label directly in normalized content
            section = self._extract_between(
                r"Portfolio Managers?\.",
                "(Purchase and Sale|Tax Information)"
            )
        if not section:
            return []
        norm = re.sub(r"\s+", " ", section)
        # iShares format: "Portfolio Managers. FirstName LastName, FirstName LastName and FirstName LastName"
        match = re.search(
            r"Portfolio Managers?\.\s+([^.]+?)\s*(?:are primarily responsible|\(the|$)",
            norm, re.IGNORECASE
        )
        if match:
            names_text = match.group(1).strip()
            # Split on ', ' and ' and ' — handle joined words like 'MarcusTom' with CamelCase split
            raw_names = re.split(r",\s*|\s+and\s+", names_text)
            names = []
            for n in raw_names:
                n = n.strip().rstrip(",")
                # Split CamelCase joins e.g. 'MarcusTom' -> 'Marcus Tom'
                n = re.sub(r"([a-z])([A-Z])", r"\1 \2", n)
                words = n.split()
                if 2 <= len(words) <= 5 and words[0][0].isupper():
                    names.append(n)
            false_positives = {"Portfolio Manager", "Portfolio Managers"}
            return [n for n in dict.fromkeys(names) if n not in false_positives]
        return []

    def get_managers_section(self) -> Optional[str]:
        """iShares manager section is a single paragraph under Management."""
        section = self._extract_between_raw(
            r"(?m)^Management$",
            r"(?:Purchase and Sale|Tax Information)"
        )
        if not section:
            return None
        return re.sub(r"\s+", " ", section).strip()


class FidelityProspectusExtractor(ProspectusExtractor):
    """497K extractor tuned for Fidelity filings."""

    def get_objective(self) -> Optional[str]:
        result = self._extract_between("Investment Objective", "Fee Table")
        if not result:
            result = self._extract_between("Investment Objective", "Fees and Expenses")
        return result

    def get_strategies(self) -> Optional[str]:
        return self._extract_between("Principal Investment Strategies", "Principal Investment Risks")

    def get_risks(self) -> Optional[str]:
        end_anchors = r"(?:\nPerformance\n|\nPerformance History|\nInvestment Advis)"
        pattern = rf"Principal Investment Risks\s*(.*?)\s*{end_anchors}"
        match = re.search(pattern, self._scoped_raw, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        end_norm = r"(?:Performance(?:\s+History)?|Investment Advis(?:er|or)\b)"
        match = re.search(
            rf"Principal Investment Risks\s*(.*?)\s*{end_norm}",
            self.normalized_content, re.IGNORECASE
        )
        return match.group(1).strip() if match else None

    def get_managers(self) -> List[str]:
        section = self._extract_between(r"Portfolio Manager\(s\)", "(Purchase and Sale|Tax Information)")
        if not section:
            section = self._extract_between(r"Portfolio Managers?", "(Purchase and Sale|Tax Information)")
        if not section:
            return []
        # Fidelity lists managers as "Name (Co-Portfolio Manager)" — extract name before parenthetical
        names = []
        # Try parenthetical style first: "Name (Co-Portfolio Manager since ...)"
        for m in re.finditer(r"([A-ZÀ-ÿ][^\(]{2,40}?)\s+\(Co-Portfolio Manager", section):
            name = m.group(1).strip()
            words = name.split()
            if 2 <= len(words) <= 5 and words[0][0].isupper():
                names.append(name)
        if names:
            return list(dict.fromkeys(names))
        # Fallback: same sentence-split approach as Vanguard
        role_keywords = re.compile(
            r",\s+(?:CFA|Portfolio Manager|Co-Portfolio Manager|Senior Managing Director|Managing Director|Principal)",
            re.IGNORECASE
        )
        chunks = re.split(r"\.\s+", section)
        for chunk in chunks:
            chunk = re.sub(r"^\d+\s+", "", chunk.strip())
            role_match = role_keywords.search(chunk)
            if role_match:
                name = chunk[:role_match.start()].strip()
                words = name.split()
                if 2 <= len(words) <= 5 and words[0][0].isupper():
                    names.append(name)
        return list(dict.fromkeys(names))

    def get_investment_advisor(self) -> Optional[str]:
        pattern = r"Investment Advis(?:er|or)[:\.]?\s+([A-Z][a-zA-Z\s,\.&]+?)\s+\("
        match = re.search(pattern, self.normalized_content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return super().get_investment_advisor()
