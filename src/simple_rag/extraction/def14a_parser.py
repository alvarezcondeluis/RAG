import logging
import re
from typing import Optional

from ..models.company import CompanyEntity, FilingMetadata, Filing10K, ExecutiveCompensation

logger = logging.getLogger(__name__)


class Def14AParser:
    """
    Parser for SEC Form DEF 14A (Proxy Statement) filings.

    DEF 14A filings contain the annual proxy statement sent to shareholders.
    This class focuses on the **Pay-versus-Performance** table introduced by
    the SEC in 2022, which discloses CEO total compensation, actually-paid
    compensation, and total shareholder return.

    Typical usage::

        parser = Def14AParser()
        parser.extract_executive_compensation(company, obj, filing_metadata, filing_url)
    """

    def extract_executive_compensation(
        self,
        company: CompanyEntity,
        obj,
        filing_metadata: FilingMetadata,
        filing_url: Optional[str] = None,
    ) -> None:
        """
        Extract executive compensation data and attach it to the company entity.

        The method reads the Pay-versus-Performance table from the proxy object
        returned by ``edgar`` and populates an :class:`ExecutiveCompensation`
        record on the company.  If no :class:`Filing10K` exists for the filing
        date yet, a minimal one is created so the compensation can be stored
        alongside the rest of the annual filing data.

        Args:
            company: :class:`CompanyEntity` to update in-place.
            obj: Proxy statement object from ``edgar``
                 (result of ``filing.obj()``).
            filing_metadata: Metadata describing the parent filing.
            filing_url: URL of the source DEF 14A filing (stored for reference).
        """
        # Step 1: Try to find the CEO from named_executives by role (most accurate)
        # This avoids confusing Executive Chairman / Founder with the actual CEO.
        ceo_name = None
        _CEO_ROLE_KEYWORDS = ("chief executive officer", "ceo")
        named_execs = getattr(obj, "named_executives", None) or []
        for exec_entry in named_execs:
            role = getattr(exec_entry, "role", None)
            if role and any(kw in str(role).lower() for kw in _CEO_ROLE_KEYWORDS):
                name = getattr(exec_entry, "name", None)
                if name and isinstance(name, str) and name.strip():
                    ceo_name = name.strip()
                    break

        # Step 2: Fall back to direct attributes on the proxy object
        if not ceo_name:
            for attr in ("peo_name", "ceo_name", "principal_executive_officer_name",
                         "named_executive_officer", "nec_name"):
                val = getattr(obj, attr, None)
                if val and isinstance(val, str) and val.strip():
                    ceo_name = val.strip()
                    break

        # Always extract fiscal_year_end — independently of whether CEO name was found.
        # Try named_executives first (most reliable when present), then pay_vs_performance.
        fiscal_year_end = None
        import pandas as _pd
        from datetime import datetime as _dt

        if named_execs:
            try:
                raw = getattr(named_execs[0], "fiscal_year_end", None)
                if raw:
                    fiscal_year_end = _dt.strptime(str(raw)[:10], "%Y-%m-%d").date()
            except Exception:
                pass

        if fiscal_year_end is None:
            pvp = getattr(obj, "pay_vs_performance", None)
            if pvp is not None:
                try:
                    pvp_df = pvp if isinstance(pvp, _pd.DataFrame) else (
                        pvp.to_dataframe() if hasattr(pvp, "to_dataframe") else _pd.DataFrame(pvp)
                    )
                    if "fiscal_year_end" in pvp_df.columns and not pvp_df.empty:
                        raw_date = pvp_df["fiscal_year_end"].dropna().iloc[-1]  # last = most recent
                        try:
                            fiscal_year_end = _dt.strptime(str(raw_date)[:10], "%Y-%m-%d").date()
                        except Exception:
                            pass
                except Exception:
                    pass

        # Fallback: inspect the pay_vs_performance DataFrame for CEO name
        if not ceo_name:
            pvp = getattr(obj, "pay_vs_performance", None)
            if pvp is not None:
                try:
                    pvp_df = pvp if isinstance(pvp, _pd.DataFrame) else (
                        pvp.to_dataframe() if hasattr(pvp, "to_dataframe") else _pd.DataFrame(pvp)
                    )
                    # Only look at columns whose values are strings (not numeric compensation)
                    name_cols = [
                        c for c in pvp_df.columns
                        if any(k in str(c).lower() for k in ("name",))
                        and pvp_df[c].dropna().apply(lambda v: isinstance(v, str)).any()
                    ]
                    if name_cols:
                        val = pvp_df[name_cols[0]].dropna()
                        if not val.empty:
                            ceo_name = str(val.iloc[0]).strip()
                except Exception:
                    pass

        # Fallback: fetch the raw filing HTML and search for the CEO name
        # str(obj) is just a one-line header for some filings, so we go to the source
        if not ceo_name and filing_url:
            try:
                import httpx
                resp = httpx.get(filing_url, timeout=15, follow_redirects=True)
                html_text = resp.text
                # "Sundar Pichai, Chief Executive Officer" or "CEO" followed by name
                for pattern in (
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z.]+)+),?\s+(?:Chief Executive Officer\b|CEO\b)',
                    r'(?:Chief Executive Officer|CEO)\s*[:\-–]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z.]+)+)',
                ):
                    m = re.search(pattern, html_text)
                    if m:
                        ceo_name = m.group(1)
                        break
            except Exception as e:
                logger.debug(f"Could not fetch filing HTML for CEO name: {e}")

        if not ceo_name:
            logger.warning(
                f"Could not extract CEO name from DEF 14A. "
                f"Available obj attrs: {[a for a in dir(obj) if not a.startswith('_')]}"
            )

        ex_compensation = ExecutiveCompensation(
            url=filing_url,
            form=obj.form,
            text=str(obj),
            ceo_name=ceo_name,
            ceo_compensation=float(obj.peo_total_comp) if obj.peo_total_comp is not None else None,
            ceo_actually_paid=float(obj.peo_actually_paid_comp) if obj.peo_actually_paid_comp is not None else None,
            shareholder_return=float(obj.total_shareholder_return) if obj.total_shareholder_return is not None else None,
            fiscal_year_end=fiscal_year_end,
        )

        filing_date = filing_metadata.filing_date
        if filing_date not in company.filings_10k:
            company.filings_10k[filing_date] = Filing10K(
                filing_metadata=filing_metadata
            )

        company.executive_compensation = ex_compensation
