import logging
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
        ex_compensation = ExecutiveCompensation(
            url=filing_url,
            form=obj.form,
            text=str(obj),
            ceo_name=obj.peo_name if hasattr(obj, "peo_name") else None,
            ceo_compensation=obj.peo_total_comp,
            ceo_actually_paid=obj.peo_actually_paid_comp,
            shareholder_return=obj.total_shareholder_return,
        )

        filing_date = filing_metadata.filing_date
        if filing_date not in company.filings_10k:
            company.filings_10k[filing_date] = Filing10K(
                filing_metadata=filing_metadata
            )

        company.executive_compensation = ex_compensation
