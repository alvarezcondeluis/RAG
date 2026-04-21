import logging
from typing import List

from ..models.company import InsiderTransaction

logger = logging.getLogger(__name__)


class Form4Parser:
    """
    Parser for SEC Form 4 (Statement of Changes in Beneficial Ownership) filings.

    Form 4 documents are filed by insiders (officers, directors, and 10%+ shareholders)
    whenever they buy or sell shares. This class extracts individual transactions
    from a filing summary object produced by the ``edgar`` library.

    Typical usage::

        parser = Form4Parser()
        transactions = parser.extract_insider_transactions(summary, filing_url)
    """

    def extract_insider_transactions_batch(self, filing_batch: List[tuple]) -> List[InsiderTransaction]:
        """
        Process a batch of insider filings sequentially.

        Intended to be called from a worker process inside a
        ``ProcessPoolExecutor``.  Each element of *filing_batch* is a
        ``(summary, filing_url)`` tuple as returned by the ``edgar`` library.

        Args:
            filing_batch: List of ``(ownership_summary, filing_url)`` tuples.

        Returns:
            Flat list of all :class:`InsiderTransaction` objects extracted from
            every filing in the batch.
        """
        all_transactions: List[InsiderTransaction] = []

        for summary, filing_url in filing_batch:
            try:
                transactions = self.extract_insider_transactions(summary, filing_url)
                all_transactions.extend(transactions)
            except Exception as e:
                logger.error(f"Error processing filing in batch: {e}")
                continue

        return all_transactions

    def extract_insider_transactions(self, summary, filing_url: str = None) -> List[InsiderTransaction]:
        """
        Parse a Form 4 ownership summary and return individual transactions.

        Derivative securities (options, RSUs) are skipped so that only
        real share movements are returned.  Transaction codes are mapped to
        human-readable labels:

        * ``P`` → BUY
        * ``S`` → SELL
        * ``A`` → GRANT
        * ``M`` → VESTING
        * ``F`` → TAX  (tax-withholding)
        * ``G`` → GIFT

        Args:
            summary: Ownership summary object from ``edgar`` (result of
                ``filing.obj().get_ownership_summary()``).
            filing_url: URL of the source filing (stored for reference).

        Returns:
            List of :class:`InsiderTransaction` objects, one per non-derivative
            transaction line in the form.
        """
        transactions: List[InsiderTransaction] = []

        # Common fields for all transaction lines in this report
        report_date = summary.reporting_date
        insider_name = summary.insider_name
        position = summary.position
        form_type = summary.form_type
        final_remaining_shares = int(summary.remaining_shares) if summary.remaining_shares else 0

        for tx in summary.transactions:
            # Skip derivatives — only real share movements matter for price impact
            if tx.security_type == "derivative":
                continue

            # Resolve transaction code across different attribute naming conventions
            code = (
                getattr(tx, "code", None)
                or getattr(tx, "transaction_code", None)
                or getattr(tx, "trans_code", None)
            )
            if not code:
                continue
            code = code.upper()

            tx_type_map = {
                "P": "BUY",
                "S": "SELL",
                "A": "GRANT",
                "M": "VESTING",
                "F": "TAX",
                "G": "GIFT",
            }
            tx_type = tx_type_map.get(code, "UNKNOWN")

            try:
                shares = int(tx.shares) if tx.shares else 0
                price = float(tx.price_per_share) if tx.price_per_share else 0.0
                value = shares * price
            except (ValueError, TypeError):
                shares = 0
                price = 0.0
                value = 0.0

            transaction_record = InsiderTransaction(
                date=str(report_date),
                insider_name=str(insider_name),
                position=str(position),
                transaction_type=tx_type,
                shares=shares,
                price=price,
                value=value,
                remaining_shares=final_remaining_shares,
                filing_url=filing_url,
                form=form_type,
            )
            transactions.append(transaction_record)

            if tx_type in ("BUY", "SELL"):
                icon = "🟢" if tx_type == "BUY" else "🔴"
                logger.debug(f"   {icon} {tx_type}: {shares} shares @ ${price:.2f}")

        return transactions
