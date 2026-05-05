"""
Asset category classification for SEC N-PORT / N-CSR holdings.

Each holding carries an `asset_category` code (e.g. "EC", "ABS-MBS").
This module maps those codes to structured metadata so downstream code
can group, filter, and label holdings without repeating strings everywhere.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetCategory:
    code: str
    name: str
    description: str
    category: str       # "Bonds", "Equities", or "Alternatives"
    subcategory: str    # e.g. "Securitized Debt", "Corporate Debt", "Common Stock"


ASSET_CATEGORY_MAP: dict[str, AssetCategory] = {
    # ── BONDS (Debt) ──────────────────────────────────────────────────────────
    "ABS-MBS": AssetCategory(
        code="ABS-MBS",
        name="Asset-Backed Security — Mortgage-Backed Security",
        description=(
            "Securities backed by pools of mortgage loans (residential or commercial). "
            "Typical issuers: Fannie Mae, Freddie Mac, Ginnie Mae. "
            "Almost always carry an ISIN; rarely carry an equity ticker."
        ),
        category="Bonds",
        subcategory="Securitized Debt",
    ),
    "ABS-CBDO": AssetCategory(
        code="ABS-CBDO",
        name="Asset-Backed Security — Collateralized Bond/Debt Obligation",
        description=(
            "Structured credit products backed by pools of bonds or loans (CLOs, CDOs). "
            "Cash flows are tranched by seniority."
        ),
        category="Bonds",
        subcategory="Securitized Debt",
    ),
    "ABS-O": AssetCategory(
        code="ABS-O",
        name="Asset-Backed Security — Other",
        description=(
            "ABS backed by non-mortgage assets such as auto loans, credit card "
            "receivables, student loans, or equipment leases."
        ),
        category="Bonds",
        subcategory="Securitized Debt",
    ),
    "DBT": AssetCategory(
        code="DBT",
        name="Debt Instrument",
        description=(
            "Standard fixed- or floating-rate bonds: government bonds, corporate bonds, "
            "municipal bonds, notes, and debentures."
        ),
        category="Bonds",
        subcategory="Corporate / Government Debt",
    ),
    "DCO": AssetCategory(
        code="DCO",
        name="Debt — Convertible",
        description=(
            "Bonds that may be converted into equity shares at a set price. "
            "Hybrid instrument that sits between pure debt and equity."
        ),
        category="Bonds",
        subcategory="Convertible Debt",
    ),
    "DCR": AssetCategory(
        code="DCR",
        name="Debt — Convertible (Contingent/CoCo)",
        description=(
            "Contingent convertible bonds that automatically convert or write down "
            "when the issuer's capital falls below a regulatory threshold."
        ),
        category="Bonds",
        subcategory="Convertible Debt",
    ),
    "DE": AssetCategory(
        code="DE",
        name="Debt — Equity-Linked",
        description=(
            "Bonds whose return is linked to the performance of one or more equity "
            "indices or baskets rather than a fixed coupon."
        ),
        category="Bonds",
        subcategory="Structured / Hybrid Debt",
    ),
    "DFE": AssetCategory(
        code="DFE",
        name="Debt — Foreign Exchange-Linked",
        description=(
            "Bonds whose principal or coupon is linked to FX rates or currency indices."
        ),
        category="Bonds",
        subcategory="Structured / Hybrid Debt",
    ),
    "DIR": AssetCategory(
        code="DIR",
        name="Debt — Interest-Rate-Linked",
        description=(
            "Bonds whose return is linked to interest-rate benchmarks (e.g. CPI-linked "
            "inflation bonds, LIBOR/SOFR floaters with exotic features)."
        ),
        category="Bonds",
        subcategory="Structured / Hybrid Debt",
    ),
    "SN": AssetCategory(
        code="SN",
        name="Structured Note",
        description=(
            "Debt securities with embedded derivatives; payoff depends on underlying "
            "indices, rates, or baskets. Issued by banks as tailored instruments."
        ),
        category="Bonds",
        subcategory="Structured / Hybrid Debt",
    ),

    # ── EQUITIES ─────────────────────────────────────────────────────────────
    "EC": AssetCategory(
        code="EC",
        name="Equity — Common Stock",
        description=(
            "Ordinary shares representing ownership in a company. Typically carry "
            "voting rights and residual claim on assets. Usually have both ISIN and Ticker."
        ),
        category="Equities",
        subcategory="Common Stock",
    ),
    "EP": AssetCategory(
        code="EP",
        name="Equity — Preferred Stock",
        description=(
            "Preferred shares with priority over common stock for dividends and "
            "liquidation proceeds. Often fixed-dividend; may or may not be convertible."
        ),
        category="Equities",
        subcategory="Preferred Stock",
    ),
    "RA": AssetCategory(
        code="RA",
        name="Real Assets / REIT",
        description=(
            "Real estate investment trusts and other real-asset vehicles. Trade like "
            "equities on exchanges; income is primarily derived from property or "
            "infrastructure cash flows."
        ),
        category="Equities",
        subcategory="Real Assets",
    ),

    # ── ALTERNATIVES & CASH ──────────────────────────────────────────────────
    "LON": AssetCategory(
        code="LON",
        name="Loan",
        description=(
            "Direct loans or participations in syndicated loans. Not publicly traded; "
            "may or may not have an ISIN. Includes senior secured, second lien, "
            "and bridge loans."
        ),
        category="Alternatives",
        subcategory="Loans",
    ),
    "STIV": AssetCategory(
        code="STIV",
        name="Short-Term Investment Vehicle",
        description=(
            "Money-market instruments, Treasury bills, commercial paper, and other "
            "cash-equivalent holdings used for liquidity management."
        ),
        category="Alternatives",
        subcategory="Cash & Money Market",
    ),
    "OTHER": AssetCategory(
        code="OTHER",
        name="Other",
        description=(
            "Instruments that do not fit neatly into the categories above, including "
            "derivatives, warrants, rights, partnership interests, and miscellaneous "
            "alternative investments."
        ),
        category="Alternatives",
        subcategory="Other",
    ),
}


def get_category(code: str) -> AssetCategory:
    """Return the AssetCategory for a given code.

    Raises KeyError with a helpful message if the code is unknown.
    """
    try:
        return ASSET_CATEGORY_MAP[code]
    except KeyError:
        known = ", ".join(sorted(ASSET_CATEGORY_MAP))
        raise KeyError(
            f"Unknown asset category code {code!r}. Known codes: {known}"
        ) from None


def is_bond(code: str) -> bool:
    """Return True if the code maps to the Bonds category."""
    cat = ASSET_CATEGORY_MAP.get(code)
    return cat is not None and cat.category == "Bonds"


def is_equity(code: str) -> bool:
    """Return True if the code maps to the Equities category."""
    cat = ASSET_CATEGORY_MAP.get(code)
    return cat is not None and cat.category == "Equities"
