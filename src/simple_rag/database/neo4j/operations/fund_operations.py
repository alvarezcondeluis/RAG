"""
Fund operations module for Neo4j database.
Combines all fund-related operation mixins into a single class.

Sub-modules:
- provider_ops: Provider, Trust, ShareClass, Person, Sector, CreditRating
- fund_creation_ops: Fund creation, managers, financial highlights, allocations
- holdings_ops: Holdings ingestion, charts, company tickers
- fund_node_ops: Risks, performance, strategy, profile, derivative nodes
"""

from .provider_ops import ProviderOperations
from .fund_creation_ops import FundCreationOperations
from .holdings_ops import HoldingsOperations
from .fund_node_ops import FundNodeOperations


class FundOperations(
    ProviderOperations,
    FundCreationOperations,
    HoldingsOperations,
    FundNodeOperations
):
    """
    All Fund-related CRUD operations for Neo4j database.
    Combines provider, fund creation, holdings, and node operations via mixin inheritance.
    """
    pass
