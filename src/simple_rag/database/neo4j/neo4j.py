"""
Neo4j Database Controller.

This is the main entry point for all Neo4j database operations.
It acts as a thin controller/interface that composes functionality from
specialized modules via mixin inheritance:

    - base.py:              Connection management, Docker auto-start, query execution
    - schema_manager.py:    Constraints, indexes, database management
    - operations/
        provider_ops.py:        Provider/Trust/ShareClass/Person/Sector/CreditRating
        fund_creation_ops.py:   Fund creation, managers, financial highlights, allocations
        holdings_ops.py:        Holdings ingestion, charts, company tickers
        fund_node_ops.py:       Risks, performance, strategy, profile, derivative nodes
        company_crud_ops.py:    Company CRUD, 10-K filings, sections, metrics, CEO, insider
        company_ingestion_ops.py: Bulk ingestion from EDGAR filings

Usage:
    from src.simple_rag.database.neo4j.neo4j import Neo4jDatabase

    db = Neo4jDatabase(auto_start=True)
    db.create_constraints()
    db.create_fund(fund_data)
    db.add_10k_filing(...)
    db.close()

Note:
    The original monolithic implementation is preserved in neo4j_original.py for reference.
"""

from typing import Optional, List, Dict, Any
import logging

from .base import Neo4jDatabaseBase
from .schema_manager import SchemaManager
from .operations.fund_operations import FundOperations
from .operations.company_operations import CompanyOperations

logger = logging.getLogger(__name__)


class Neo4jDatabase(SchemaManager, FundOperations, CompanyOperations):
    """
    Unified Neo4j database controller.

    Composes all database functionality via mixin inheritance:
    - Neo4jDatabaseBase:  Connection, Docker, _execute_query, _execute_write, close
    - SchemaManager:      create_constraints, create_indexes, reset_database, get_database_stats
    - FundOperations:     create_fund, add_managers, create_fund_holdings, add_chart_to_fund, ...
    - CompanyOperations:  create_or_update_company, add_10k_filing, ingest_companies_batch, ...

    Example:
        >>> db = Neo4jDatabase(auto_start=True)
        >>> db.create_constraints()
        >>> db.create_fund(fund_data)
        >>> db.close()
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auto_start: bool = True
    ):
        """
        Initialize the Neo4j database controller.

        Args:
            uri: Neo4j connection URI (default from config)
            username: Neo4j username (default from config)
            password: Neo4j password (default from config)
            auto_start: Auto-start Neo4j via Docker Compose if not running
        """
        super().__init__(uri, username, password, auto_start)
        logger.info("✅ Neo4jDatabase initialized")

    # ==================== QUERY METHODS ====================

    def query(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.

        Args:
            cypher_query: The Cypher query string to execute
            parameters: Optional dictionary of parameters for the query

        Returns:
            List of result records as dictionaries

        Example:
            results = db.query(
                "MATCH (f:Fund {ticker: $ticker}) RETURN f",
                {"ticker": "VTI"}
            )
        """
        return self._execute_query(cypher_query, parameters)

    def clear_database(self):
        """Clear entire database (use with caution!)."""
        query = "MATCH (n) DETACH DELETE n"
        self._execute_write(query)
        logger.warning("Database cleared!")

    # ==================== UTILITY ====================

    def __repr__(self):
        return f"<Neo4jDatabase(uri='{self.uri}')>"

    def __str__(self):
        return f"Neo4jDatabase connected to {self.uri}"
