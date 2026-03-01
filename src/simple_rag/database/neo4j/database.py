"""
Unified Neo4j Database Controller.

This module provides a clean, modular implementation of Neo4jDatabase that uses
mixin inheritance to combine all operations into a single unified interface.

Architecture:
    Neo4jDatabase
    ├── Neo4jDatabaseBase      (base.py)           - Connection, Docker, query execution
    ├── SchemaManager          (schema_manager.py)  - Constraints, indexes, DB management
    ├── FundOperations         (operations/)        - All fund-related CRUD
    │   ├── ProviderOperations                      - Provider/Trust/ShareClass/Person/Sector
    │   ├── FundCreationOperations                  - Fund creation, managers, highlights, allocations
    │   ├── HoldingsOperations                      - Holdings, charts, company tickers
    │   └── FundNodeOperations                      - Risks, performance, strategy, profile, derivative
    └── CompanyOperations      (operations/)        - All company-related CRUD
        ├── CompanyCrudOperations                   - Company, 10-K, sections, metrics, CEO, insider
        └── CompanyIngestionOperations              - Bulk ingestion from EDGAR

Usage:
    from src.simple_rag.database.neo4j.database import Neo4jDatabase
    
    db = Neo4jDatabase(auto_start=True)
    db.create_constraints()
    db.create_fund(fund_data)
    db.add_10k_filing(...)
    db.close()
"""

from typing import Optional, List, Dict, Any
from datetime import date
import logging
from .base import Neo4jDatabaseBase
from .schema_manager import SchemaManager
from .operations.fund_operations import FundOperations
from .operations.company_operations import CompanyOperations

logger = logging.getLogger(__name__)


class Neo4jDatabase(SchemaManager, FundOperations, CompanyOperations):
    """
    Unified Neo4j database controller using modular mixin architecture.
    
    This class combines functionality from:
    - Neo4jDatabaseBase: Connection management, Docker startup, query execution
    - SchemaManager: Constraints, indexes, database management
    - FundOperations: All fund-related CRUD (providers, trusts, funds, holdings, etc.)
    - CompanyOperations: All company-related CRUD (companies, 10-K filings, etc.)
    
    All methods are directly available on this class via mixin inheritance.
    
    Example:
        >>> db = Neo4jDatabase(auto_start=True)
        >>> stats = db.get_database_stats()
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
        Initialize the unified Neo4j database controller.
        
        Args:
            uri: Neo4j connection URI (default from config)
            username: Neo4j username (default from config)
            password: Neo4j password (default from config)
            auto_start: Auto-start Neo4j via Docker Compose if not running
        """
        # Initialize the base class (connection management)
        super().__init__(uri, username, password, auto_start)
        logger.info("✅ Neo4jDatabase controller initialized (modular architecture)")
    
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
    
    # ==================== ADDITIONAL UTILITY METHODS ====================
    
    def __repr__(self):
        """String representation."""
        return f"<Neo4jDatabase(uri='{self.uri}', modular=True)>"
    
    def __str__(self):
        """Human-readable string."""
        return f"Neo4jDatabase connected to {self.uri} (modular architecture)"
