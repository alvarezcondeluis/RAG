"""
Neo4j database package with modular architecture.

Architecture:
    Neo4jDatabase              (neo4j.py)            - Controller / main entry point
    ├── Neo4jDatabaseBase      (base.py)             - Connection, Docker, query execution
    ├── SchemaManager          (schema_manager.py)   - Constraints, indexes, DB management
    ├── FundOperations         (operations/)          - All fund-related CRUD
    │   ├── ProviderOperations (provider_ops.py)      - Provider/Trust/ShareClass/Person/Sector
    │   ├── FundCreationOps    (fund_creation_ops.py) - Fund creation, managers, highlights
    │   ├── HoldingsOperations (holdings_ops.py)      - Holdings, charts, company tickers
    │   └── FundNodeOperations (fund_node_ops.py)     - Risks, performance, strategy, profile
    └── CompanyOperations      (operations/)          - All company-related CRUD
        ├── CompanyCrudOps     (company_crud_ops.py)  - Company, 10-K, sections, metrics
        └── CompanyIngestionOps(company_ingestion_ops.py) - Bulk EDGAR ingestion

Usage:
    from src.simple_rag.database.neo4j import Neo4jDatabase

    db = Neo4jDatabase(auto_start=True)
    db.create_constraints()
    db.create_fund(fund_data)
    db.close()
"""

# Main controller
from .neo4j import Neo4jDatabase

# Modular components for direct/granular use
from .base import Neo4jDatabaseBase
from .schema_manager import SchemaManager
from .operations.fund_operations import FundOperations
from .operations.company_operations import CompanyOperations

__all__ = [
    'Neo4jDatabase',
    'Neo4jDatabaseBase',
    'SchemaManager',
    'FundOperations',
    'CompanyOperations',
]
