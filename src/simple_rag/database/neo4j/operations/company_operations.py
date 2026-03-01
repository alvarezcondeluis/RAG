"""
Company operations module for Neo4j database.
Combines all company-related operation mixins into a single class.

Sub-modules:
- company_crud_ops: Company CRUD, 10-K filings, sections, metrics, CEO, insider transactions
- company_ingestion_ops: Bulk ingestion of company data from EDGAR filings
                         (inherits from CompanyCrudOperations so it can call CRUD methods)
"""

from .company_ingestion_ops import CompanyIngestionOperations


class CompanyOperations(CompanyIngestionOperations):
    """
    All Company/10-K related CRUD operations for Neo4j database.
    
    Inheritance chain:
        CompanyOperations -> CompanyIngestionOperations -> CompanyCrudOperations -> Neo4jDatabaseBase
    
    This gives access to all company CRUD + batch ingestion methods.
    """
    pass
