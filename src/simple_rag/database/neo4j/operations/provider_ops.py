"""
Provider and Trust operations for Neo4j database.
"""

from typing import Optional, List, Dict, Any
import logging
from ..base import Neo4jDatabaseBase

logger = logging.getLogger(__name__)


class ProviderOperations(Neo4jDatabaseBase):
    """Provider and Trust CRUD operations."""
    
    def get_or_create_provider(self, name: str) -> Dict[str, Any]:
        """
        Get or create Provider node.
        
        Args:
            name: Name of the provider/fund company
            
        Returns:
            Dictionary containing the Provider node properties
            
        Example:
            provider = db.get_or_create_provider("The Vanguard Group, Inc.")
        """
        if not name or name.strip() == "":
            logger.warning("Cannot create provider with empty name")
            return None
            
        query = """
        MERGE (p:Provider {name: $name})
        ON CREATE SET
            p.createdAt = timestamp()
        ON MATCH SET
            p.updatedAt = timestamp()
        RETURN p
        """
        
        try:
            result = self._execute_write(query, {"name": name.strip()})
            if result:
                logger.info(f"✅ Got/Created provider: {name}")
                return result[0]["p"]
            return None
        except Exception as e:
            logger.error(f"❌ Error creating provider {name}: {e}")
            return None
    
    def get_all_providers(self) -> List[Dict[str, Any]]:
        """
        Get all providers in the database.
        
        Returns:
            List of provider dictionaries with their properties
            
        Example:
            providers = db.get_all_providers()
            for provider in providers:
                print(f"{provider['name']} - {provider['fund_count']} funds")
        """
        query = """
        MATCH (p:Provider)
        OPTIONAL MATCH (p)-[:MANAGES]->(t:Trust)-[:ISSUES]->(f:Fund)
        WITH p, count(DISTINCT f) as fund_count
        RETURN p.name as name,
               p.createdAt as created_at,
               p.updatedAt as updated_at,
               fund_count
        ORDER BY p.name
        """
        
        try:
            result = self._execute_query(query)
            logger.info(f"Found {len(result)} providers")
            return result
        except Exception as e:
            logger.error(f"Error getting providers: {e}")
            return []
    
    def get_provider_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific provider by name.
        
        Args:
            name: Name of the provider to find
            
        Returns:
            Provider dictionary with properties or None if not found
            
        Example:
            provider = db.get_provider_by_name("The Vanguard Group, Inc.")
        """
        query = """
        MATCH (p:Provider {name: $name})
        OPTIONAL MATCH (p)-[:MANAGES]->(t:Trust)-[:ISSUES]->(f:Fund)
        WITH p, count(DISTINCT f) as fund_count, collect(DISTINCT f.ticker) as fund_tickers
        RETURN p.name as name,
               p.createdAt as created_at,
               p.updatedAt as updated_at,
               fund_count,
               fund_tickers
        """
        
        try:
            result = self._execute_query(query, {"name": name})
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Error getting provider {name}: {e}")
            return None
    
    def get_provider_funds(self, provider_name: str) -> List[Dict[str, Any]]:
        """
        Get all funds managed by a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            List of fund dictionaries
            
        Example:
            funds = db.get_provider_funds("The Vanguard Group, Inc.")
            for fund in funds:
                print(f"{fund['ticker']}: {fund['name']}")
        """
        query = """
        MATCH (p:Provider {name: $provider_name})-[:MANAGES]->(t:Trust)-[:ISSUES]->(f:Fund)
        RETURN f.ticker as ticker,
               f.name as name,
               f.netAssets as net_assets,
               f.expenseRatio as expense_ratio,
               t.name as trust_name
        ORDER BY f.netAssets DESC
        """
        
        try:
            result = self._execute_query(query, {"provider_name": provider_name})
            logger.info(f"Found {len(result)} funds for provider {provider_name}")
            return result
        except Exception as e:
            logger.error(f"Error getting funds for provider {provider_name}: {e}")
            return []
    
    def update_provider_info(self, name: str, **properties) -> Optional[Dict[str, Any]]:
        """
        Update provider properties.
        
        Args:
            name: Name of the provider to update
            **properties: Key-value pairs of properties to update
            
        Returns:
            Updated provider dictionary or None if not found
            
        Example:
            db.update_provider_info(
                "The Vanguard Group, Inc.",
                website="https://www.vanguard.com",
                headquarters="Malvern, PA"
            )
        """
        if not properties:
            logger.warning("No properties provided for update")
            return None
        
        # Build SET clause dynamically
        set_clauses = [f"p.{key} = ${key}" for key in properties.keys()]
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        MATCH (p:Provider {{name: $name}})
        SET {set_clause}, p.updatedAt = timestamp()
        RETURN p
        """
        
        params = {"name": name}
        params.update(properties)
        
        try:
            result = self._execute_write(query, params)
            if result:
                logger.info(f"✅ Updated provider: {name}")
                return result[0]["p"]
            else:
                logger.warning(f"Provider not found: {name}")
                return None
        except Exception as e:
            logger.error(f"❌ Error updating provider {name}: {e}")
            return None
    
    def get_or_create_trust(self, provider_name: str, name: str) -> Dict[str, Any]:
        """
        Get or create Trust node.
        FIXED: Uses Composite ID to prevent merging generic trust names across providers.
        """
        # 1. Generate a Unique Composite ID
        # e.g. "THE_VANGUARD_GROUP_INC_INDEX_FUNDS" vs "BLACKROCK_INC_INDEX_FUNDS"
        trust_id = f"{provider_name}_{name}".upper().replace(" ", "_").strip()
        
        query = """
        MERGE (p:Provider {name: $provider_name})
        
        // 2. MERGE on the ID, not the Name
        MERGE (t:Trust {id: $trust_id})
        ON CREATE SET 
            t.name = $name,
            t.created_at = timestamp()
            
        MERGE (p)-[:MANAGES]->(t)
        RETURN t, p
        """
        
        # 3. Pass the new ID parameter
        result = self._execute_write(query, {
            "name": name, 
            "provider_name": provider_name,
            "trust_id": trust_id
        })
        
        if result:
            return result[0]["t"]
        return None
    
    def get_or_create_share_class(self, name: str, description: str) -> Dict[str, Any]:
        """Get or create ShareClass node."""
        query = """
        MERGE (sc:ShareClass {name: $name})
        ON CREATE SET
            sc.description = $description,
            sc.created_at = datetime()
        ON MATCH SET
            sc.description = $description,
            sc.updated_at = datetime()
        RETURN sc
        """
        result = self._execute_write(query, {"name": name, "description": description})
        if result:
            logger.info(f"Got/Created share class: {name}")
            return result[0]["sc"]
        return None

    def get_or_create_person(self, name: str) -> Dict[str, Any]:
        """Get or create Person node."""
        query = """
        MERGE (p:Person {name: $name})
        RETURN p
        """
        result = self._execute_write(query, {"name": name})
        if result:
            logger.info(f"Got/Created person: {name}")
            return result[0]["p"]
        return None
    
    def get_or_create_sector(self, name: str) -> Dict[str, Any]:
        """Get or create Sector node."""
        query = """
        MERGE (s:Sector {name: $name})
        RETURN s
        """
        result = self._execute_write(query, {"name": name})
        if result:
            logger.info(f"Got/Created sector: {name}")
            return result[0]["s"]
        return None
    
    def get_or_create_credit_rating(self, name: str) -> Dict[str, Any]:
        """Get or create CreditRating node."""
        query = """
        MERGE (c:CreditRating {name: $name})
        RETURN c
        """
        result = self._execute_write(query, {"name": name})
        if result:
            logger.info(f"Got/Created credit rating: {name}")
            return result[0]["c"]
        return None
