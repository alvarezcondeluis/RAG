import subprocess
import time
import os
from typing import Optional, List, Dict, Any
import logging
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
from .config import settings
from src.simple_rag.models.fund import FundData

logger = logging.getLogger(__name__)


class Neo4jDatabase:
    """
    Neo4j database manager for fund data using native Neo4j driver with Cypher queries.
    
    Usage:
        db = Neo4jDatabase(auto_start=True)
        fund = db.create_fund(ticker="VTI", name="Vanguard Total Stock Market ETF")
        results = db.get_fund_by_ticker("VTI")
        db.close()
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auto_start: bool = True
    ):
        """
        Initialize Neo4j database connection using native driver.
        
        Args:
            uri: Neo4j connection URI (default from config)
            username: Neo4j username (default from config)
            password: Neo4j password (default from config)
            auto_start: Auto-start Neo4j via Docker Compose if not running
        """
        self.uri = uri or settings.NEO4J_URL
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.container_name = settings.NEO4J_CONTAINER_NAME
        self.driver: Optional[Driver] = None

        if auto_start:
            self._ensure_neo4j_running()
        
        self._connect()
    
    def _is_neo4j_running(self) -> bool:
        """Check if Neo4j is accessible."""
        try:
            test_driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            test_driver.verify_connectivity()
            test_driver.close()
            return True
        except (ServiceUnavailable, AuthError, Exception):
            return False
    
    def _start_neo4j_docker(self) -> bool:
        """Start Neo4j via Docker Compose."""
        try:
            
            # Check if container is already running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if self.container_name in result.stdout:
                logger.info(f"Neo4j container {self.container_name} is already running")
                return True
            
            # Check if docker-compose.yml exists
            compose_file = settings.get_docker_compose_path()
            if not compose_file.exists():
                logger.error(f"docker-compose.yml not found at {compose_file}")
                return False
            project_dir = os.path.dirname(compose_file)

            # Start using docker-compose
            logger.info(f"Starting Neo4j using docker-compose from {compose_file}")
            subprocess.run(
                [
                    "docker", "compose", 
                    "-f", compose_file,       # Point to the specific file
                    "--project-directory", project_dir, # Ensure volumes map correctly
                    "up", "-d"                     # Run in background
                ], 
                check=True,
                capture_output=True
            )
            
            # Wait for Neo4j to be ready
            logger.info("Waiting for Neo4j to be ready...")
            for i in range(30):
                time.sleep(2)
                if self._is_neo4j_running():
                    logger.info("Neo4j is ready!")
                    return True
            
            logger.warning("Neo4j did not become ready in time")
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Neo4j via docker-compose: {e}")
            return False
        except FileNotFoundError:
            logger.error("Docker or docker-compose not found. Please install Docker to use auto-start.")
            return False
    
    def _ensure_neo4j_running(self):
        """Ensure Neo4j is running, start if needed."""
        if not self._is_neo4j_running():
            logger.info("Neo4j not running, attempting to start...")
            if not self._start_neo4j_docker():
                logger.warning("Could not auto-start Neo4j. Please start it manually.")
        else:
            logger.info("Neo4j is already running")
    
    def _connect(self):
        """Establish connection to Neo4j using native driver."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600
            )
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def _execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a write Cypher query in a transaction."""
        with self.driver.session() as session:
            result = session.execute_write(lambda tx: list(tx.run(query, parameters or {})))
            return [record.data() for record in result]
    
    # ==================== DATABASE MANAGEMENT ====================
    
    def reset_database(self):
        """Delete all nodes and relationships from the database."""
        with self.driver.session() as session:
            query = "MATCH (n) DETACH DELETE n"
            session.run(query)
            print("✅ Database reset complete")
    
    def delete_all_funds(self):
        """Delete only Fund nodes and their relationships."""
        with self.driver.session() as session:
            query = "MATCH (f:Fund) DETACH DELETE f"
            session.run(query)
            print("✅ Deleted all Fund nodes")
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Get statistics about the database contents.
        
        Returns:
            Dictionary with counts of each node type and total relationships
        """
        with self.driver.session() as session:
            # Count nodes by label
            node_query = """
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
            YIELD value
            RETURN label, value.count as count
            """
            
            # Fallback if APOC not available
            simple_node_query = """
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            """
            
            # Count relationships
            rel_query = """
            MATCH ()-[r]->()
            RETURN count(r) as total_relationships
            """
            
            stats = {}
            
            try:
                # Try APOC version
                result = session.run(node_query)
                for record in result:
                    stats[record["label"]] = record["count"]
            except Exception:
                # Fallback to simple version
                result = session.run(simple_node_query)
                for record in result:
                    label = record["label"] if record["label"] else "Unknown"
                    stats[label] = record["count"]
            
            # Get relationship count
            rel_result = session.run(rel_query)
            stats["total_relationships"] = rel_result.single()["total_relationships"]
            
            return stats
    
    # ==================== SCHEMA SETUP ====================
    
    def create_constraints(self):
        """
        Creates uniqueness constraints and indexes based on the Pydantic schema.
        Run this ONCE before ingesting data.
        """
        queries = [
            # --- 1. CORE FUND ENTITIES ---
            
            "CREATE CONSTRAINT fund_ticker_unique IF NOT EXISTS FOR (f:Fund) REQUIRE f.ticker IS UNIQUE",
            
            # Fast lookup by Name (since ticker might be "N/A")
            "CREATE INDEX fund_name_index IF NOT EXISTS FOR (f:Fund) ON (f.name)",

            # --- 2. ORGANIZATIONS & PEOPLE ---
            # Providers (Registrants) should not be duplicated
            "CREATE CONSTRAINT trust_name_unique IF NOT EXISTS FOR (t:Trust) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT provider_name_unique IF NOT EXISTS FOR (p:Provider) REQUIRE p.name IS UNIQUE",
            
            # Managers should be unique nodes
            "CREATE CONSTRAINT manager_name_unique IF NOT EXISTS FOR (m:Manager) REQUIRE m.name IS UNIQUE",
        ]

        with self.driver.session() as session:
            print("🚧 Setting up constraints...")
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    print(f"⚠️ Warning: {e}")
            print("✅ Constraints and Indexes configured.")
    
    # ==================== NODE CREATION ====================
    
    def get_or_create_provider(self, name: str) -> Dict[str, Any]:
        """Get or create Provider node."""
        query = """
        MERGE (p:Provider {name: $name})
        RETURN p
        """
        result = self._execute_write(query, {"name": name})
        if result:
            logger.info(f"Got/Created provider: {name}")
            print(result)
            return result[0]["p"]
        return None
    
    def get_or_create_trust(self, provider_name: str, name: str) -> Dict[str, Any]:
        """Get or create Trust node and link it to a Provider via MANAGES relationship."""
        query = """
        MERGE (p:Provider {name: $provider_name})
        MERGE (t:Trust {name: $name})
        MERGE (p)-[:MANAGES]->(t)
        RETURN t, p
        """
        result = self._execute_write(query, {"name": name, "provider_name": provider_name})
        if result:
            logger.info(f"Got/Created trust: {name} managed by provider: {provider_name}")
            return result[0]["t"]
        return None
        
    def create_fund(self, fund: FundData):
        """
        Create or update a Fund node with all its relationships.
        
        Args:
            fund: FundData object containing all fund information
            
        Returns:
            Dict with created nodes or None if error
        """
        try:
            print(f"📊 Creating fund: {fund.ticker} - {fund.name}")
            
            # Create performance ID by combining ticker and report_date
            perf_id = f"{fund.ticker}_{fund.report_date.isoformat()}" if fund.report_date else fund.ticker
            
            params = {
                # Identity
                "ticker": fund.ticker,
                "name": fund.name,
                "trust": fund.registrant,
                "provider": fund.provider,
                "exchange": fund.security_exchange,
                "perf_id": perf_id,
                # Classification
                "share_class": fund.share_class,
                # Metrics
                "net_assets": fund.net_assets,
                "expense_ratio": fund.expense_ratio,
                "costs_per_10k": fund.costs_per_10k,
                "advisory_fees": fund.advisory_fees,
                "turnover_rate": fund.turnover_rate,
                "n_holdings": fund.n_holdings,
                "report_date": fund.report_date,
                # Profile
                "embedding": fund.embedding,
                "summary_prospectus": fund.summary_prospectus,
                "risks": fund.risks,
                "objective": fund.objective,
                "strategies": fund.strategies,
                "performance_commentary": fund.performance_commentary,
            }

            query = """
            MERGE (t:Trust {name: $trust})
            MERGE (p:Provider {name: $provider})
            MERGE (f:Fund {ticker: $ticker})
            MERGE (sc:ShareClass {name: $share_class})
            ON CREATE SET
                f.name = $name,
                f.security_exchange = $exchange,
                f.share_class = $share_class,
                f.net_assets = $net_assets,
                f.expense_ratio = $expense_ratio,
                f.costs_per_10k = $costs_per_10k,
                f.advisory_fees = $advisory_fees,
                f.turnover_rate = $turnover_rate,
                f.n_holdings = $n_holdings,
                f.report_date = $report_date,
                f.created_at = timestamp()
            ON MATCH SET
                f.name = $name,
                f.security_exchange = $exchange,
                f.share_class = $share_class,
                f.net_assets = $net_assets,
                f.expense_ratio = $expense_ratio,
                f.costs_per_10k = $costs_per_10k,
                f.advisory_fees = $advisory_fees,
                f.turnover_rate = $turnover_rate,
                f.n_holdings = $n_holdings,
                f.report_date = $report_date,
                f.updated_at = timestamp()
            
            MERGE (t)-[:ISSUES]->(f)
            MERGE (p)-[:MANAGES]->(t)
            MERGE (f)-[:DEFINED_BY]->(prof)
            ON CREATE SET
                prof.created_at = timestamp()
                prof.summary_prospectus = $summary_prospectus,
                prof.risks = $risks,
                prof.objective = $objective,
                prof.strategies = $strategies,
                prof.performance_commentary = $performance_commentary,
            ON MATCH SET
                prof.updated_at = timestamp()
                prof.summary_prospectus = $summary_prospectus,
                prof.risks = $risks,
                prof.objective = $objective,
                prof.strategies = $strategies,
                prof.performance_commentary = $performance_commentary,
            MERGE (f)-[:HAS_SHARE_CLASS]->(sc)
            
            RETURN f, t, p
            """
            
            result = self._execute_write(query, params)
            
            if result:
                print(f"✅ Successfully created/updated fund: {fund.ticker}")
                return result[0]
            else:
                print(f"⚠️  Warning: No result returned for fund: {fund.ticker}")
                return None
                
        except Exception as e:
            print(f"❌ ERROR creating fund {fund.ticker}: {str(e)}")
            logger.error(f"Failed to create fund {fund.ticker}: {e}", exc_info=True)
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


    def get_or_create_asset(self, name: str, ticker: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Get or create Asset node."""
        properties = {"name": name}
        if ticker:
            properties["ticker"] = ticker
        properties.update(kwargs)
        
        # Build SET clause
        set_clauses = ", ".join([f"a.{key} = ${key}" for key in properties.keys()])
        
        # Match by name and ticker if provided
        match_clause = "MERGE (a:Asset {name: $name" + (", ticker: $ticker" if ticker else "") + "})"
        
        query = f"""
        {match_clause}
        SET {set_clauses}
        RETURN a
        """
        
        result = self._execute_write(query, properties)
        if result:
            logger.info(f"Got/Created asset: {name}")
            return result[0]["a"]
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
    
    # ==================== RELATIONSHIP CREATION ====================
    
    def link_provider_trust(self, provider_name: str, trust_name: str):
        """Create MANAGES relationship between Provider and Trust."""
        query = """
        MATCH (p:Provider {name: $provider_name})
        MATCH (t:Trust {name: $trust_name})
        MERGE (p)-[:MANAGES]->(t)
        """
        self._execute_write(query, {"provider_name": provider_name, "trust_name": trust_name})
        logger.info(f"Linked {provider_name} -> {trust_name}")
    
    def link_trust_fund(self, trust_name: str, fund_ticker: str):
        """Create ISSUES relationship between Trust and Fund."""
        query = """
        MATCH (t:Trust {name: $trust_name})
        MATCH (f:Fund {ticker: $fund_ticker})
        MERGE (t)-[:ISSUES]->(f)
        """
        self._execute_write(query, {"trust_name": trust_name, "fund_ticker": fund_ticker})
        logger.info(f"Linked {trust_name} -> {fund_ticker}")
    
    def link_fund_asset(
        self,
        fund_ticker: str,
        asset_name: str,
        weight: Optional[float] = None,
        n_shares: Optional[float] = None,
        date: Optional[str] = None,
        market_value: Optional[float] = None
    ):
        """Create HAS_ASSET relationship between Fund and Asset."""
        properties = {}
        if weight is not None:
            properties["weight"] = weight
        if n_shares is not None:
            properties["n_shares"] = n_shares
        if date is not None:
            properties["date"] = date
        if market_value is not None:
            properties["market_value"] = market_value
        
        set_clause = ", ".join([f"r.{key} = ${key}" for key in properties.keys()]) if properties else ""
        set_statement = f"SET {set_clause}" if set_clause else ""
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        MATCH (a:Asset {{name: $asset_name}})
        MERGE (f)-[r:HAS_ASSET]->(a)
        {set_statement}
        """
        
        params = {"fund_ticker": fund_ticker, "asset_name": asset_name}
        params.update(properties)
        self._execute_write(query, params)
        logger.info(f"Linked fund {fund_ticker} -> asset {asset_name}")
    
    def link_fund_manager(
        self,
        fund_ticker: str,
        person_name: str,
        since: Optional[str] = None,
        role: Optional[str] = None
    ):
        """Create MANAGED_BY relationship between Fund and Person."""
        properties = {}
        if since:
            properties["since"] = since
        if role:
            properties["role"] = role
        
        set_clause = ", ".join([f"r.{key} = ${key}" for key in properties.keys()]) if properties else ""
        set_statement = f"SET {set_clause}" if set_clause else ""
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        MATCH (p:Person {{name: $person_name}})
        MERGE (f)-[r:MANAGED_BY]->(p)
        {set_statement}
        """
        
        params = {"fund_ticker": fund_ticker, "person_name": person_name}
        params.update(properties)
        self._execute_write(query, params)
        logger.info(f"Linked fund {fund_ticker} -> manager {person_name}")
    
    def link_fund_sector(
        self,
        fund_ticker: str,
        sector_name: str,
        weight: Optional[float] = None
    ):
        """Create HAS_SECTOR_ALLOCATION relationship."""
        set_statement = "SET r.weight = $weight" if weight is not None else ""
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        MATCH (s:Sector {{name: $sector_name}})
        MERGE (f)-[r:HAS_SECTOR_ALLOCATION]->(s)
        {set_statement}
        """
        
        params = {"fund_ticker": fund_ticker, "sector_name": sector_name}
        if weight is not None:
            params["weight"] = weight
        
        self._execute_write(query, params)
        logger.info(f"Linked fund {fund_ticker} -> sector {sector_name}")
    
    def link_fund_credit_rating(
        self,
        fund_ticker: str,
        rating_name: str,
        weight: Optional[float] = None
    ):
        """Create HAS_CREDIT_RATING relationship."""
        set_statement = "SET r.weight = $weight" if weight is not None else ""
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        MATCH (c:CreditRating {{name: $rating_name}})
        MERGE (f)-[r:HAS_CREDIT_RATING]->(c)
        {set_statement}
        """
        
        params = {"fund_ticker": fund_ticker, "rating_name": rating_name}
        if weight is not None:
            params["weight"] = weight
        
        self._execute_write(query, params)
        logger.info(f"Linked fund {fund_ticker} -> credit rating {rating_name}")
    
    # ==================== COMPLEX NODE CREATION ====================
    
    def create_risks_node(
        self,
        fund_ticker: str,
        content: str,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create Risks node and link to Fund."""
        set_statement = "SET r.date = $date" if date else ""
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        CREATE (risks:Risks {{content: $content}})
        CREATE (f)-[r:HAS_RISKS]->(risks)
        {set_statement}
        RETURN risks
        """
        
        params = {"fund_ticker": fund_ticker, "content": content}
        if date:
            params["date"] = date
        
        result = self._execute_write(query, params)
        if result:
            logger.info(f"Created risks node for fund {fund_ticker}")
            return result[0]["risks"]
        return None
    
    def create_performance_node(
        self,
        fund_ticker: str,
        return_1y: Optional[float] = None,
        return_5y: Optional[float] = None,
        return_10y: Optional[float] = None,
        return_inception: Optional[float] = None,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create TrailingPerformance node and link to Fund."""
        perf_props = {}
        if return_1y is not None:
            perf_props["return_1y"] = return_1y
        if return_5y is not None:
            perf_props["return_5y"] = return_5y
        if return_10y is not None:
            perf_props["return_10y"] = return_10y
        if return_inception is not None:
            perf_props["return_inception"] = return_inception
        
        perf_set = ", ".join([f"p.{k} = ${k}" for k in perf_props.keys()])
        rel_set = "SET r.date = $date" if date else ""
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        CREATE (p:TrailingPerformance)
        SET {perf_set}
        CREATE (f)-[r:HAS_PERFORMANCE]->(p)
        {rel_set}
        RETURN p
        """
        
        params = {"fund_ticker": fund_ticker}
        params.update(perf_props)
        if date:
            params["date"] = date
        
        result = self._execute_write(query, params)
        if result:
            logger.info(f"Created performance node for fund {fund_ticker}")
            return result[0]["p"]
        return None
    
    def create_annual_performance_node(
        self,
        fund_ticker: str,
        year: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create AnnualPerformance node and link to Fund."""
        properties = {"year": year}
        properties.update(kwargs)
        
        set_clauses = ", ".join([f"ap.{k} = ${k}" for k in properties.keys()])
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        CREATE (ap:AnnualPerformance)
        SET {set_clauses}
        CREATE (f)-[:HAS_ANNUAL_PERFORMANCE]->(ap)
        RETURN ap
        """
        
        params = {"fund_ticker": fund_ticker}
        params.update(properties)
        
        result = self._execute_write(query, params)
        if result:
            logger.info(f"Created annual performance node for fund {fund_ticker}, year {year}")
            return result[0]["ap"]
        return None
    
    def create_strategy_node(
        self,
        fund_ticker: str,
        full_text_md: str,
        objective: Optional[str] = None,
        strategies: Optional[str] = None,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create InvestmentStrategy node and link to Fund."""
        properties = {"full_text_md": full_text_md}
        if objective:
            properties["objective"] = objective
        if strategies:
            properties["strategies"] = strategies
        
        set_clauses = ", ".join([f"s.{k} = ${k}" for k in properties.keys()])
        rel_set = "SET r.date = $date" if date else ""
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        CREATE (s:InvestmentStrategy)
        SET {set_clauses}
        CREATE (f)-[r:WITH_SUMMARY_PROSPECTUS]->(s)
        {rel_set}
        RETURN s
        """
        
        params = {"fund_ticker": fund_ticker}
        params.update(properties)
        if date:
            params["date"] = date
        
        result = self._execute_write(query, params)
        if result:
            logger.info(f"Created strategy node for fund {fund_ticker}")
            return result[0]["s"]
        return None
    
    def create_profile_node(
        self,
        fund_ticker: str,
        text: str,
        embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Create Profile node with embeddings and link to Fund."""
        properties = {"text": text}
        if embedding:
            properties["embedding"] = embedding
        
        set_clauses = ", ".join([f"p.{k} = ${k}" for k in properties.keys()])
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        CREATE (p:Profile)
        SET {set_clauses}
        CREATE (f)-[:DEFINED_BY]->(p)
        RETURN p
        """
        
        params = {"fund_ticker": fund_ticker}
        params.update(properties)
        
        result = self._execute_write(query, params)
        if result:
            logger.info(f"Created profile node for fund {fund_ticker}")
            return result[0]["p"]
        return None
    
    def create_derivative_node(
        self,
        fund_ticker: str,
        name: str,
        fair_value_lvl: Optional[str] = None,
        derivative_type: Optional[str] = None,
        weight: Optional[float] = None,
        date: Optional[str] = None,
        termination_date: Optional[str] = None,
        unrealized_pnl: Optional[float] = None,
        amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create Derivative node and link to Fund."""
        # Build derivative properties
        deriv_props = {"name": name}
        if fair_value_lvl:
            deriv_props["fair_value_lvl"] = fair_value_lvl
        if derivative_type:
            deriv_props["type"] = derivative_type
        
        # Build relationship properties
        rel_props = {}
        if weight is not None:
            rel_props["weight"] = weight
        if date:
            rel_props["date"] = date
        if termination_date:
            rel_props["termination_date"] = termination_date
        if unrealized_pnl is not None:
            rel_props["unrealized_pnl"] = unrealized_pnl
        if amount is not None:
            rel_props["amount"] = amount
        
        deriv_set = ", ".join([f"d.{k} = $deriv_{k}" for k in deriv_props.keys()])
        rel_set = ", ".join([f"r.{k} = $rel_{k}" for k in rel_props.keys()]) if rel_props else ""
        rel_statement = f"SET {rel_set}" if rel_set else ""
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        MERGE (d:Derivative {{name: $deriv_name}})
        SET {deriv_set}
        MERGE (f)-[r:HAS_DERIVATIVE]->(d)
        {rel_statement}
        RETURN d
        """
        
        params = {"fund_ticker": fund_ticker}
        params.update({f"deriv_{k}": v for k, v in deriv_props.items()})
        params.update({f"rel_{k}": v for k, v in rel_props.items()})
        
        result = self._execute_write(query, params)
        if result:
            logger.info(f"Created/Updated derivative {name} for fund {fund_ticker}")
            return result[0]["d"]
        return None
    
    # ==================== QUERY METHODS ====================
    
    def get_fund_by_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get fund by ticker."""
        query = """
        MATCH (f:Fund {ticker: $ticker})
        RETURN f
        """
        result = self._execute_query(query, {"ticker": ticker})
        if result:
            return result[0]["f"]
        return None
    
    def get_fund_with_holdings(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get fund with all its holdings."""
        query = """
        MATCH (f:Fund {ticker: $ticker})
        OPTIONAL MATCH (f)-[r:HAS_ASSET]->(a:Asset)
        RETURN f, 
               collect({asset: a, weight: r.weight, n_shares: r.n_shares, 
                       market_value: r.market_value, date: r.date}) as holdings
        """
        result = self._execute_query(query, {"ticker": ticker})
        if result:
            return {
                "fund": result[0]["f"],
                "holdings": result[0]["holdings"]
            }
        return None
    
    def get_all_funds(self) -> List[Dict[str, Any]]:
        """Get all funds."""
        query = """
        MATCH (f:Fund)
        RETURN f
        ORDER BY f.ticker
        """
        result = self._execute_query(query)
        return [record["f"] for record in result]
    
    def get_funds_by_provider(self, provider_name: str) -> List[Dict[str, Any]]:
        """Get all funds managed by a provider."""
        query = """
        MATCH (p:Provider {name: $provider_name})-[:MANAGES]->(t:Trust)-[:ISSUES]->(f:Fund)
        RETURN f
        ORDER BY f.ticker
        """
        result = self._execute_query(query, {"provider_name": provider_name})
        return [record["f"] for record in result]
    
    def search_funds_by_name(self, name_pattern: str) -> List[Dict[str, Any]]:
        """Search funds by name pattern."""
        query = """
        MATCH (f:Fund)
        WHERE toLower(f.name) CONTAINS toLower($pattern)
        RETURN f
        ORDER BY f.ticker
        """
        result = self._execute_query(query, {"pattern": name_pattern})
        return [record["f"] for record in result]
    
    def delete_fund(self, ticker: str):
        """Delete fund and all its relationships."""
        query = """
        MATCH (f:Fund {ticker: $ticker})
        DETACH DELETE f
        """
        self._execute_write(query, {"ticker": ticker})
        logger.info(f"Deleted fund: {ticker}")
    
    def clear_database(self):
        """Clear entire database (use with caution!)."""
        query = "MATCH (n) DETACH DELETE n"
        self._execute_write(query)
        logger.warning("Database cleared!")
