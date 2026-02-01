import subprocess
import time
import os
from typing import Optional, List, Dict, Any
from datetime import date
import logging
import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
from .config import settings
from src.simple_rag.models.fund import FundData, AverageReturnSnapshot

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
        
        # Ensure Neo4j is actually ready before connecting
        max_wait_attempts = 10
        for attempt in range(max_wait_attempts):
            if self._is_neo4j_running():
                break
            if attempt < max_wait_attempts - 1:  # Don't sleep on last attempt
                logger.info(f"Waiting for Neo4j to be ready... (attempt {attempt + 1}/{max_wait_attempts})")
                time.sleep(3)
        
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
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {self.uri}. Is Neo4j running?")
            logger.error(f"Try: docker ps to check if Neo4j container is running")
            raise
        except AuthError as e:
            logger.error(f"Authentication failed for Neo4j. Check username/password.")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.error(f"Connection URI: {self.uri}")
            logger.error(f"Username: {self.username}")
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
        """Delete all nodes, relationships, constraints, and indexes from the database."""
        with self.driver.session() as session:
            # Step 1: Drop all constraints
            print("🗑️  Dropping all constraints...")
            constraints_query = "SHOW CONSTRAINTS"
            constraints = session.run(constraints_query)
            constraint_count = 0
            
            for constraint in constraints:
                constraint_name = constraint.get("name")
                if constraint_name:
                    drop_query = f"DROP CONSTRAINT {constraint_name} IF EXISTS"
                    session.run(drop_query)
                    constraint_count += 1
            
            print(f"   ✅ Dropped {constraint_count} constraints")
            
            # Step 2: Drop all indexes
            print("🗑️  Dropping all indexes...")
            indexes_query = "SHOW INDEXES"
            indexes = session.run(indexes_query)
            index_count = 0
            
            for index in indexes:
                index_name = index.get("name")
                # Skip constraint-backed indexes (already dropped with constraints)
                if index_name and index.get("type") != "RANGE":
                    try:
                        drop_query = f"DROP INDEX {index_name} IF EXISTS"
                        session.run(drop_query)
                        index_count += 1
                    except Exception:
                        pass  # Some indexes are auto-managed
            
            print(f"   ✅ Dropped {index_count} indexes")
            
            # Step 3: Delete all nodes and relationships
            count_query = """
            MATCH (n)
            RETURN count(n) as nodeCount
            """
            result = session.run(count_query)
            node_count = result.single()["nodeCount"]
            
            print(f"🗑️  Deleting {node_count} nodes from database...")
            
            # Delete in batches to avoid memory issues with large databases
            batch_size = 10000
            deleted_total = 0
            
            while True:
                delete_query = f"""
                MATCH (n)
                WITH n LIMIT {batch_size}
                DETACH DELETE n
                RETURN count(n) as deleted
                """
                result = session.run(delete_query)
                deleted = result.single()["deleted"]
                
                if deleted == 0:
                    break
                    
                deleted_total += deleted
                print(f"   ⏳ Deleted {deleted_total}/{node_count} nodes...")
            
            # Step 4: Verify deletion
            verify_query = "MATCH (n) RETURN count(n) as remaining"
            result = session.run(verify_query)
            remaining = result.single()["remaining"]
            
            if remaining == 0:
                print("✅ Database reset complete - all data, constraints, and indexes removed")
            else:
                print(f"⚠️  Warning: {remaining} nodes still remain in database")
    
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
            
            "CREATE CONSTRAINT sector_name_unique IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE"
            "CREATE CONSTRAINT region_name_unique IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE"
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
        
    def create_fund(self, fund: FundData):
        """
        Create a Fund and a time-specific Profile, linking all chunks to the Profile.
        PRESERVES HISTORY: Different report dates = Different profiles/risks.
        """
        try:
            # 1. VALIDATION & DEFAULTS
            if not fund.ticker:
                print(f"⚠️ Skipping fund with no ticker: {fund.name}")
                return None
            
            # Format Date for ID generation (e.g., "2024-09-30")
            date_str = fund.report_date.isoformat() if fund.report_date else "LATEST"
            
            # 2. GENERATE IDs (Time-Aware)
            # We append the date to IDs so we don't overwrite history
            profile_id = f"{fund.ticker}_{date_str}"
            obj_id = f"{fund.ticker}_{date_str}_obj"
            perf_id = f"{fund.ticker}_{date_str}_perf"

            print(f"📊 Creating Profile {date_str} for: {fund.ticker}")

            # 3. PREPARE LISTS FOR CYPHER (Time-Aware IDs)
            # We intentionally modify the chunk ID to include the date here
            risk_list_data = []
            if hasattr(fund, 'risks_chunks') and fund.risks_chunks:
                for chunk in fund.risks_chunks:
                    # Skip chunks without required fields
                    if not hasattr(chunk, 'id') or not hasattr(chunk, 'text'):
                        continue
                    risk_list_data.append({
                        # unique_id: "VTSAX_2024-09-30_risk_0"
                        "id": f"{getattr(chunk, 'id', 'unknown')}_{date_str}", 
                        "title": getattr(chunk, 'title', 'Untitled'),
                        "text": chunk.text,
                        "vector": getattr(chunk, 'embedding', None)
                    })

            strategy_list_data = []
            if hasattr(fund, 'strategies_chunks') and fund.strategies_chunks:
                for chunk in fund.strategies_chunks:
                    # Skip chunks without required fields
                    if not hasattr(chunk, 'id') or not hasattr(chunk, 'text'):
                        continue
                    strategy_list_data.append({
                        "id": f"{getattr(chunk, 'id', 'unknown')}_{date_str}", 
                        "title": getattr(chunk, 'title', 'Untitled'),
                        "text": chunk.text,
                        "vector": getattr(chunk, 'embedding', None)
                    })

            # 4. CYPHER QUERY
            query = """
            // --- A. STATIC FUND NODES (Merge ensures we reuse existing) ---
            MERGE (t:Trust {name: $trust})
            MERGE (p:Provider {name: $provider})
            MERGE (f:Fund {ticker: $ticker})
            MERGE (sc:ShareClass {name: $share_class})
            
            // Connect Static Relationships
            MERGE (t)-[:ISSUES]->(f)
            MERGE (p)-[:MANAGES]->(t)
            MERGE (f)-[:HAS_SHARE_CLASS]->(sc)
            
            // Update Static Properties (Name might change rarely)
            SET f.name = $name,
                f.securityExchange = $exchange,
                f.costsPer10k = $costs_per_10k,
                f.advisoryFees = $advisory_fees,
                f.numberHoldings = $n_holdings,
                f.expenseRatio = $expense_ratio,
                f.turnoverRate = $turnover_rate,
                f.netAssets = $net_assets,
                f.updatedAt = timestamp()

            // --- B. TEMPORAL PROFILE NODE (Create new if date differs) ---
            MERGE (prof:Profile {id: $profile_id})
            ON CREATE SET
                prof.summaryProspectus = $summary_prospectus,
                prof.createdAt = timestamp()
            ON MATCH SET
                prof.summaryProspectus = $summary_prospectus,
                prof.updatedAt = timestamp()
            
            // Connect Fund to Profile (History kept via 'date' property on rel)
            MERGE (f)-[rel:DEFINED_BY]->(prof)
            SET rel.date = $report_date

            // --- C. SINGLE NODES (Objective & Performance) ---
            // Connected to PROFILE, not Fund
            
            FOREACH (_ IN CASE WHEN $objective_text IS NOT NULL THEN [1] ELSE [] END |
                MERGE (obj:Objective {id: $obj_id})
                SET obj.text = $objective_text,
                    obj.embedding = $objective_vector
                MERGE (prof)-[:HAS_OBJECTIVE]->(obj)
            )

            FOREACH (_ IN CASE WHEN $perf_text IS NOT NULL THEN [1] ELSE [] END |
                MERGE (perf:PerformanceCommentary {id: $perf_id})
                SET perf.text = $perf_text,
                    perf.embedding = $perf_vector
                MERGE (prof)-[:HAS_PERFORMANCE_COMMENTARY]->(perf)
            )

            // --- D. CHUNK NODES (Risks & Strategies) ---
            // Connected to PROFILE

            WITH prof, f
            UNWIND $risk_list as r_item
            MERGE (rc:RiskChunk {id: r_item.id})
            SET rc.title = r_item.title,
                rc.text  = r_item.text,
                rc.embedding = r_item.vector
            MERGE (prof)-[:HAS_RISK]->(rc)

            WITH prof, f
            UNWIND $strat_list as s_item
            MERGE (sc:StrategyChunk {id: s_item.id})
            SET sc.title = s_item.title,
                sc.text  = s_item.text,
                sc.embedding = s_item.vector
            MERGE (prof)-[:HAS_STRATEGY]->(sc)
            
            RETURN f.ticker as ticker
            """

            # 5. PARAMETERS
            params = {
                # Static
                "ticker": fund.ticker,
                "name": fund.name or "Unknown",
                "trust": fund.registrant or "Unknown",
                "provider": fund.provider or "Unknown",
                "exchange": fund.security_exchange or "N/A",
                "share_class": fund.share_class or "N/A",
                "costs_per_10k": getattr(fund, 'costs_per_10k', 0),
                "advisory_fees": getattr(fund, 'advisory_fees', 0.0),
                "n_holdings": getattr(fund, 'n_holdings', 0),
                
                # Profile (Temporal)
                "profile_id": profile_id,
                "report_date": fund.report_date,
                "net_assets": fund.net_assets or 0.0,
                "expense_ratio": fund.expense_ratio or 0.0,
                "turnover_rate": fund.turnover_rate or 0.0,
                "summary_prospectus": getattr(fund, 'summary_prospectus', ""),
                
                # Single Nodes - Handle missing attributes safely
                "obj_id": obj_id,
                "objective_text": getattr(fund, 'objective', None) if getattr(fund, 'objective', None) not in [None, "N/A", ""] else None,
                "objective_vector": getattr(fund, 'objective_embedding', None),
                
                "perf_id": perf_id,
                "perf_text": getattr(fund, 'performance_commentary', None) if getattr(fund, 'performance_commentary', None) not in [None, "N/A", ""] else None,
                "perf_vector": getattr(fund, 'performance_commentary_embedding', None),
                
                # Lists - Already safe since we check fund.risks_chunks and fund.strategies_chunks above
                "risk_list": risk_list_data,
                "strat_list": strategy_list_data
            }

            with self.driver.session() as session:
                result = session.run(query, params)
                return result.single()

        except Exception as e:
            print(f"❌ Error creating fund {fund.ticker}: {e}")
            return None
    
    def create_indexes(self, dimensions=768):
        """
        Creates Indexes and Constraints, then VERIFIES they are online.
        """
        print("⚙️  Initializing Graph Schema & Indexes...")

        # 1. VECTOR INDEXES (For Similarity Search)
        vector_queries = [
            f"""
            CREATE VECTOR INDEX risk_vector_index IF NOT EXISTS
            FOR (n:Risk) ON (n.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX strategy_vector_index IF NOT EXISTS
            FOR (n:Strategy) ON (n.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX objective_vector_index IF NOT EXISTS
            FOR (n:Objective) ON (n.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX commentary_vector_index IF NOT EXISTS
            FOR (n:PerformanceCommentary) ON (n.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        ]
        
        # 2. FULLTEXT INDEXES (For Fuzzy Name Search)
        fulltext_queries = [
            """
            // 1. Index for Providers (e.g., "The Vanguard Group")
            CREATE FULLTEXT INDEX providerNameIndex IF NOT EXISTS
            FOR (p:Provider) ON EACH [p.name]
            """,
            """
            // 2. Index for Trusts (e.g., "Vanguard Index Funds")
            CREATE FULLTEXT INDEX trustNameIndex IF NOT EXISTS
            FOR (t:Trust) ON EACH [t.name]
            """,
            """
            // 3. Index for Funds (Name AND Ticker)
            // We include ticker here so 'VTI' and 'Total Stock' both work
            CREATE FULLTEXT INDEX fundNameIndex IF NOT EXISTS
            FOR (f:Fund) ON EACH [f.name, f.ticker]
            """,
            """
            // 4. Index for Managers (Person nodes)
            CREATE FULLTEXT INDEX personNameIndex IF NOT EXISTS
            FOR (p:Person) ON EACH [p.name]
            """
        ]

        # 2. CONSTRAINTS (Crucial for Ingestion Speed & Data Integrity)
        constraint_queries = [
            # Core Entities
            "CREATE CONSTRAINT fund_ticker_unique IF NOT EXISTS FOR (f:Fund) REQUIRE f.ticker IS UNIQUE",
            "CREATE CONSTRAINT share_class_unique IF NOT EXISTS FOR (s:ShareClass) REQUIRE s.ticker IS UNIQUE",
            
            # Temporal / Structure
            "CREATE CONSTRAINT profile_id_unique IF NOT EXISTS FOR (p:Profile) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT portfolio_id_unique IF NOT EXISTS FOR (p:Portfolio) REQUIRE p.id IS UNIQUE", # Added Portfolio
            
            # The "Performance Saver" for your 14k holdings
            "CREATE CONSTRAINT holding_id_unique IF NOT EXISTS FOR (h:Holding) REQUIRE h.id IS UNIQUE",
            
            # Text Chunks
            "CREATE CONSTRAINT section_risk_unique IF NOT EXISTS FOR (r:Risk) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT section_strat_unique IF NOT EXISTS FOR (s:Strategy) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT section_obj_unique IF NOT EXISTS FOR (o:Objective) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT section_comm_unique IF NOT EXISTS FOR (c:PerformanceCommentary) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT perf_id_unique IF NOT EXISTS FOR (tp:TrailingPerformance) REQUIRE tp.id IS UNIQUE"
        ]

        # 3. FULLTEXT INDEXES (For Keyword Search)
        fulltext_queries = [
            """
            // 1. Index for Providers (e.g., "The Vanguard Group")
            CREATE FULLTEXT INDEX providerNameIndex IF NOT EXISTS
            FOR (p:Provider) ON EACH [p.name]
            """,
            """
            // 2. Index for Trusts (e.g., "Vanguard Index Funds")
            CREATE FULLTEXT INDEX trustNameIndex IF NOT EXISTS
            FOR (t:Trust) ON EACH [t.name]
            """,
            """
            // 3. Index for Funds (Name AND Ticker)
            // We include ticker here so 'VTI' and 'Total Stock' both work
            CREATE FULLTEXT INDEX fundNameIndex IF NOT EXISTS
            FOR (f:Fund) ON EACH [f.name, f.ticker]
            """,
            """
            // 4. Index for Managers (Person nodes)
            CREATE FULLTEXT INDEX personNameIndex IF NOT EXISTS
            FOR (p:Person) ON EACH [p.name]
            """,
            """
            // 5. Index for Sections (Keyword Search)
            CREATE FULLTEXT INDEX section_keyword_index IF NOT EXISTS
            FOR (n:Section) ON EACH [n.text]
            """
        ]

        try:
            with self.driver.session() as session:
                # A. Run Creations
                print("   ... Applying Vector Indexes")
                for q in vector_queries: session.run(q)
                
                print("   ... Applying Unique Constraints")
                for q in constraint_queries: session.run(q)
                
                print("   ... Applying Fulltext Indexes")
                for q in fulltext_queries: session.run(q)

                # B. VERIFICATION STEP (The Debugger)
                print("\n🔎 VERIFYING INDEX STATUS:")
                print(f"{'INDEX NAME':<30} | {'TYPE':<15} | {'STATE':<10} | {'PROVIDER'}")
                print("-" * 75)

                result = session.run("SHOW INDEXES")
                
                # We count them to ensure they match expectations
                count = 0
                for record in result:
                    data = record.data()
                    name = data.get("name", "")
                    itype = data.get("type", "")
                    state = data.get("state", "")
                    provider = (
                        data.get("provider")
                        or data.get("indexProvider")
                        or data.get("indexProviderDescriptor")
                        or ""
                    )
                    
                    # Filter out internal Neo4j indexes (token lookups) for cleaner output
                    if "token" not in provider:
                        print(f"{name:<30} | {itype:<15} | {state:<10} | {provider}")
                        count += 1
                
                print("-" * 75)
                print(f"✅ Total User Indexes Found: {count}\n")
                
        except Exception as e:
            print(f"❌ Error managing indexes: {e}")
            import traceback
            traceback.print_exc()
    
    def add_managers(self, fund_ticker: str, managers: List[Dict[str, Any]]):
        """
        Add managers to a fund by creating Person nodes and MANAGED_BY relationships.
        
        Args:
            fund_ticker: The ticker of the fund to add managers to
            managers: List of manager dictionaries with keys:
                     - name: Manager name (required)
                     - role: Manager role (optional)
                     - since: Start date (optional)
        """
        try:
            print(f"👥 Adding {len(managers)} managers to fund {fund_ticker}")
            
            for manager in managers:
                if not manager:
                    print(f"⚠️ Skipping manager without name: {manager}")
                    continue
                
                # Create Person node and link to fund
                query = """
                MATCH (f:Fund {ticker: $fund_ticker})
                MERGE (p:Person {name: $manager_name})
                ON CREATE SET
                    p.createdAt = timestamp()
                ON MATCH SET
                    p.updatedAt = timestamp()
                MERGE (f)-[r:MANAGED_BY]->(p)
                SET r.createdAt = timestamp()
                RETURN p, f, r
                """
                
                params = {
                    "fund_ticker": fund_ticker,
                    "manager_name": manager,
                }
                
                result = self._execute_write(query, params)
                if result:
                    print(f"✅ Added manager: {manager} to {fund_ticker}")
                else:
                    print(f"⚠️ Failed to add manager: {manager}")
            
        except Exception as e:
            print(f"❌ Error adding managers to fund {fund_ticker}: {e}")
            logger.error(f"Failed to add managers to fund {fund_ticker}: {e}", exc_info=True)

    def add_financial_highlight(
        self,
        fund_ticker: str,
        year: int,
        turnover: float,
        expense_ratio: float,
        total_return: float,
        net_assets: float,
        net_assets_value_begining: float,
        net_assets_value_end: float,
        net_income_ratio: float
    ):
        """
        Add financial highlights to a fund for a specific year.
        
        Args:
            fund_ticker: The ticker of the fund
            year: The year for these financial highlights (e.g., 2024)
            turnover: Portfolio turnover rate (percentage)
            expense_ratio: Total expense ratio (percentage)
            total_return: Total return for the period (percentage)
            net_assets: Total net assets under management (in millions)
            net_assets_value_begining: Price of one share at period start
            net_assets_value_end: Price of one share at period end
            net_income_ratio: Net investment income ratio (percentage)
        
        Returns:
            The created FinancialHighlights node or None if failed
        """
        try:
            # Ensure year is an integer
            year = int(year)
            highlights_id = f"{fund_ticker}_{year}_highlights"
            
            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            MERGE (fh:FinancialHighlight {id: $highlights_id})
            ON CREATE SET
                fh.turnover = $turnover,
                fh.expenseRatio = $expense_ratio,
                fh.totalReturn = $total_return,
                fh.netAssets = $net_assets,
                fh.netAssetsValueBeginning = $net_assets_value_begining,
                fh.netAssetsValueEnd = $net_assets_value_end,
                fh.netIncomeRatio = $net_income_ratio,
                fh.createdAt = timestamp()
            ON MATCH SET
                fh.turnover = $turnover,
                fh.expenseRatio = $expense_ratio,
                fh.totalReturn = $total_return,
                fh.netAssets = $net_assets,
                fh.netAssetsValueBeginning = $net_assets_value_begining,
                fh.netAssetsValueEnd = $net_assets_value_end,
                fh.netIncomeRatio = $net_income_ratio,
                fh.updatedAt = timestamp()
            MERGE (f)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh)
            SET r.year = $year,
                r.createdAt = timestamp()
            RETURN fh, f, r
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "highlights_id": highlights_id,
                "year": year,
                "turnover": turnover,
                "expense_ratio": expense_ratio,
                "total_return": total_return,
                "net_assets": net_assets,
                "net_assets_value_begining": net_assets_value_begining,
                "net_assets_value_end": net_assets_value_end,
                "net_income_ratio": net_income_ratio,
            }
            
            result = self._execute_write(query, params)
            if result:
                print(f"✅ Added financial highlights for {fund_ticker} ({year})")
                return result[0]["fh"]
            else:
                print(f"⚠️ Fund {fund_ticker} not found")
                return None
                
        except Exception as e:
            print(f"❌ Error adding financial highlights to {fund_ticker}: {e}")
            logger.error(f"Failed to add financial highlights to {fund_ticker}: {e}", exc_info=True)
            return None

  

    def add_sector_allocations(
        self,
        sectors_df,
        fund_ticker: str,
        report_date: str
    ):
        """
        Add multiple sector allocations from a DataFrame.
        
        Args:
            sectors_df: DataFrame with 2 columns - first column is sector name, second is weight
            fund_ticker: The ticker of the fund
            report_date: The date of this allocation report (e.g., "2024-12-31")
        
        Returns:
            Number of sectors successfully added
        """
        try:
            if sectors_df is None or sectors_df.empty:
                print(f"⚠️ No sector data provided for {fund_ticker}")
                return 0
            
            cols = sectors_df.columns.tolist()
            if len(cols) < 2:
                print(f"❌ DataFrame must have at least 2 columns (sector name, weight)")
                return 0
            
            sector_col = cols[0]
            weight_col = cols[1]
            
            sectors_data = []
            for _, row in sectors_df.iterrows():
                sector_name = row[sector_col]
                weight = row[weight_col]
                
                if sector_name and weight is not None:
                    sectors_data.append({
                        "sector_name": str(sector_name).strip(),
                        "weight": float(weight)
                    })
            
            if not sectors_data:
                print(f"⚠️ No valid sector data found for {fund_ticker}")
                return 0
            
            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            UNWIND $sectors AS sector
            MERGE (s:Sector {name: sector.sector_name})
            ON CREATE SET
                s.createdAt = timestamp()
            MERGE (f)-[r:HAS_SECTOR_ALLOCATION]->(s)
            SET r.weight = sector.weight,
                r.reportDate = $report_date,
                r.updatedAt = timestamp()
            RETURN count(s) as count
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "sectors": sectors_data,
                "report_date": report_date,
            }
            
            result = self._execute_write(query, params)
            if result:
                count = result[0]["count"]
                print(f"✅ Added {count} sector allocations to {fund_ticker} ({report_date})")
                return count
            else:
                print(f"⚠️ Fund {fund_ticker} not found")
                return 0
                
        except Exception as e:
            print(f"❌ Error adding sector allocations to {fund_ticker}: {e}")
            logger.error(f"Failed to add sector allocations to {fund_ticker}: {e}", exc_info=True)
            return 0

    def add_geographic_allocations(
        self,
        geographic_df,
        fund_ticker: str,
        report_date: str
    ):
        """
        Add multiple geographic allocations from a DataFrame.
        
        Args:
            geographic_df: DataFrame with 2 columns - first column is region name, second is weight
            fund_ticker: The ticker of the fund
            report_date: The date of this allocation report (e.g., "2024-12-31")
        
        Returns:
            Number of regions successfully added
        """
        try:
            if geographic_df is None or geographic_df.empty:
                print(f"⚠️ No geographic data provided for {fund_ticker}")
                return 0
            
            cols = geographic_df.columns.tolist()
            if len(cols) < 2:
                print(f"❌ DataFrame must have at least 2 columns (region name, weight)")
                return 0
            
            region_col = cols[0]
            weight_col = cols[1]
            
            regions_data = []
            for _, row in geographic_df.iterrows():
                region_name = row[region_col]
                weight = row[weight_col]
                
                if region_name and weight is not None:
                    regions_data.append({
                        "region_name": str(region_name).strip(),
                        "weight": float(weight)
                    })
            
            if not regions_data:
                print(f"⚠️ No valid geographic data found for {fund_ticker}")
                return 0
            
            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            UNWIND $regions AS region
            MERGE (r:Region {name: region.region_name})
            ON CREATE SET
                r.createdAt = timestamp()
            MERGE (f)-[rel:HAS_GEOGRAPHIC_ALLOCATION]->(r)
            SET rel.weight = region.weight,
                rel.reportDate = $report_date,
                rel.updatedAt = timestamp()
            RETURN count(r) as count
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "regions": regions_data,
                "report_date": report_date,
            }
            
            result = self._execute_write(query, params)
            if result:
                count = result[0]["count"]
                print(f"✅ Added {count} geographic allocations to {fund_ticker} ({report_date})")
                return count
            else:
                print(f"⚠️ Fund {fund_ticker} not found")
                return 0
                
        except Exception as e:
            print(f"❌ Error adding geographic allocations to {fund_ticker}: {e}")
            logger.error(f"Failed to add geographic allocations to {fund_ticker}: {e}", exc_info=True)
            return 0
 

    def create_fund_holdings(self, ticker:str, series_id: str, report_date: Optional[date], holdings_df):
        """
        High-Performance Ingestion for Large Funds (>4k holdings).
        Uses Batching + CREATE logic for maximum speed.
        """
        if series_id is None:
            print(f"⚠️ No series_id provided for {ticker}, skipping portfolio and holdings creation")
            return
        
        if holdings_df is None or holdings_df.empty:
            print(f"⚠️ No holdings found for {series_id}")
            return

        print(f"💰 Processing {len(holdings_df)} holdings for {series_id}...")

        # 1. CLEANING & ID GENERATION (Python Side)
        
        df = holdings_df.replace({np.nan: None})
        
        # Ensure numerics
        for col in ['shares', 'market_value', 'weight_pct']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: float(x) if x is not None else 0.0)

        # ID Generation using ISIN only
        def generate_id(row):
            return str(row.get('isin', 'UNKNOWN')).strip()

        df['node_id'] = df.apply(generate_id, axis=1)
        
        # Convert to list
        all_holdings = df.to_dict('records')
        total_count = len(all_holdings)

        # 2. SETUP PARENT NODES (One-time setup for the Fund/Portfolio)
        # We do this OUTSIDE the batch loop so we don't repeat it
        
        portfolio_id = series_id
        
        setup_query = """
        MATCH (fund:Fund {ticker: $fund_ticker})
        MERGE (port:Portfolio {id: $portfolio_id})
        ON CREATE SET
            port.ticker = $fund_ticker,
            port.count = $total_count,
            port.createdAt = timestamp()
        MERGE (fund)-[r:HAS_PORTFOLIO]->(port)
        ON CREATE SET
            r.date = $report_date,
            r.createdAt = timestamp()
        ON MATCH SET
            r.date = $report_date,
            r.updatedAt = timestamp()
        """
        self._execute_write(setup_query, {
            "fund_ticker": ticker, 
            "portfolio_id": portfolio_id, 
            "report_date": report_date,
            "total_count": total_count
        })

        # 3. BATCHED INSERTION (The Speed Fix)
        # We process holdings in chunks of 2,000 to prevent memory overload
        BATCH_SIZE = 1000
        
        batch_query = """
        MATCH (port:Portfolio {id: $portfolio_id})
        UNWIND $batch as row
        
        // A. Handle the Company Node (MERGE is fine here - it's shared)
        MERGE (h:Holding {id: row.node_id})
        ON CREATE SET
            h.name = row.name,
            h.ticker = row.ticker,
            h.cusip = row.cusip,
            h.isin = row.isin,
            h.lei = row.lei,
            h.country = row.country,
            h.sector = row.sector,
            h.assetCategory = row.asset_category,
            h.assetDesc = row.asset_category_desc,
            h.issuerCategory = row.issuer_category,
            h.issuerDesc = row.issuer_category_desc,
            h.createdAt = timestamp()

        // B. Create Link (Use CREATE, not MERGE for speed)
        // Since 'port' is unique to this report, we know links don't exist yet.
        MERGE (port)-[rel:CONTAINS]->(h)
        ON CREATE SET
            rel.shares = row.shares,
            rel.marketValue = row.market_value,
            rel.weight = row.weight_pct,
            rel.currency = row.currency,
            rel.fairValueLevel = row.fair_value_level,
            rel.isRestricted = row.is_restricted,
            rel.payoffProfile = row.payoff_profile
        """

        try:
            with self.driver.session() as session:
                for i in range(0, total_count, BATCH_SIZE):
                    batch = all_holdings[i : i + BATCH_SIZE]
                    session.run(batch_query, {
                        "portfolio_id": portfolio_id,
                        "batch": batch
                    })
                    print(f"   ⏳ {ticker}: Ingested batch {i} to {i + len(batch)}...")
            
            print(f"✅ Successfully ingested {total_count} holdings for {ticker}")

        except Exception as e:
            print(f"❌ Error ingesting holdings for {ticker}: {e}")

    
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
    
    # ==================== NODE CREATION ====================
    
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
        performance_data: Optional[AverageReturnSnapshot] = None,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create TrailingPerformance node and link to Fund."""
        if performance_data is None:
            return None
        
        perf_props = {}
        if performance_data:
                    
            if hasattr(performance_data, 'return_1y') and performance_data.return_1y is not None:
                perf_props["return1y"] = performance_data.return_1y
            if hasattr(performance_data, 'return_5y') and performance_data.return_5y is not None:
                perf_props["return5y"] = performance_data.return_5y
            if hasattr(performance_data, 'return_10y') and performance_data.return_10y is not None:
                perf_props["return10y"] = performance_data.return_10y
            if hasattr(performance_data, 'return_inception') and performance_data.return_inception is not None:
                perf_props["returnInception"] = performance_data.return_inception
        
        perf_id = fund_ticker + "_" + date if date else fund_ticker
        
        # Build SET clause only if there are properties to set
        perf_set = ", ".join([f"tp.{k} = ${k}" for k in perf_props.keys()])
        set_clause = f"SET {perf_set}" if perf_set else ""
        
        query = f"""
        MATCH (f:Fund {{ticker: $fund_ticker}})
        MERGE (tp:TrailingPerformance {{id: $perf_id}})
        {set_clause}
        MERGE (f)-[r:HAS_TRAILING_PERFORMANCE {{date: $date}}]->(tp)
        RETURN tp
        """
        
        params = {"fund_ticker": fund_ticker, "perf_id": perf_id}
        params.update(perf_props)
        if date:
            params["date"] = date
        
        result = self._execute_write(query, params)
        if result:
            logger.info(f"Created performance node for fund {fund_ticker}")
            print(result)
            return result[0]["tp"]
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
    
    def query(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.
        
        Args:
            cypher_query: The Cypher query string to execute
            parameters: Optional dictionary of parameters for the query
            
        Returns:
            List of result records as dictionaries
            
        Example:
            # Get fund by ticker
            results = db.query(
                "MATCH (f:Fund {ticker: $ticker}) RETURN f",
                {"ticker": "VTI"}
            )
            
            # Get all funds
            results = db.query("MATCH (f:Fund) RETURN f ORDER BY f.ticker")
            
            # Complex query with multiple nodes
            results = db.query('''
                MATCH (f:Fund {ticker: $ticker})-[r:HAS_ASSET]->(a:Asset)
                RETURN f, collect({asset: a, weight: r.weight}) as holdings
            ''', {"ticker": "VTI"})
        """
        return self._execute_query(cypher_query, parameters)
    
    def clear_database(self):
        """Clear entire database (use with caution!)."""
        query = "MATCH (n) DETACH DELETE n"
        self._execute_write(query)
        logger.warning("Database cleared!")
    
    def add_chart_to_fund(
        self, 
        fund_ticker: str, 
        title: str, 
        category: str, 
        svg_content: str, 
        date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Add a chart (Image node) to a fund with HAS_CHART relationship.
        
        Args:
            fund_ticker: The ticker of the fund to add the chart to
            title: Title of the chart
            category: Category/type of the chart (e.g., 'performance', 'allocation', 'sector')
            svg_content: The SVG content as a string
            date: Date associated with the chart (e.g., report date)
            
        Returns:
            Dictionary with created Image node or None if fund not found
        """
        try:
            # Generate unique ID for the image using hash of content
            import hashlib
            content_hash = hashlib.md5(svg_content.encode()).hexdigest()[:12]
            image_id = f"{fund_ticker}_{category}_{content_hash}"
            
            query = """
            MATCH (f:Fund {ticker: $fund_ticker})
            MERGE (img:Image {id: $image_id})
            ON CREATE SET
                img.title = $title,
                img.category = $category,
                img.svg = $svg_content,
                img.createdAt = timestamp()
            ON MATCH SET
                img.title = $title,
                img.category = $category,
                img.svg = $svg_content,
                img.updatedAt = timestamp()
            MERGE (f)-[r:HAS_CHART]->(img)
            RETURN img, f
            """
            
            params = {
                "fund_ticker": fund_ticker,
                "image_id": image_id,
                "title": title,
                "category": category,
                "svg_content": svg_content
            }
            
            result = self._execute_write(query, params)
            
            if result:
                print(f"✅ Added chart '{title}' ({category}) to fund {fund_ticker}")
                return result[0]
            else:
                print(f"⚠️ Fund {fund_ticker} not found")
                return None
                
        except Exception as e:
            print(f"❌ Error adding chart to fund {fund_ticker}: {e}")
            logger.error(f"Failed to add chart to fund {fund_ticker}: {e}", exc_info=True)
            return None
    
    def get_fund_charts(self, fund_ticker: str) -> List[Dict[str, Any]]:
        """
        Get all charts for a specific fund.
        
        Args:
            fund_ticker: The ticker of the fund
            
        Returns:
            List of chart dictionaries with image data and relationship date
        """
        query = """
        MATCH (f:Fund {ticker: $fund_ticker})-[r:HAS_CHART]->(img:Image)
        RETURN img.id as id,
               img.title as title,
               img.category as category,
               img.svg as svg_content,
               img.createdAt as created_at
        """
        
        result = self._execute_query(query, {"fund_ticker": fund_ticker})
        return result
