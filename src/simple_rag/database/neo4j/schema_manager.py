from typing import Dict
import logging
from .base import Neo4jDatabaseBase

logger = logging.getLogger(__name__)


class SchemaManager(Neo4jDatabaseBase):
    """
    Manages Neo4j database schema including constraints, indexes, and database operations.
    Inherits connection management from Neo4jDatabaseBase.
    """
    
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
            
            "CREATE CONSTRAINT sector_name_unique IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT region_name_unique IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE",
            # Managers should be unique nodes
            "CREATE CONSTRAINT manager_name_unique IF NOT EXISTS FOR (m:Manager) REQUIRE m.name IS UNIQUE",
            
            # --- 3. COMPANY DATA ENTITIES ---
            
            # Company nodes - unique by ticker
            "CREATE CONSTRAINT company_ticker_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
            
            # Document nodes - unique by accession number
            "CREATE CONSTRAINT document_accession_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.accesionNumber IS UNIQUE",
            
            # 10KFiling nodes - unique by id
            "CREATE CONSTRAINT filing_10k_id_unique IF NOT EXISTS FOR (f:`10KFiling`) REQUIRE f.id IS UNIQUE",
            
            # Person nodes - unique by name
            "CREATE CONSTRAINT person_name_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            
            # CompensationPackage nodes - unique by id
            "CREATE CONSTRAINT compensation_package_id_unique IF NOT EXISTS FOR (cp:CompensationPackage) REQUIRE cp.id IS UNIQUE",
            
            # InsiderTransaction nodes - unique by id
            "CREATE CONSTRAINT insider_transaction_id_unique IF NOT EXISTS FOR (it:InsiderTransaction) REQUIRE it.id IS UNIQUE",
            
            # FinancialMetric nodes - unique by id
            "CREATE CONSTRAINT financial_metric_id_unique IF NOT EXISTS FOR (fm:FinancialMetric) REQUIRE fm.id IS UNIQUE",
            
            # Segment nodes - unique by id
            "CREATE CONSTRAINT segment_id_unique IF NOT EXISTS FOR (seg:Segment) REQUIRE seg.id IS UNIQUE",
            
            # Section nodes - unique by id (for all section types)
            "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
        ]

        with self.driver.session() as session:
            print("🚧 Setting up constraints...")
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    print(f"⚠️ Warning: {e}")
            print("✅ Constraints and Indexes configured.")
    
    def create_indexes(self, dimensions=768):
        """
        Creates Indexes and Constraints, then VERIFIES they are online.
        """
        print("⚙️  Initializing Graph Schema & Indexes...")

        # 1. VECTOR INDEXES (For Similarity Search)
        vector_queries = [
            f"""
            CREATE VECTOR INDEX risk_vector_index IF NOT EXISTS
            FOR (n:RiskChunk) ON (n.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX strategy_vector_index IF NOT EXISTS
            FOR (n:StrategyChunk) ON (n.embedding)
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
            """,
            f"""
            CREATE VECTOR INDEX section_vector_index IF NOT EXISTS
            FOR (n:Section) ON (n.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        ]
        
        # 2. CONSTRAINTS (Crucial for Ingestion Speed & Data Integrity)
        constraint_queries = [
            # Core Entities
            "CREATE CONSTRAINT fund_ticker_unique IF NOT EXISTS FOR (f:Fund) REQUIRE f.ticker IS UNIQUE",
            "CREATE CONSTRAINT share_class_unique IF NOT EXISTS FOR (s:ShareClass) REQUIRE s.ticker IS UNIQUE",
            
            # Temporal / Structure
            "CREATE CONSTRAINT profile_id_unique IF NOT EXISTS FOR (p:Profile) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT portfolio_id_unique IF NOT EXISTS FOR (p:Portfolio) REQUIRE p.id IS UNIQUE",
            
            # The "Performance Saver" for your 14k holdings
            "CREATE CONSTRAINT holding_id_unique IF NOT EXISTS FOR (h:Holding) REQUIRE h.id IS UNIQUE",
            
            # Text Chunks
            "CREATE CONSTRAINT risk_chunk_unique IF NOT EXISTS FOR (r:RiskChunk) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT strategy_chunk_unique IF NOT EXISTS FOR (s:StrategyChunk) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT objective_unique IF NOT EXISTS FOR (o:Objective) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT commentary_unique IF NOT EXISTS FOR (c:PerformanceCommentary) REQUIRE c.id IS UNIQUE",
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
            // 5. Index for Companies (Name AND Ticker)
            CREATE FULLTEXT INDEX companyNameIndex IF NOT EXISTS
            FOR (c:Company) ON EACH [c.name, c.ticker]
            """,
            """
            // 6. Index for Sections (Keyword Search)
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
