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
        Creates uniqueness constraints aligned with the canonical knowledge-graph schema.
        Run this ONCE before ingesting data. Idempotent (IF NOT EXISTS).
        """
        queries = [
            # --- 1. CORE FUND DOMAIN ---
            "CREATE CONSTRAINT fund_ticker_unique IF NOT EXISTS FOR (f:Fund) REQUIRE f.ticker IS UNIQUE",
            "CREATE CONSTRAINT trust_name_unique IF NOT EXISTS FOR (t:Trust) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT provider_name_unique IF NOT EXISTS FOR (p:Provider) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT share_class_name_unique IF NOT EXISTS FOR (s:ShareClass) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT portfolio_series_unique IF NOT EXISTS FOR (p:Portfolio) REQUIRE p.seriesId IS UNIQUE",

            # --- 2. ALLOCATIONS / GROUPING ---
            "CREATE CONSTRAINT sector_name_unique IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT region_name_unique IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE",

            # --- 3. ORGANIZATIONS & PEOPLE ---
            "CREATE CONSTRAINT company_ticker_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
            "CREATE CONSTRAINT person_name_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",

            # --- 4. SOURCE DOCUMENTS ---
            # Single source of truth — every SEC filing dedupes here.
            "CREATE CONSTRAINT document_accession_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.accessionNumber IS UNIQUE",

            # --- 5. SECURITIES ---
            # Holdings are global per ISIN (the same security held by multiple
            # portfolios shares one node). Neo4j unique constraints ignore NULL,
            # so holdings without an ISIN are still allowed.
            "CREATE CONSTRAINT holding_isin_unique IF NOT EXISTS FOR (h:Holding) REQUIRE h.isin IS UNIQUE",

            # --- 6. CHARTS ---
            # holdings_ops.add_chart_to_fund builds id = "{ticker}_{category}_{md5(svg)}".
            "CREATE CONSTRAINT image_id_unique IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE",
        ]

        with self.driver.session() as session:
            print("🚧 Setting up constraints...")
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    print(f"⚠️ Warning: {e}")
            print("✅ Constraints configured.")
    
    def create_indexes(self, dimensions=768):
        """
        Creates indexes (vector, range, fulltext, relationship) aligned with the
        canonical schema. Run after create_constraints(). Idempotent.
        """
        print("⚙️  Initializing Graph Schema & Indexes...")

        # 1. VECTOR INDEXES (Similarity Search)
        # Profile/Filing10K chunks live on (:Chunk {embedding}).
        # Section:Objective and Section:BusinessInformation store embedding
        # directly on the Section node per the schema.
        vector_queries = [
            f"""
            CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
            FOR (n:Chunk) ON (n.embedding)
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
            """,
        ]

        # 2. RANGE / LOOKUP INDEXES 
        range_queries = [
            # Funds — name fallback when ticker is N/A
            "CREATE INDEX fund_name_index IF NOT EXISTS FOR (f:Fund) ON (f.name)",

            # Documents — temporal queries
            "CREATE INDEX document_filing_date_index IF NOT EXISTS FOR (d:Document) ON (d.filingDate)",
            "CREATE INDEX document_reporting_date_index IF NOT EXISTS FOR (d:Document) ON (d.reportingDate)",
            "CREATE INDEX document_type_index IF NOT EXISTS FOR (d:Document) ON (d.type)",

            # Holdings — lookup by security identifier
            "CREATE INDEX holding_ticker_index IF NOT EXISTS FOR (h:Holding) ON (h.ticker)",
            "CREATE INDEX holding_lei_index IF NOT EXISTS FOR (h:Holding) ON (h.lei)",

            # Sections — sectionType is used to MATCH the right Section subtype
            "CREATE INDEX section_section_type_index IF NOT EXISTS FOR (s:Section) ON (s.sectionType)",
            "CREATE INDEX section_title_index IF NOT EXISTS FOR (s:Section) ON (s.title)",

            # Companies — CIK is the SEC identifier
            "CREATE INDEX company_cik_index IF NOT EXISTS FOR (c:Company) ON (c.cik)",

            # Portfolios — temporal queries
            "CREATE INDEX portfolio_date_index IF NOT EXISTS FOR (p:Portfolio) ON (p.date)",
        ]

        # 3. RELATIONSHIP-PROPERTY INDEXES (Year-keyed traversal performance)
        # Most fund-side queries filter by year on these relationships.
        relationship_queries = [
            "CREATE INDEX rel_reports_in_year IF NOT EXISTS FOR ()-[r:REPORTS_IN]-() ON (r.year)",
            "CREATE INDEX rel_defined_by_year IF NOT EXISTS FOR ()-[r:DEFINED_BY]-() ON (r.year)",
            "CREATE INDEX rel_managed_by_year IF NOT EXISTS FOR ()-[r:MANAGED_BY]-() ON (r.year)",
            "CREATE INDEX rel_has_financial_highlight_year IF NOT EXISTS FOR ()-[r:HAS_FINANCIAL_HIGHLIGHT]-() ON (r.year)",
            "CREATE INDEX rel_has_sector_allocation_year IF NOT EXISTS FOR ()-[r:HAS_SECTOR_ALLOCATION]-() ON (r.year)",
            "CREATE INDEX rel_has_region_allocation_year IF NOT EXISTS FOR ()-[r:HAS_REGION_ALLOCATION]-() ON (r.year)",
            "CREATE INDEX rel_has_average_returns_year IF NOT EXISTS FOR ()-[r:HAS_AVERAGE_RETURNS]-() ON (r.year)",
            "CREATE INDEX rel_has_chart_year IF NOT EXISTS FOR ()-[r:HAS_CHART]-() ON (r.year)",
            "CREATE INDEX rel_has_table_year IF NOT EXISTS FOR ()-[r:HAS_TABLE]-() ON (r.year)",
        ]

        # 4. FULLTEXT INDEXES (Keyword Search)
        fulltext_queries = [
            """
            CREATE FULLTEXT INDEX providerNameIndex IF NOT EXISTS
            FOR (p:Provider) ON EACH [p.name]
            """,
            """
            CREATE FULLTEXT INDEX trustNameIndex IF NOT EXISTS
            FOR (t:Trust) ON EACH [t.name]
            """,
            """
            CREATE FULLTEXT INDEX fundNameIndex IF NOT EXISTS
            FOR (f:Fund) ON EACH [f.name, f.ticker]
            """,
            """
            CREATE FULLTEXT INDEX personNameIndex IF NOT EXISTS
            FOR (p:Person) ON EACH [p.name]
            """,
            """
            CREATE FULLTEXT INDEX companyNameIndex IF NOT EXISTS
            FOR (c:Company) ON EACH [c.name, c.ticker]
            """,
            """
            CREATE FULLTEXT INDEX holdingNameIndex IF NOT EXISTS
            FOR (h:Holding) ON EACH [h.name, h.ticker]
            """,
            """
            CREATE FULLTEXT INDEX section_keyword_index IF NOT EXISTS
            FOR (n:Section) ON EACH [n.text, n.title]
            """,
            """
            CREATE FULLTEXT INDEX chunk_keyword_index IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text, c.title]
            """,
        ]

        try:
            with self.driver.session() as session:
                print("   ... Applying Vector Indexes")
                for q in vector_queries: session.run(q)

                print("   ... Applying Range/Lookup Indexes")
                for q in range_queries: session.run(q)

                print("   ... Applying Relationship-Property Indexes")
                for q in relationship_queries: session.run(q)

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
