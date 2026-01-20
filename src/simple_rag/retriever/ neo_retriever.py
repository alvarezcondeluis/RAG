from neo4j import GraphDatabase
from typing import List, Dict, Optional

class FundRetriever:
    def __init__(self, driver, embedding_model):
        """
        Args:
            driver: Neo4j driver instance.
            embedding_model: Object with an .embed_query(text) method.
        """
        self.driver = driver
        self.model = embedding_model

    def vector_search_section(self, query: str, section_type: str, top_k: int = 5) -> List[Dict]:
        """
        Performs semantic search on a SPECIFIC section type (Risk, Strategy, Objective, Commentary).
        
        Args:
            section_type: 'Risk', 'Strategy', 'Objective', or 'PerformanceCommentary'
        """
        # 1. Generate Embedding
        query_vector = self.model.embed_query(query)

        # 2. Select the correct index based on type
        index_map = {
            "Risk": "risk_vector_index",
            "Strategy": "strategy_vector_index",
            "Objective": "objective_vector_index",
            "PerformanceCommentary": "commentary_vector_index"
        }
        
        index_name = index_map.get(section_type)
        
        if not index_name:
            raise ValueError(f"Unknown section type: {section_type}")

        # 3. Cypher Query
        # Note: We handle the different path for Commentary vs Profile Sections
        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $vector)
        YIELD node, score
        
        // Match back to Fund based on node labels
        OPTIONAL MATCH (node)<-[:HAS_{section_type.upper()}]-(parent)
        
        // Logic: If parent is Fund (Commentary), use it. 
        // If parent is Profile (Risk/Strat), find the Fund connected to it.
        OPTIONAL MATCH (parent)<-[:DESCRIBED_BY]-(f:Fund)
        WITH node, score, COALESCE(f, parent) as fund 
        // ^ If 'parent' is already the Fund (for Commentary), use it.

        RETURN 
            fund.ticker as ticker, 
            fund.name as name, 
            node.text as text, 
            score
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, index_name=index_name, vector=query_vector, k=top_k)
            return [record.data() for record in result]

    def hybrid_search(self, query: str, min_score: float = 0.8) -> List[Dict]:
        """
        Advanced: Searches across ALL sections and aggregates results.
        Good for vague user queries like "Safe tech funds".
        """
        query_vector = self.model.embed_query(query)
        
        cypher = """
        // 1. Search Strategies
        CALL db.index.vector.queryNodes('strategy_vector_index', 5, $vector)
        YIELD node as strat, score as s_score
        MATCH (strat)<-[:HAS_STRATEGY]-(p)<-[:DESCRIBED_BY]-(f:Fund)
        
        UNION
        
        // 2. Search Risks
        CALL db.index.vector.queryNodes('risk_vector_index', 5, $vector)
        YIELD node as risk, score as r_score
        MATCH (risk)<-[:HAS_RISK]-(p)<-[:DESCRIBED_BY]-(f:Fund)

        UNION
        
        // 3. Search Commentary (Directly connected)
        CALL db.index.vector.queryNodes('commentary_vector_index', 5, $vector)
        YIELD node as comm, score as c_score
        MATCH (comm)<-[:HAS_COMMENTARY]-(f:Fund)

        RETURN f.name as fund, f.ticker as ticker, max(score) as best_match
        ORDER BY best_match DESC
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, vector=query_vector)
            return [record.data() for record in result]

    def keyword_search(self, keyword: str) -> List[Dict]:
        """
        Uses Fulltext Index (BM25) for exact keyword matches.
        """
        cypher = """
        CALL db.index.fulltext.queryNodes("section_keyword_index", $keyword)
        YIELD node, score
        MATCH (node)<--(parent)
        OPTIONAL MATCH (parent)<-[:DESCRIBED_BY]-(f:Fund)
        RETURN COALESCE(f.name, parent.name) as fund, node.text as content, score
        """
        with self.driver.session() as session:
            result = session.run(cypher, keyword=keyword)
            return [record.data() for record in result]
            
    def structured_filter(self, min_assets: int = 0, max_fee: float = 1.0):
        """
        Standard SQL-style filtering (No AI).
        """
        cypher = """
        MATCH (f:Fund)
        WHERE f.net_assets >= $assets AND f.expense_ratio <= $fee
        RETURN f.ticker, f.name, f.expense_ratio
        ORDER BY f.net_assets DESC LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(cypher, assets=min_assets, fee=max_fee)
            return [record.data() for record in result]