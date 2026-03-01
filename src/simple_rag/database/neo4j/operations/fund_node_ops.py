"""
Fund node creation operations for Neo4j database.
Includes risks, performance, strategy, profile, and derivative node creation.
"""

from typing import Optional, List, Dict, Any
from datetime import date
import logging
from ..base import Neo4jDatabaseBase
from src.simple_rag.models.fund import AverageReturnSnapshot

logger = logging.getLogger(__name__)


class FundNodeOperations(Neo4jDatabaseBase):
    """Fund-related node creation operations (risks, performance, strategy, profile, derivative)."""
    
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
        CREATE (f)-[r:HAS_RISK_NODE]->(risks)
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
        date: Optional[date] = None,
        accession_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create TrailingPerformance node and link to Fund.
        
        Args:
            fund_ticker: Fund ticker symbol
            performance_data: Performance metrics snapshot
            date: Date object for the performance data
            accession_number: Optional SEC accession number to link to source Document
        
        Returns:
            Created TrailingPerformance node or None
        """
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
        
        # Convert date object to string for Neo4j storage
        date_str = date.isoformat() if date else None
        perf_id = fund_ticker + "_" + date_str if date_str else fund_ticker
        
        # Extract year from date for easy filtering
        year = None
        if date:
            year = date.year
        
        # Build SET clause for performance properties
        perf_set_parts = [f"tp.{k} = ${k}" for k in perf_props.keys()]
        
        # Add year property if available
        if year:
            perf_set_parts.append("tp.year = $year")
        
        perf_set = ", ".join(perf_set_parts)
        set_clause = f"SET {perf_set}" if perf_set else ""
        
        # Build query with optional Document linking
        if accession_number:
            query = f"""
            MATCH (f:Fund {{ticker: $fund_ticker}})
            MERGE (tp:TrailingPerformance {{id: $perf_id}})
            {set_clause}
            MERGE (f)-[r:HAS_TRAILING_PERFORMANCE {{date: $date}}]->(tp)
            
            // Link to source Document if accession number provided
            WITH tp
            MATCH (d:Document {{id: $accession_number}})
            MERGE (tp)-[:EXTRACTED_FROM]->(d)
            
            RETURN tp
            """
        else:
            query = f"""
            MATCH (f:Fund {{ticker: $fund_ticker}})
            MERGE (tp:TrailingPerformance {{id: $perf_id}})
            {set_clause}
            MERGE (f)-[r:HAS_TRAILING_PERFORMANCE {{date: $date}}]->(tp)
            RETURN tp
            """
        
        params = {"fund_ticker": fund_ticker, "perf_id": perf_id}
        params.update(perf_props)
        if date_str:
            params["date"] = date_str
        if year:
            params["year"] = year
        if accession_number:
            params["accession_number"] = accession_number
        
        result = self._execute_write(query, params)
        if result:
            logger.info(f"Created performance node for fund {fund_ticker} (year: {year})")
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
