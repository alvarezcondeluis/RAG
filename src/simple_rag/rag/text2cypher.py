import requests
import json
import subprocess
import time
from typing import Optional


class CypherTranslator:
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        api_url: str = "http://localhost:11434/api/generate",
        auto_start_ollama: bool = True
    ):
        """
        Wraps the local Ollama LLM for text-to-Cypher translation.
        
        Args:
            model_name: Ollama model to use
            api_url: Ollama API endpoint
            auto_start_ollama: Whether to auto-start Ollama if not running
        """
        self.model_name = model_name
        self.api_url = api_url
        self.ollama_process = None
        
        # Complete Neo4j schema
        self.schema = """
        # Fund Management Structure
        (:Provider {name})-[:MANAGES]->(:Trust {name})-[:ISSUES]->(:Fund {
            ticker, name, securityExchange, costsPer10k, advisoryFees, numberHoldings, 
            expenseRatio, netAssets, turnoverRate, createdAt, updatedAt
        })
        
        # Fund Share Classes
        (:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name})
        
        # Fund Profile and Content Sections (Temporal - versioned by date)
        (:Fund)-[:DEFINED_BY {date}]->(:Profile {
            id, summaryProspectus, createdAt, updatedAt
        })
        (:Profile)-[:HAS_OBJECTIVE]->(:Objective {id, text, embedding})
        (:Profile)-[:HAS_PERFORMANCE]->(:PerformanceCommentary {id, text, embedding})
        (:Profile)-[:HAS_RISK]->(:RiskChunk {id, title, text, embedding})
        (:Profile)-[:HAS_STRATEGY]->(:StrategyChunk {id, title, text, embedding})
        
        # Fund Financial Highlights (by year)
        (:Fund)-[:HAS_FINANCIAL_HIGHLIGHT {year}]->(:FinancialHighlight {
            id, turnover, expenseRatio, totalReturn, netAssets, 
            netAssetsValueBeginning, netAssetsValueEnd, netIncomeRatio, createdAt, updatedAt
        })
        
        # Fund Charts/Images
        (:Fund)-[:HAS_CHART {date}]->(:Image {id, title, category, svg, createdAt, updatedAt})
        
        # Fund Management Team
        (:Fund)-[:MANAGED_BY]->(:Person {name, createdAt, updatedAt})
        
        # Fund Allocations (by report date)
        (:Fund)-[:HAS_SECTOR_ALLOCATION {weight, reportDate}]->(:Sector {name, createdAt})
        (:Fund)-[:HAS_GEOGRAPHIC_ALLOCATION {weight, reportDate}]->(:Region {name, createdAt})
        
        # Fund Holdings Structure
        (:Fund)-[:HAS_PORTFOLIO]->(:Portfolio {id, ticker, date, count, createdAt})
        (:Portfolio)-[:CONTAINS {shares, marketValue, weight, currency, fairValueLevel, isRestricted, payoffProfile}]->(:Holding {
            id, name, ticker, cusip, isin, lei, country, sector, assetCategory, 
            assetDesc, issuerCategory, issuerDesc, createdAt
        })
        
        # Note: Vector indexes exist on embedding properties for semantic search:
        # - RiskChunk.embedding, StrategyChunk.embedding, Objective.embedding, PerformanceCommentary.embedding
        # Use vector similarity search for finding similar content across funds.

        # Match ticker for symbols (e.g., 'VTI') and name for titles (e.g., 'Vanguard Total Stock').
        """
        
        # Check and start Ollama if needed
        if auto_start_ollama:
            self._ensure_ollama_running()
    
    def _is_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def _start_ollama_server(self) -> bool:
        """Start Ollama server in a subprocess."""
        try:
            print("Starting Ollama server...")
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Wait for server to be ready (max 30 seconds)
            max_wait = 30
            wait_interval = 1
            elapsed = 0
            
            while elapsed < max_wait:
                if self._is_ollama_running():
                    print(f"✓ Ollama server started successfully (took {elapsed}s)")
                    return True
                time.sleep(wait_interval)
                elapsed += wait_interval
            
            print("⚠ Ollama server started but not responding yet")
            return False
            
        except FileNotFoundError:
            print("✗ Ollama command not found. Please install Ollama first.")
            print("  Visit: https://ollama.ai/download")
            return False
        except Exception as e:
            print(f"✗ Failed to start Ollama server: {e}")
            return False
    
    def _ensure_ollama_running(self) -> bool:
        """Ensure Ollama server is running, start it if not."""
        if self._is_ollama_running():
            print("✓ Ollama server is already running")
            return True
        
        print("Ollama server not detected, attempting to start...")
        return self._start_ollama_server()
    
    def translate(self, user_query: str, temperature: float = 0.1) -> Optional[str]:
        """
        Translate natural language query to Cypher query.
        
        Args:
            user_query: Natural language question
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Cypher query string or None if error
        """
        prompt = f"""You are a Neo4j Cypher expert. Convert the natural language question to a valid Cypher query.

Neo4j Schema:
{self.schema}

Rules:
1. Output ONLY the Cypher query, no explanations or markdown
2. Use proper Cypher syntax with MATCH, WHERE, RETURN
3. Use property names exactly as shown in schema
4. For numeric comparisons, use appropriate operators (>, <, =, etc.)
5. For text search, use CONTAINS or regular expressions
6. Always return relevant properties

Question: {user_query}

Cypher Query:"""
        
        try:
            # Call Ollama API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 512
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            
            # Extract and clean the Cypher query
            raw_cypher = response_json.get("response", "").strip()
            
            # Remove markdown code blocks if present
            cleaned_cypher = raw_cypher.replace("```cypher", "").replace("```", "").strip()
            
            # Remove any leading/trailing quotes
            cleaned_cypher = cleaned_cypher.strip('"\'')
            
            if not cleaned_cypher:
                print("⚠ Empty Cypher query generated")
                return None
            
            print(f"✓ Generated Cypher: {cleaned_cypher[:100]}...")
            return cleaned_cypher
            
        except requests.exceptions.Timeout:
            print("✗ Ollama API timeout - query took too long")
            return None
        except requests.exceptions.RequestException as e:
            print(f"✗ Ollama API error: {e}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error in translation: {e}")
            return None
    
    def stop_ollama_server(self):
        """Stop the Ollama server if it was started by this instance."""
        if self.ollama_process:
            print("Stopping Ollama server...")
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=5)
                print("✓ Ollama server stopped")
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
                print("✓ Ollama server force stopped")
            self.ollama_process = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_ollama_server()