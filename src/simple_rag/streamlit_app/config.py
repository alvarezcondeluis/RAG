"""
Configuration for Streamlit chatbot application.
"""

import os
from pathlib import Path

# Application Settings
APP_TITLE = "SEC Filings Intelligence"
APP_SUBTITLE = "KNOWLEDGE GRAPH · RAG PIPELINE · EDGAR DATA"
APP_ICON = "◆"

# Styling
COLORS = {
    "primary": "#1a2332",      # Deep navy
    "secondary": "#2c3e50",    # Slate blue
    "accent": "#d4af37",       # Gold
    "success": "#27ae60",      # Green (positive metrics)
    "warning": "#f39c12",      # Amber (caution)
    "danger": "#e74c3c",       # Red (negative metrics)
    "background": "#f8f9fa",   # Light gray
    "card": "#ffffff",         # White
    "text_primary": "#2c3e50", # Dark gray
    "text_secondary": "#7f8c8d" # Medium gray
}

# LLM Settings (Groq API)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "mixtral-8x7b-32768"  # Fast and capable
GROQ_TEMPERATURE = 0.1  # Low temperature for factual responses
GROQ_MAX_TOKENS = 2048

# Mock mode (for development without LLM)
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Neo4j Settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# RAG Settings
TOP_K_RESULTS = 5  # Number of documents to retrieve
HYBRID_SEARCH_WEIGHT = 0.7  # Weight for semantic vs graph search

# Query Examples
SAMPLE_QUERIES = [
    "Apple FY2024 revenue breakdown",
    "VTSAX top 10 holdings",
    "MSFT key risk factors",
    "Tesla CEO compensation 2024",
    "NVDA insider trading activity",
    "Compare AAPL vs MSFT margins",
]

# Paths
BASE_DIR = Path(__file__).parent
STYLES_DIR = BASE_DIR / "styles"
