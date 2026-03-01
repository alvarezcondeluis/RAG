"""
Main Streamlit application for SEC Filings Intelligence Assistant.

A professional chatbot interface for querying SEC filings and fund data
using RAG with Neo4j knowledge graph integration.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.simple_rag.streamlit_app import config
from src.simple_rag.streamlit_app.components import chat_interface, sidebar
from src.simple_rag.streamlit_app.utils import session_manager
from src.simple_rag.database.neo4j import Neo4jDatabase


def load_css():
    """Load custom CSS styling."""
    css_file = config.STYLES_DIR / "main.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "db" not in st.session_state:
        try:
            st.session_state.db = Neo4jDatabase(auto_start=False)
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {e}")
            st.session_state.db = None
    
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0


def render_header():
    """Render the application header with bracket-corner decorations."""
    db_connected = st.session_state.get("db") is not None
    status_text = "SYSTEM READY · NEO4J CONNECTED" if db_connected else "NEO4J DISCONNECTED"
    
    st.markdown(
        f"""
        <div class="main-header">
            <span class="header-bottom-left"></span>
            <span class="header-bottom-right"></span>
            <h1>◆ SEC Filings Intelligence</h1>
            <div class="subtitle">KNOWLEDGE GRAPH · RAG PIPELINE · EDGAR DATA</div>
            <div class="status-pill">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def generate_mock_response(query: str) -> dict:
    """
    Generate a mock response for development/testing.
    
    Args:
        query: User query
        
    Returns:
        Dictionary with response and sources
    """
    # Simple keyword-based mock responses
    query_lower = query.lower()
    
    if "apple" in query_lower or "aapl" in query_lower:
        return {
            "response": """Based on Apple's latest 10-K filing (FY2024), here are the key revenue segments:

**Product Revenue Breakdown:**
- iPhone: $200.6B (50.9% of total revenue)
- Mac: $29.4B (7.5%)
- iPad: $28.3B (7.2%)
- Wearables, Home & Accessories: $37.0B (9.4%)

**Services Revenue:**
- Services: $85.2B (21.6% of total revenue)

**Geographic Breakdown:**
- Americas: $167.0B (42.4%)
- Europe: $101.3B (25.7%)
- Greater China: $67.8B (17.2%)
- Japan: $25.0B (6.3%)
- Rest of Asia Pacific: $33.2B (8.4%)

Total net sales for FY2024 reached $394.3 billion, representing a 2% increase year-over-year.""",
            "sources": [
                {
                    "title": "Apple Inc. - 10-K Annual Report (FY2024)",
                    "type": "10-K",
                    "date": "2024-11-01",
                    "company": "Apple Inc.",
                    "ticker": "AAPL",
                    "relevance": 0.95,
                    "snippet": "Net sales by category: iPhone $200,583M, Mac $29,357M, iPad $28,300M...",
                    "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193"
                },
                {
                    "title": "Apple Inc. - Business Segment Information",
                    "type": "10-K Section",
                    "date": "2024-11-01",
                    "company": "Apple Inc.",
                    "ticker": "AAPL",
                    "relevance": 0.89,
                    "snippet": "The Company manages its business primarily on a geographic basis...",
                    "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193"
                }
            ]
        }
    
    elif "risk" in query_lower:
        return {
            "response": """Key risk factors identified in recent SEC filings include:

**Market & Competition Risks:**
- Intense competition in technology markets
- Rapid technological changes requiring continuous innovation
- Dependence on third-party suppliers and manufacturers

**Operational Risks:**
- Global supply chain disruptions
- Cybersecurity threats and data privacy concerns
- Dependence on key personnel

**Financial Risks:**
- Foreign exchange rate fluctuations
- Changes in tax laws and regulations
- Economic downturns affecting consumer spending

**Regulatory Risks:**
- Increasing government regulation of technology companies
- Antitrust investigations and litigation
- Data protection and privacy regulations (GDPR, CCPA)

These factors could materially affect business operations and financial results.""",
            "sources": [
                {
                    "title": "Risk Factors - Item 1A",
                    "type": "10-K Section",
                    "date": "2024-11-01",
                    "company": "Technology Company",
                    "ticker": "TECH",
                    "relevance": 0.92,
                    "snippet": "We face intense competition in the markets in which we operate...",
                    "url": "#"
                }
            ]
        }
    
    elif "vtsax" in query_lower or "fund" in query_lower or "holding" in query_lower:
        return {
            "response": """**Vanguard Total Stock Market Index Fund (VTSAX) - Top Holdings:**

1. **Apple Inc. (AAPL)** - 6.8% of portfolio
   - Market Value: $45.2B
   - Shares: 412.3M

2. **Microsoft Corp. (MSFT)** - 6.2% of portfolio
   - Market Value: $41.1B
   - Shares: 98.7M

3. **NVIDIA Corp. (NVDA)** - 4.9% of portfolio
   - Market Value: $32.5B
   - Shares: 245.1M

4. **Amazon.com Inc. (AMZN)** - 3.8% of portfolio
   - Market Value: $25.2B
   - Shares: 125.4M

5. **Alphabet Inc. Class A (GOOGL)** - 2.1% of portfolio
   - Market Value: $13.9B
   - Shares: 95.2M

**Fund Statistics:**
- Total Net Assets: $663.4B
- Number of Holdings: 3,647
- Expense Ratio: 0.04%
- Turnover Rate: 4%""",
            "sources": [
                {
                    "title": "VTSAX - Portfolio Holdings",
                    "type": "N-CSR Filing",
                    "date": "2024-12-31",
                    "company": "Vanguard Total Stock Market Index Fund",
                    "ticker": "VTSAX",
                    "relevance": 0.97,
                    "snippet": "Schedule of Investments as of December 31, 2024...",
                    "url": "#"
                }
            ]
        }
    
    else:
        # Generic response
        return {
            "response": f"""I understand you're asking about: "{query}"

This is a mock response for demonstration purposes. In production, this would:

1. **Retrieve relevant documents** from the Neo4j knowledge graph
2. **Perform semantic search** on SEC filings (10-K, DEF 14A, Form 4)
3. **Generate a response** using Groq API with retrieved context
4. **Cite sources** with links to original SEC EDGAR filings

**Sample capabilities:**
- Financial analysis (revenue, metrics, segments)
- Risk factor summaries
- Fund holdings and allocations
- Executive compensation analysis
- Insider trading activity
- Cross-domain queries (funds → companies → filings)

Try asking about specific companies (Apple, Microsoft) or funds (VTSAX)!""",
            "sources": [
                {
                    "title": "Sample Document",
                    "type": "Mock",
                    "date": "2024-01-01",
                    "company": "Example Corp",
                    "ticker": "EXMP",
                    "relevance": 0.75,
                    "snippet": "This is a mock source citation for demonstration...",
                    "url": "#"
                }
            ]
        }


def process_query(query: str, filters: dict) -> dict:
    """
    Process user query and generate response.
    
    Args:
        query: User query string
        filters: Dictionary of filters from sidebar
        
    Returns:
        Dictionary with response and sources
    """
    # For now, use mock responses
    # TODO: Integrate actual RAG pipeline with Groq API
    
    if config.MOCK_MODE or not config.GROQ_API_KEY:
        return generate_mock_response(query)
    
    # Future: Real RAG pipeline
    # from src.simple_rag.streamlit_app.utils.rag_pipeline import RAGPipeline
    # rag = RAGPipeline(st.session_state.db, groq_client)
    # return rag.process_query(query, filters)
    
    return generate_mock_response(query)


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar
    filters = sidebar.render_sidebar()
    
    # Main chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    chat_interface.display_messages(st.session_state.messages)
    
    # Chat input
    if prompt := st.chat_input("Ask about SEC filings, funds, or companies..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing SEC filings..."):
                response_data = process_query(prompt, filters)
                
                # Display response
                st.markdown(response_data["response"])
                
                # Display sources
                if response_data.get("sources"):
                    with st.expander(f"📚 Sources ({len(response_data['sources'])})"):
                        chat_interface.display_sources(response_data["sources"])
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["response"],
            "sources": response_data.get("sources", [])
        })
        
        # Increment query count
        st.session_state.query_count += 1
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Query suggestions — empty state
    if len(st.session_state.messages) == 0:
        st.markdown('<div class="suggestion-label">— SUGGESTED QUERIES —</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, example in enumerate(config.SAMPLE_QUERIES[:6]):
            with cols[i % 3]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.rerun()


if __name__ == "__main__":
    main()
