"""
Main Streamlit application for SEC Filings Intelligence Assistant.

Routes between a pipeline configuration page and the chat interface.
When configured, queries go through the real RAG pipeline with streaming.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# Add parent directories to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from src.simple_rag.streamlit_app import config
from src.simple_rag.streamlit_app.components import chat_interface, sidebar, config_page
from src.simple_rag.streamlit_app.pipeline_bridge import (
    process_query as pipeline_process_query,
    shutdown_pipeline,
    verify_connection,
    PipelineQueryResult,
)
from src.simple_rag.rag.answer_generation.result_classifier import ResultType


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
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "pipeline_configured" not in st.session_state:
        st.session_state.pipeline_configured = False
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def render_header():
    """Render the application header with pipeline status."""
    pipeline = st.session_state.get("pipeline")
    if pipeline:
        cfg = pipeline.config
        status_text = (
            f"PIPELINE READY &middot; {cfg.cypher_backend.upper()} / {cfg.answer_provider_name.upper()}"
        )
        pill_class = "status-pill"
    else:
        status_text = "PIPELINE NOT CONFIGURED"
        pill_class = "status-pill status-pill-config"

    st.markdown(
        f"""
        <div class="main-header">
            <span class="header-bottom-left"></span>
            <span class="header-bottom-right"></span>
            <h1>&#9670; SEC Filings Intelligence</h1>
            <div class="subtitle">KNOWLEDGE GRAPH &middot; RAG PIPELINE &middot; EDGAR DATA</div>
            <div class="{pill_class}">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Mock response fallback ───────────────────────────────────────────────────

def generate_mock_response(query: str) -> dict:
    """Generate a mock response for development/testing."""
    query_lower = query.lower()

    if "apple" in query_lower or "aapl" in query_lower:
        return {
            "response": (
                "Based on Apple's latest 10-K filing (FY2024), here are the key revenue segments:\n\n"
                "**Product Revenue:** iPhone $200.6B, Mac $29.4B, iPad $28.3B, Wearables $37.0B\n\n"
                "**Services Revenue:** $85.2B (21.6% of total)\n\n"
                "Total net sales: **$394.3 billion** (+2% YoY)"
            ),
            "sources": [],
        }
    elif "vtsax" in query_lower or "fund" in query_lower or "holding" in query_lower:
        return {
            "response": (
                "**VTSAX Top Holdings:** AAPL (6.8%), MSFT (6.2%), NVDA (4.9%), AMZN (3.8%), GOOGL (2.1%)\n\n"
                "Total Net Assets: $663.4B | Holdings: 3,647 | Expense Ratio: 0.04%"
            ),
            "sources": [],
        }
    else:
        return {
            "response": (
                f'This is a **mock response** for: "{query}"\n\n'
                "The real RAG pipeline is not configured. "
                "Please configure it from the setup page to get real answers."
            ),
            "sources": [],
        }


# ── Query processing ─────────────────────────────────────────────────────────

def handle_real_query(query: str) -> None:
    """Process a query through the real RAG pipeline with thinking UI and streaming."""
    pipeline = st.session_state.pipeline

    # Verify Neo4j connection
    if not verify_connection(pipeline):
        st.error("Neo4j connection lost. Please reconfigure the pipeline.")
        st.session_state.pipeline_configured = False
        st.session_state.pipeline = None
        st.rerun()
        return

    verbose = pipeline.config.verbose

    # Phase 1: Thinking UI — run pipeline steps
    with st.status("Querying knowledge graph...", expanded=True) as status:
        try:
            steps, result = pipeline_process_query(query, pipeline)
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Pipeline error: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error processing query: {e}",
            })
            return

        # Display step details
        for step in steps:
            st.write(f"**{step.step.title()}:** {step.detail} ({step.elapsed:.2f}s)")

        if verbose and result.cypher:
            st.code(result.cypher, language="cypher")

        if result.error:
            status.update(label="No results", state="error")
            st.warning(result.error)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.error,
            })
            return

        status.update(label="Analysis complete", state="complete", expanded=False)

    # Phase 2: Stream the answer
    full_response = st.write_stream(result.token_stream)

    # Phase 3: Rich results (charts, tables)
    if result.charts:
        for chart in result.charts:
            with st.expander(f"Chart: {chart.get('title', 'Visualization')}"):
                st.html(chart.get("svg", ""))

    if result.tabular and len(result.tabular) <= 50:
        with st.expander(f"Data Table ({len(result.tabular)} rows)"):
            st.dataframe(pd.DataFrame(result.tabular), use_container_width=True)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "metadata": {
            "category": result.category,
            "confidence": result.confidence,
            "cypher": result.cypher,
            "result_type": result.result_type.value,
        },
    })


def handle_mock_query(query: str) -> None:
    """Process a query with mock responses (fallback)."""
    response_data = generate_mock_response(query)
    st.markdown(response_data["response"])

    if response_data.get("sources"):
        with st.expander(f"Sources ({len(response_data['sources'])})"):
            chat_interface.display_sources(response_data["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_data["response"],
        "sources": response_data.get("sources", []),
    })


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_css()
    initialize_session_state()

    # Route: config page or chat interface
    if not st.session_state.pipeline_configured:
        config_page.render_config_page()
        return

    # ── Chat interface ───────────────────────────────────────────────────────
    render_header()
    filters = sidebar.render_sidebar()

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history
    chat_interface.display_messages(st.session_state.messages)

    # Chat input
    if prompt := st.chat_input("Ask about SEC filings, funds, or companies..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            pipeline = st.session_state.get("pipeline")
            if pipeline and not config.MOCK_MODE:
                handle_real_query(prompt)
            else:
                handle_mock_query(prompt)

        st.session_state.query_count += 1

    st.markdown("</div>", unsafe_allow_html=True)

    # Query suggestions — empty state
    if len(st.session_state.messages) == 0:
        st.markdown(
            '<div class="suggestion-label">&mdash; SUGGESTED QUERIES &mdash;</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, example in enumerate(config.SAMPLE_QUERIES[:6]):
            with cols[i % 3]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.rerun()


if __name__ == "__main__":
    main()
