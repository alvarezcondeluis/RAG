"""
Sidebar component — Bloomberg terminal aesthetic.
Filters, pipeline info, session management, and database status.
"""

import streamlit as st
from typing import Dict, Any


def render_sidebar() -> Dict[str, Any]:
    """
    Render sidebar with filters, pipeline status, and settings.

    Returns:
        Dictionary of filter values.
    """
    with st.sidebar:
        # Header
        st.markdown(
            '<div class="sidebar-header">SEC INTELLIGENCE</div>',
            unsafe_allow_html=True,
        )

        # ── Pipeline info ────────────────────────────────────────────────────
        pipeline = st.session_state.get("pipeline")
        if pipeline:
            cfg = pipeline.config
            st.markdown("### Pipeline")
            server = f" ({cfg.openai_compatible_host}:{cfg.openai_compatible_port})" if cfg.cypher_backend == "openai" else ""
            st.markdown(
                f"""<div class="pipeline-info">
                <strong>Text2Cypher:</strong> {cfg.cypher_backend}{server} / {cfg.cypher_model}<br>
                <strong>Answer LLM:</strong> {cfg.answer_provider_name} / {cfg.answer_model}<br>
                <strong>Schema:</strong> {'ON' if cfg.use_schema_injection else 'OFF'}
                &nbsp;&middot;&nbsp;
                <strong>Entity Res:</strong> {'ON' if cfg.enable_entity_resolution else 'OFF'}
                &nbsp;&middot;&nbsp;
                <strong>Few-Shot:</strong> {'ON' if cfg.enable_few_shot else 'OFF'}<br>
                <strong>Retry:</strong> {'ON' if cfg.retry_module else 'OFF'} ({cfg.retry_strategy})
                &nbsp;&middot;&nbsp;
                <strong>VecEmbed:</strong> {'ON' if cfg.embed_vector_queries else 'OFF'}
                </div>""",
                unsafe_allow_html=True,
            )

            if st.button("Reconfigure", use_container_width=True, key="_reconfigure_btn"):
                from src.simple_rag.streamlit_app.pipeline_bridge import shutdown_pipeline
                shutdown_pipeline(pipeline)
                st.session_state.pipeline_configured = False
                st.session_state.pipeline = None
                st.rerun()

            st.divider()

        # ── Data source filters ──────────────────────────────────────────────
        st.markdown("### Filters")

        companies = st.multiselect(
            "Companies",
            options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
            help="Filter by company ticker",
        )

        funds = st.multiselect(
            "Funds",
            options=["VTSAX", "VTI", "VOO", "VFIAX", "VEXMX"],
            help="Filter by fund ticker",
        )

        st.divider()

        st.markdown("### Search Mode")

        search_mode = st.selectbox(
            "Strategy",
            options=["Hybrid", "Semantic", "Graph"],
            index=0,
            help="Choose retrieval strategy",
            label_visibility="collapsed",
        )

        st.divider()

        # ── Conversation management ──────────────────────────────────────────
        st.markdown("### Session")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear", use_container_width=True, key="_clear_btn"):
                st.session_state.messages = []
                st.session_state.query_count = 0
                st.rerun()
        with col2:
            if st.button("Export", use_container_width=True, key="_export_btn"):
                _export_conversation()

        st.divider()

        # ── Session statistics ───────────────────────────────────────────────
        st.markdown("### Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.get("query_count", 0))
        with col2:
            st.metric("Messages", len(st.session_state.get("messages", [])))

        st.divider()

        # ── Status indicator ─────────────────────────────────────────────────
        from src.simple_rag.streamlit_app import config

        if pipeline:
            st.caption("PIPELINE ACTIVE")
        elif config.MOCK_MODE:
            st.caption("MOCK MODE ACTIVE")

        st.caption("SEC Filings Intelligence v0.2.0")

    return {
        "companies": companies,
        "funds": funds,
        "search_mode": search_mode,
    }


def _export_conversation():
    """Export conversation history to JSON."""
    import json
    from datetime import datetime

    messages = st.session_state.get("messages", [])
    if not messages:
        st.warning("No conversation to export")
        return

    export_data = {
        "exported_at": datetime.now().isoformat(),
        "query_count": st.session_state.get("query_count", 0),
        "messages": messages,
    }

    st.download_button(
        label="Download",
        data=json.dumps(export_data, indent=2, default=str),
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )
