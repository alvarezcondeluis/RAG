"""
Sidebar component — Bloomberg terminal aesthetic.
Filters, search settings, session management, and database status.
"""

import streamlit as st
from typing import Dict, Any


def render_sidebar() -> Dict[str, Any]:
    """
    Render sidebar with filters and settings.
    
    Returns:
        Dictionary of filter values
    """
    with st.sidebar:
        # Header
        st.markdown(
            '<div class="sidebar-header">SEC INTELLIGENCE</div>',
            unsafe_allow_html=True,
        )

        # Data source filters
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

        # Conversation management
        st.markdown("### Session")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.query_count = 0
                st.rerun()
        with col2:
            if st.button("💾 Export", use_container_width=True):
                _export_conversation()

        st.divider()

        # Session statistics as metric cards
        st.markdown("### Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.get("query_count", 0))
        with col2:
            st.metric("Messages", len(st.session_state.get("messages", [])))

        st.divider()

        # Mock mode indicator
        from src.simple_rag.streamlit_app import config

        if config.MOCK_MODE or not config.GROQ_API_KEY:
            st.caption("⚠ MOCK MODE ACTIVE")

        # Footer
        st.caption("SEC Filings Intelligence v0.1.0")

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
        label="📥 Download",
        data=json.dumps(export_data, indent=2),
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )
