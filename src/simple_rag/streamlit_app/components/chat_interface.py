"""
Chat interface components — Bloomberg terminal aesthetic.
Handles message display, source citations with filing badges, and loading states.
"""

import streamlit as st
from typing import List, Dict, Any


def _filing_badge_class(filing_type: str) -> str:
    """Map filing type string to a CSS badge class."""
    t = filing_type.upper().replace(" ", "").replace("-", "")
    if "10K" in t: return "badge-10k"
    if "8K"  in t: return "badge-8k"
    if "DEF14A" in t or "DEF" in t: return "badge-def14a"
    if "FORM4" in t or "4" == t: return "badge-form4"
    return "badge-default"


def _relevance_class(score: float) -> str:
    """Return CSS class based on relevance score tier."""
    if score >= 0.90: return "relevance-high"
    if score >= 0.75: return "relevance-medium"
    return "relevance-low"


def display_messages(messages: List[Dict[str, Any]]):
    """
    Display chat message history.
    
    Args:
        messages: List of message dictionaries with role, content, and optional sources
    """
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.markdown(content)
            
            # Display sources if available (for assistant messages)
            if role == "assistant" and message.get("sources"):
                with st.expander(f"◆ Sources ({len(message['sources'])})"):
                    display_sources(message["sources"])


def display_sources(sources: List[Dict[str, Any]]):
    """
    Display source citations with colour-coded filing badges and relevance tiers.
    
    Args:
        sources: List of source dictionaries with metadata
    """
    for i, source in enumerate(sources, 1):
        relevance = source.get("relevance", 0)
        rel_cls = _relevance_class(relevance)
        badge_cls = _filing_badge_class(source.get("type", ""))

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-header">
                    <span class="source-title">{i}. {source.get('title', 'Untitled')}</span>
                    <span class="source-relevance {rel_cls}">{relevance:.0%}</span>
                </div>
                <div class="source-metadata">
                    <span class="badge-filing {badge_cls}">{source.get('type', 'DOC')}</span>
                    {source.get('company', 'N/A')}
                    <span style="color:var(--text-3)">·</span>
                    {source.get('ticker', '')}
                    <span style="color:var(--text-3)">·</span>
                    {source.get('date', '')}
                </div>
                <div class="source-snippet">
                    "{source.get('snippet', 'No preview available')}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # SEC EDGAR link
        url = source.get("url", "#")
        if url and url != "#":
            st.markdown(f"[↗ View on SEC EDGAR]({url})")


def display_loading():
    """Display a spinner-ring loading indicator."""
    st.markdown(
        """
        <div class="loading-ring">
            <div class="ring"></div>
            <span>Querying knowledge graph…</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
