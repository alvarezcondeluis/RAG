"""
Main Streamlit application for SEC Filings Intelligence Assistant.

Routes between a pipeline configuration page and the chat interface.
When configured, queries go through the real RAG pipeline with streaming.
"""

import streamlit as st
from pathlib import Path
import sys
import time
import pandas as pd

# Add parent directories to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from src.simple_rag.streamlit_app import config
from src.simple_rag.streamlit_app.components import chat_interface, sidebar, config_page
from src.simple_rag.streamlit_app.pipeline_bridge import (
    process_query as pipeline_process_query,
    verify_connection,
)


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
    if "terms_accepted" not in st.session_state:
        st.session_state.terms_accepted = False
    if "pending_example" not in st.session_state:
        st.session_state.pending_example = None


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
        <div class="main-header" style="overflow:visible;">
            <span class="header-bottom-left"></span>
            <span class="header-bottom-right"></span>
            <h1 style="line-height:0.4; padding-bottom:0.15em;">&#9670; SEC Filings Intelligence</h1>
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
        if verbose and result.answer_messages:
            with st.expander("Answer prompt (system + user)", expanded=False):
                for msg in result.answer_messages:
                    st.markdown(f"**`{msg['role'].upper()}`**")
                    st.text(msg["content"])

        if result.error:
            if result.error.startswith("WRITE_BLOCKED:"):
                op = result.error[len("WRITE_BLOCKED:"):].strip()
                status.update(label="Security violation blocked", state="error")
                st.markdown(
                    f"""
                    <div style="border:2px solid #ff4b4b;border-radius:8px;padding:1rem 1.25rem;
                                background:#2a0a0a;color:#ff6b6b;font-family:monospace;">
                        <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.5rem;">
                            🚨 DESTRUCTIVE OPERATION BLOCKED — <code>{op}</code>
                        </div>
                        <div style="color:#ffaaaa;line-height:1.6;">
                            This system operates in <strong>READ-ONLY</strong> mode.<br>
                            Your request generated a <strong>{op}</strong> operation against the database,
                            which has been permanently blocked.<br><br>
                            Write operations (<code>CREATE</code>, <code>DELETE</code>, <code>MERGE</code>,
                            <code>SET</code>, <code>REMOVE</code>, <code>DROP</code>) are never permitted.<br>
                            This incident has been logged.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                user_message = f"Destructive operation ({op}) blocked. This system is read-only."
            elif result.error.startswith("NOT_RELATED:"):
                status.update(label="Out of scope", state="error")
                user_message = result.error[len("NOT_RELATED:"):].strip()
                st.markdown(
                    f'<div class="not-related-msg">&#9888; {user_message}</div>',
                    unsafe_allow_html=True,
                )
            else:
                status.update(label="No results", state="error")
                st.warning(result.error)
                user_message = result.error
            st.session_state.messages.append({
                "role": "assistant",
                "content": user_message,
            })
            return

        status.update(label="Analysis complete", state="complete", expanded=False)

    # Phase 2: Stream the answer
    # Escape bare $ so Streamlit's markdown renderer doesn't treat them as LaTeX delimiters.
    # \$ in CommonMark renders as a literal $, so stored content is correct for history replay.
    def _escape_dollars(stream):
        for chunk in stream:
            yield chunk.replace("$", r"\$")

    t_gen = time.time()
    full_response = st.write_stream(_escape_dollars(result.token_stream))
    gen_time = time.time() - t_gen
    dispatch_time = steps[0].elapsed if steps else 0.0

    # Discrete timing line
    st.markdown(
        f'<div class="response-timing">'
        f'⏱ translation&nbsp;{dispatch_time:.1f}s'
        f'&ensp;·&ensp;'
        f'generation&nbsp;{gen_time:.1f}s'
        f'</div>',
        unsafe_allow_html=True,
    )

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
            "dispatch_time": dispatch_time,
            "gen_time": gen_time,
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

    # Route: terms acceptance → config → chat
    if not st.session_state.terms_accepted:
        _render_terms()
        return

    if not st.session_state.pipeline_configured:
        config_page.render_config_page()
        return

    # ── Chat interface ───────────────────────────────────────────────────────
    render_header()
    filters = sidebar.render_sidebar()

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history
    chat_interface.display_messages(st.session_state.messages)

    # Reserve the examples slot NOW so we can clear it synchronously before
    # the pipeline blocks — prevents stale examples showing during LLM calls.
    examples_slot = st.empty()

    # Resolve active prompt: typed input takes priority, then a clicked example
    active_prompt = None
    if prompt := st.chat_input("Ask about SEC filings, funds, or companies..."):
        active_prompt = prompt
    elif st.session_state.pending_example:
        active_prompt = st.session_state.pending_example
        st.session_state.pending_example = None

    if active_prompt:
        examples_slot.empty()  # clear examples immediately, before the blocking LLM call
        st.session_state.messages.append({
            "role": "user",
            "content": active_prompt,
        })
        with st.chat_message("user"):
            st.markdown(active_prompt)
        with st.chat_message("assistant"):
            pipeline = st.session_state.get("pipeline")
            if pipeline and not config.MOCK_MODE:
                handle_real_query(active_prompt)
            else:
                handle_mock_query(active_prompt)
        st.session_state.query_count += 1

    st.markdown("</div>", unsafe_allow_html=True)

    # Query suggestions — only when no messages and nothing is being processed
    if len(st.session_state.messages) == 0 and not active_prompt:
        with examples_slot:
            st.markdown(
                '<div class="suggestion-label">&mdash; SUGGESTED QUERIES &mdash;</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(3)
            for i, example in enumerate(config.SAMPLE_QUERIES[:6]):
                with cols[i % 3]:
                    if st.button(example, key=f"example_{i}", use_container_width=True):
                        st.session_state.pending_example = example
                        st.rerun()

    if len(st.session_state.messages) > 0:
        _render_footer()


def _render_terms():
    """Full-page terms and disclaimer acceptance screen."""
    st.markdown(
        """
        <div class="terms-container">
            <div class="terms-title">&#9670; Terms of Use &amp; Disclaimer</div>
            <div class="terms-body">
                <p>By using <strong>SEC Filings Intelligence</strong> you acknowledge and agree to the following:</p>
                <ul>
                    <li><strong>Not investment advice.</strong> All content generated by this tool is for
                    informational and research purposes only. Nothing here constitutes a recommendation
                    to buy, sell, or hold any security.</li>
                    <li><strong>AI-generated content.</strong> Responses are produced by large language
                    models and may contain inaccuracies, omissions, or hallucinations. Always verify
                    against official primary sources.</li>
                    <li><strong>Data sources.</strong> Financial data is retrieved from public
                    <strong>SEC EDGAR</strong> filings. This tool is not affiliated with or endorsed
                    by the U.S. Securities and Exchange Commission.</li>
                    <li><strong>No liability.</strong> The authors assume no responsibility for decisions
                    made based on the output of this tool.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, col2, _ = st.columns([2, 1, 2])
    with col2:
        if st.button("I Understand — Continue", type="primary", use_container_width=True):
            st.session_state.terms_accepted = True
            st.rerun()


def _render_footer():
    """Render attribution and copyright footer."""
    st.markdown(
        """
        <div class="app-footer">
            <span>Data sourced from <strong>SEC EDGAR</strong> — U.S. Securities and Exchange Commission
            &nbsp;&middot;&nbsp; Not affiliated with or endorsed by the SEC</span>
            <span class="footer-copyright">&copy; 2026 SEC Filings Intelligence. All rights reserved.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
