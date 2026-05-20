"""
Configuration page — replaces the terminal-based setup_pipeline() with Streamlit widgets.
"""

import streamlit as st

from simple_rag.rag.orchestrator import PipelineConfig, TEXT2CYPHER_BACKENDS
from simple_rag.streamlit_app.pipeline_bridge import (
    get_available_providers,
    list_models_for_provider,
    init_pipeline,
)


def _fetch_models(provider_name: str, cache_key: str) -> list:
    """Fetch and cache model list for a provider."""
    cache = st.session_state.setdefault("_models_cache", {})
    if provider_name not in cache:
        cache[provider_name] = list_models_for_provider(provider_name)
    return cache[provider_name]

def render_config_page() -> None:
    """Render the pipeline configuration page."""

    # Header
    st.markdown(
        """
        <div class="main-header">
            <span class="header-bottom-left"></span>
            <span class="header-bottom-right"></span>
            <h1>&#9670; SEC Filings Intelligence</h1>
            <div class="subtitle">PIPELINE CONFIGURATION</div>
            <div class="status-pill status-pill-config">CONFIGURING</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Text2Cypher Configuration ────────────────────────────────────────────
    st.markdown(
        '<div class="config-section-title">Text2Cypher Backend</div>',
        unsafe_allow_html=True,
    )

    backend_options = {b[0]: b[1] for b in TEXT2CYPHER_BACKENDS}
    backend_defaults = {b[0]: b[2] for b in TEXT2CYPHER_BACKENDS}
    backend_keys = list(backend_options.keys())

    col1, col2 = st.columns(2)

    with col1:
        cypher_backend = st.selectbox(
            "Backend",
            options=backend_keys,
            format_func=lambda k: backend_options[k],
            index=0,
            key="_cypher_backend_select",
        )

    # Fetch models for selected backend
    with col2:
        if cypher_backend in ("groq", "ollama"):
            models = _fetch_models(cypher_backend, "_cypher_models_cache")
            if models:
                model_ids = [m.id for m in models]
                cypher_model = st.selectbox(
                    "Model",
                    options=model_ids,
                    index=0,
                    key="_cypher_model_select",
                )
            else:
                default = backend_defaults.get(cypher_backend, "")
                cypher_model = st.text_input(
                    "Model",
                    value=default,
                    key="_cypher_model_input",
                    help="Could not fetch models. Enter model name manually.",
                )
        elif cypher_backend == "openai":
            cypher_model = st.text_input(
                "Model",
                value="qwen2.5-coder",
                key="_cypher_model_openai",
                help="Enter model name for OpenAI-compatible server",
            )
        else:
            cypher_model = st.text_input("Model", value="", key="_cypher_model_fallback")

    # ── Answer Generation LLM ────────────────────────────────────────────────
    st.markdown(
        '<div class="config-section-title">Answer Generation LLM</div>',
        unsafe_allow_html=True,
    )

    providers = get_available_providers()
    # Filter out providers with missing keys
    available = [p for p in providers if p["key_status"] != "missing"]

    if not available:
        st.error("No LLM providers available. Check your API keys in `.env`.")
        return

    # Show all providers with status indicators
    provider_labels = {}
    for p in providers:
        dot = {"ok": "✅", "missing": "❌", "n/a": ""}.get(p["key_status"], "")
        provider_labels[p["name"]] = f"{p['display']} {dot}"

    available_names = [p["name"] for p in available]

    col1, col2 = st.columns(2)

    with col1:
        answer_provider_name = st.selectbox(
            "Provider",
            options=available_names,
            format_func=lambda k: provider_labels.get(k, k),
            index=0,
            key="_answer_provider_select",
        )

    with col2:
        answer_models = _fetch_models(answer_provider_name, "_answer_models_cache")
        if answer_models:
            answer_model_ids = [m.id for m in answer_models[:30]]
            answer_model = st.selectbox(
                "Model",
                options=answer_model_ids,
                index=0,
                key="_answer_model_select",
            )
        else:
            answer_model = st.text_input(
                "Model",
                value="",
                key="_answer_model_input",
                help="Could not fetch models. Enter model name manually.",
            )

    # ── OpenAI-compatible server settings (shown only for openai backend) ────
    openai_host = "localhost"
    openai_port = 1234
    if cypher_backend == "openai":
        st.markdown(
            '<div class="config-section-title">OpenAI-Compatible Server</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            openai_host = st.text_input("Host", value="localhost", key="_openai_host")
        with col2:
            openai_port = st.text_input("Port", value="1234", key="_openai_port")

    # ── Pipeline Options ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="config-section-title">Pipeline Options</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        use_schema_injection = st.toggle("Schema Injection", value=True, key="_schema_toggle")
    with col2:
        enable_entity_resolution = st.toggle("Entity Resolution", value=True, key="_entity_toggle")
    with col3:
        enable_few_shot = st.toggle("Few-Shot Examples", value=True, key="_fewshot_toggle")
    with col4:
        verbose = st.toggle("Verbose Mode", value=False, key="_verbose_toggle")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        retry_module = st.toggle("Retry on Error", value=True, key="_retry_toggle")
    with col2:
        embed_vector_queries = st.toggle("Embed Vector Queries", value=False, key="_embed_toggle")

    col1, col2 = st.columns(2)
    with col1:
        retry_strategy = st.selectbox(
            "Retry Strategy",
            options=["full", "lean"],
            index=0,
            key="_retry_strategy_select",
        )
    with col2:
        few_shot_model = st.selectbox(
            "Few-Shot Model",
            options=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "nomic-ai/nomic-embed-text-v1.5",
            ],
            index=0,
            key="_fewshot_model_select",
        )

    # ── Summary ──────────────────────────────────────────────────────────────

    server_info = f"`{openai_host}:{openai_port}`" if cypher_backend == "openai" else "—"
    summary_md = f"""
| Setting | Value |
|---|---|
| **Text2Cypher** | `{backend_options.get(cypher_backend, cypher_backend)}` / `{cypher_model}` |
| **Server** | {server_info} |
| **Answer LLM** | `{answer_provider_name}` / `{answer_model}` |
| **Schema Injection** | {'ON' if use_schema_injection else 'OFF'} |
| **Entity Resolution** | {'ON' if enable_entity_resolution else 'OFF'} |
| **Few-Shot** | {'ON' if enable_few_shot else 'OFF'} (`{few_shot_model.split('/')[-1]}`) |
| **Retry on Error** | {'ON' if retry_module else 'OFF'} (`{retry_strategy}`) |
| **Embed Vector Queries** | {'ON' if embed_vector_queries else 'OFF'} |
| **Verbose** | {'ON' if verbose else 'OFF'} |
"""
    st.markdown(summary_md)

    # ── Launch Button ────────────────────────────────────────────────────────
    st.markdown("")

    if st.button("Launch Pipeline", type="primary", use_container_width=True, key="_launch_btn"):
        config = PipelineConfig(
            cypher_backend=cypher_backend,
            cypher_model=cypher_model,
            openai_compatible_host=openai_host,
            openai_compatible_port=int(openai_port),
            answer_provider_name=answer_provider_name,
            answer_model=answer_model,
            use_schema_injection=use_schema_injection,
            enable_entity_resolution=enable_entity_resolution,
            enable_few_shot=enable_few_shot,
            verbose=verbose,
            few_shot_embedding_model=few_shot_model,
            retry_module=retry_module,
            retry_strategy=retry_strategy,
            embed_vector_queries=embed_vector_queries,
        )

        with st.spinner("Initializing pipeline... Connecting to Neo4j and loading models."):
            try:
                pipeline = init_pipeline(config)
                st.session_state.pipeline = pipeline
                st.session_state.pipeline_configured = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {e}")
