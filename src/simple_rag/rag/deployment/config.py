"""
Deployment configuration — loaded from deployment.yaml + .env overrides.

Precedence (lowest → highest):
    deployment.yaml  →  .env file  →  environment variables  →  CLI flags
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    # "docker"   → run docker compose up in ./neo4j/
    # "manual"   → assume already running, fail fast if unreachable
    start_mode: Literal["docker", "manual"] = "manual"
    # Path to the docker-compose.yml (relative to project root)
    compose_dir: str = "neo4j"
    container_name: str = "fund_graph_db"


@dataclass
class LMStudioConfig:
    base_url: str = "http://localhost:1234"
    # Model ID as shown in LM Studio (e.g. "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF")
    model_id: str = ""
    # "auto"   → launch LM Studio CLI (requires lm_studio_executable to be set)
    # "manual" → assume already running, block with instructions if unreachable
    start_mode: Literal["auto", "manual"] = "manual"
    executable: str = ""   # path to LM Studio CLI binary


@dataclass
class PipelineConfig:
    # Text2Cypher
    cypher_backend: str = "openai"   # openai-compatible = LM Studio
    cypher_model: str = "qwen2.5-coder"

    # Answer generation provider
    answer_provider: str = "groq"
    answer_model: str = "llama-3.3-70b-versatile"

    # Toggles — mirror orchestrator.PipelineConfig
    use_schema_injection: bool = True
    enable_entity_resolution: bool = True
    enable_few_shot: bool = True
    verbose: bool = False


@dataclass
class StreamlitConfig:
    port: int = 8501
    app_path: str = "src/simple_rag/streamlit_app/app.py"
    open_browser: bool = True


@dataclass
class TimeoutConfig:
    probe_timeout_s: int = 60
    probe_interval_s: float = 2.0


@dataclass
class DeploymentConfig:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    lm_studio: LMStudioConfig = field(default_factory=LMStudioConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)

    # Convenience: project root (resolved at load time)
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)


def load_config(yaml_path: Optional[str] = None, overrides: Optional[dict] = None) -> DeploymentConfig:
    """
    Load DeploymentConfig from:
    1. deployment.yaml (if it exists)
    2. .env / environment variables
    3. `overrides` dict (from CLI args)
    """
    config = DeploymentConfig()

    # ── 1. YAML ─────────────────────────────────────────────────────────────
    path = Path(yaml_path) if yaml_path else _PROJECT_ROOT / "deployment.yaml"
    if path.exists():
        try:
            import yaml  # PyYAML — optional dep
        except ImportError:
            print("  ⚠  PyYAML not installed — skipping deployment.yaml. Run: uv add pyyaml")
        else:
            with open(path) as f:
                data = yaml.safe_load(f) or {}

            n = data.get("neo4j", {})
            config.neo4j.uri = n.get("uri", config.neo4j.uri)
            config.neo4j.user = n.get("user", config.neo4j.user)
            config.neo4j.password = n.get("password", config.neo4j.password)
            config.neo4j.start_mode = n.get("start_mode", config.neo4j.start_mode)
            config.neo4j.compose_dir = n.get("compose_dir", config.neo4j.compose_dir)
            config.neo4j.container_name = n.get("container_name", config.neo4j.container_name)

            lm = data.get("lm_studio", {})
            config.lm_studio.base_url = lm.get("base_url", config.lm_studio.base_url)
            config.lm_studio.model_id = lm.get("model_id", config.lm_studio.model_id)
            config.lm_studio.start_mode = lm.get("start_mode", config.lm_studio.start_mode)
            config.lm_studio.executable = lm.get("executable", config.lm_studio.executable)

            pl = data.get("pipeline", {})
            config.pipeline.cypher_backend = pl.get("cypher_backend", config.pipeline.cypher_backend)
            config.pipeline.cypher_model = pl.get("cypher_model", config.pipeline.cypher_model)
            config.pipeline.answer_provider = pl.get("answer_provider", config.pipeline.answer_provider)
            config.pipeline.answer_model = pl.get("answer_model", config.pipeline.answer_model)
            config.pipeline.use_schema_injection = pl.get("use_schema_injection", config.pipeline.use_schema_injection)
            config.pipeline.enable_entity_resolution = pl.get("enable_entity_resolution", config.pipeline.enable_entity_resolution)
            config.pipeline.enable_few_shot = pl.get("enable_few_shot", config.pipeline.enable_few_shot)
            config.pipeline.verbose = pl.get("verbose", config.pipeline.verbose)

            st = data.get("streamlit", {})
            config.streamlit.port = st.get("port", config.streamlit.port)
            config.streamlit.app_path = st.get("app_path", config.streamlit.app_path)
            config.streamlit.open_browser = st.get("open_browser", config.streamlit.open_browser)

            to = data.get("timeouts", {})
            config.timeouts.probe_timeout_s = to.get("probe_timeout_s", config.timeouts.probe_timeout_s)
            config.timeouts.probe_interval_s = to.get("probe_interval_s", config.timeouts.probe_interval_s)

    # ── 2. Environment variables (take precedence over yaml) ─────────────────
    if os.getenv("NEO4J_URI"):
        config.neo4j.uri = os.environ["NEO4J_URI"]
    if os.getenv("NEO4J_USERNAME"):
        config.neo4j.user = os.environ["NEO4J_USERNAME"]
    if os.getenv("NEO4J_PASSWORD"):
        config.neo4j.password = os.environ["NEO4J_PASSWORD"]

    # ── 3. CLI overrides ─────────────────────────────────────────────────────
    if overrides:
        if overrides.get("neo4j_start_mode"):
            config.neo4j.start_mode = overrides["neo4j_start_mode"]
        if overrides.get("lm_studio_model"):
            config.lm_studio.model_id = overrides["lm_studio_model"]
        if overrides.get("port"):
            config.streamlit.port = overrides["port"]
        if overrides.get("answer_provider"):
            config.pipeline.answer_provider = overrides["answer_provider"]
        if overrides.get("answer_model"):
            config.pipeline.answer_model = overrides["answer_model"]

    return config
