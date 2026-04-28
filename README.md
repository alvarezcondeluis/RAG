# SEC Filings Intelligence — RAG System

Final Master's Thesis (TFM) project: A RAG system that combines a **Neo4j knowledge graph** with **LLM-powered natural language querying** for SEC financial data — mutual funds, ETFs, and company 10-K filings.

Users ask questions in plain English and the system translates them into Cypher queries, executes them against the knowledge graph, and generates analyst-grade answers with source citations.

## Pipeline

```
User Query → SetFit Classifier → Schema Slice Selection → Entity Resolution (fuzzy matching)
→ Few-Shot Example Retrieval (FAISS) → Text2Cypher LLM → Cypher Validation Loop
→ Neo4j Execution → Result Classification → Answer Generation LLM → Response
```

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **Package Manager** | `uv` (with `pyproject.toml` + `uv.lock`) |
| **Graph Database** | Neo4j 5.27 (Community, Docker) |
| **LLM Orchestration** | LangChain + LlamaIndex |
| **Text2Cypher Models** | Ollama (local), Groq API, OpenAI-compatible (llama.cpp / LM Studio) |
| **Answer Generation** | Groq, OpenRouter, Google Gemini, Ollama (pluggable) |
| **Embeddings** | `nomic-ai/nomic-embed-text-v1.5` (1536d), `all-MiniLM-L6-v2` (few-shot selector) |
| **Query Classification** | SetFit (`sentence-transformers/paraphrase-mpnet-base-v2`) |
| **Data Source** | SEC EDGAR (10-K, N-CSR, N-PORT filings via `edgartools`) |
| **Frontend** | Streamlit |
| **Visualization** | Plotly, Seaborn |

## Project Structure

```
src/simple_rag/
├── database/neo4j/              # Neo4j connection, schema, CRUD operations
│   ├── base.py                  # Base connection & query execution
│   ├── database.py              # Unified controller (mixin pattern)
│   ├── schema_manager.py        # Constraints & indexes
│   ├── config.py                # Connection config (loads .env)
│   ├── models/                  # Node & relationship definitions
│   └── operations/              # CRUD ops (company, fund, holdings, providers)
├── rag/
│   ├── text2cypher.py           # NL → Cypher translation (multi-backend)
│   ├── query_handler.py         # Orchestrates: classify → schema slice → translate
│   ├── orchestrator.py          # Terminal REPL for interactive pipeline usage
│   ├── schema_definitions.py    # Full Neo4j schema definition
│   ├── prompt_templates.py      # Text2Cypher prompt templates
│   ├── entity_resolver.py       # Fuzzy name matching (rapidfuzz)
│   ├── dynamic_few_shot.py      # FAISS-based example retrieval
│   ├── context_enrichment.py    # Supplementary queries + document provenance
│   ├── llm_providers/           # Pluggable LLM backends for answer generation
│   │   ├── base.py              # LLMProvider ABC, LLMResponse, ModelInfo
│   │   ├── groq_provider.py
│   │   ├── openrouter_provider.py
│   │   ├── gemini_provider.py
│   │   ├── ollama_provider.py
│   │   └── registry.py          # ProviderRegistry + interactive terminal selection
│   ├── answer_generation/       # Answer generation from raw Neo4j results
│   │   ├── prompt_templates.py  # Financial analyst persona prompts
│   │   └── result_classifier.py # Deterministic result type classification
│   ├── query/                   # Query classification (SetFit)
│   └── post_processing/         # Cypher validation
├── extraction/                  # SEC filing parsers (10-K, N-CSR, N-PORT)
├── evaluation/                  # Benchmarking (Text2CypherBenchmark)
│   ├── benchmark.py             # Main benchmark class
│   ├── test_set.json            # 600+ test cases with ground truth Cypher
│   └── run_benchmark.py         # Benchmark runner
├── models/                      # Pydantic models (FundData, CompanyEntity)
├── embeddings/                  # Embedding generation (Nomic)
├── retriever/                   # Vector search (Qdrant) + reranking
├── streamlit_app/               # Streamlit frontend
└── utils/                       # Helpers (charts, caching, financial mapping)
```

## Getting Started

### Prerequisites



### Environment Setup

1. **Install `uv`**:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:

   ```bash
   uv sync
   ```

3. **Configure environment variables** — create a `.env` file in the project root:

   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_PASSWORD=YourPassword
   NEO4J_DATABASE=neo4j

   GROQ_API_KEY=your_key
   OPEN_ROUTER_API_KEY=your_key
   GOOGLE_AI_STUDIO_API_KEY=your_key
   ```

### Neo4j Database

A Docker Compose configuration is provided to launch an APOC-enabled Neo4j instance.

```bash
cd neo4j
docker-compose up -d
```

- **Browser UI**: [http://localhost:7474](http://localhost:7474) (user: `neo4j`)
- **Bolt**: `bolt://localhost:7687`

## Running

| Command | Description |
|---|---|
| `uv run python -m simple_rag.rag.orchestrator` | Interactive terminal REPL (select models, ask questions) |
| `uv run python run_benchmark.py` | Run Text2Cypher benchmark (600+ test cases) |
| `uv run streamlit run src/simple_rag/streamlit_app/app.py` | Launch Streamlit frontend |
| `uv run pytest tests/` | Run tests |

## Neo4j Schema Overview

The knowledge graph models SEC financial data across two main domains:

**Fund Management** — Provider → Trust → Fund → Portfolio → Holdings, with fund profiles (strategy, risk, objectives), financial highlights, sector/region allocations, average returns, and share classes.

**Company Filings** — Company → Filing10K → Sections (risk factors, business info, financials, management discussion), with CEO compensation, insider transactions, and financial metrics with segment breakdowns.

All data nodes trace back to their source via `(:Node)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})` relationships, enabling full provenance tracking in answers.

## Tests

```bash
uv run pytest tests/                         # All tests
uv run pytest tests/test_cypher_validator.py  # Cypher validation
uv run pytest tests/test_entity_resolver.py   # Entity resolution
uv run pytest tests/test_providers.py         # LLM provider connectivity
```

## License

See [LICENSE](LICENSE).
