# SEC Filings Analysis — Hybrid RAG System

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

## Data Extraction Workflow

Before running queries, you need to extract and ingest SEC filing data into Neo4j. This is done via Jupyter notebooks in the `notebooks/` directory.

### Extraction Order (Important!)

**Execute notebooks in this specific order:**

#### 1. **Vanguard Extraction** (`notebooks/vanguard.ipynb`)
   - Extracts N-CSR, 497K (Summary Prospectus), and N-PORT filings for **Vanguard funds**
   - Uses provider-specific parsers optimized for Vanguard's document structure
   - Outputs: Fund metadata, holdings, financial highlights, profiles (objectives, strategies, risks)

#### 2. **Multi-Provider Extraction** (`notebooks/edgar.ipynb`)
   - Extracts filings for **iShares (BlackRock)** and **Fidelity** funds
   - Supports the same filing types: N-CSR, 497K, N-PORT
   - Uses `edgartools` library for SEC EDGAR API access
   - Leverages provider-specific extractors:
     - `VanguardProspectusExtractor` (Vanguard format)
     - `iSharesProspectusExtractor` (BlackRock/iShares format)
     - `FidelityProspectusExtractor` (Fidelity format)
   - Auto-detects provider via keyword matching in filing text

### Filing Types Extracted

#### Fund Filings (Vanguard, iShares, Fidelity)

| Filing Type | Description | Data Extracted |
|---|---|---|
| **N-CSR** | Certified Shareholder Report | Financial highlights (expense ratio, turnover, total return, net assets by year) |
| **497K** | Summary Prospectus | Fund name, ticker, objective, strategies, risks, managers, performance |
| **N-PORT** | Portfolio Holdings Report | Complete portfolio holdings with shares, market value, fair value levels, issuer details |

#### Company Filings (10-K Analysis)

| Filing Type | Description | Data Extracted |
|---|---|---|
| **10-K** | Annual Report | Income statement, balance sheet, cash flow, business sections, risk factors, MD&A, legal proceedings, properties |
| **DEF 14A** | Proxy Statement | Executive compensation (Pay-vs-Performance table), CEO compensation, shareholder return |
| **Form 4** | Insider Trading Report | Insider transactions (buys/sells), transaction dates, shares, prices, remaining holdings |

### Extraction Process

**Phase 1: Fund Data Extraction** (`vanguard.ipynb`, `edgar.ipynb`)
1. **Retrieves filings** from SEC EDGAR using `edgartools`
2. **Parses documents** using provider-specific extractors (`ProspectusExtractor.from_text()`)
3. **Structures data** into Pydantic models (`FundData`, `HoldingData`, etc.)
4. **Caches results** to `.pkl` files for reuse

**Phase 2: Company Data Extraction** (`neo4j.ipynb` - Part 1)
1. **Identifies companies** from fund holdings (extracts unique tickers from Portfolio → Holding relationships)
2. **Retrieves 10-K filings** using `TenKParser` for each company
3. **Parses financial statements** via XBRL concept mapping
4. **Extracts compensation data** from DEF 14A filings using `Def14AParser`
5. **Collects insider transactions** from Form 4 filings using `Form4Parser`
6. **Structures company data** into `CompanyEntity` Pydantic models

**Phase 3: Neo4j Ingestion** (`neo4j.ipynb` - Part 2)
1. **Initializes Neo4j database** (reset, create constraints & indexes)
2. **Creates provider hierarchy** (Provider → Trust nodes)
3. **Ingests fund data**:
   - Fund nodes with metadata (ticker, name, expense ratio, net assets, etc.)
   - Share classes, profiles (objectives, strategies, risks with embeddings)
   - Financial highlights (year-by-year performance metrics)
   - Sector/region allocations, average returns
4. **Ingests portfolio holdings**:
   - Portfolio nodes with reporting dates
   - Holding nodes with detailed security information
   - HAS_HOLDING relationships with shares, market value, fair value levels
   - REPRESENTS relationships linking holdings to companies
5. **Ingests company filings**:
   - Company nodes (ticker, name, CIK)
   - Filing10K nodes with document metadata
   - Section chunks (risk factors, business info, MD&A, etc.) with embeddings
   - Financial metrics and segment breakdowns
   - CEO compensation and insider transaction data
6. **Creates document provenance** via `EXTRACTED_FROM` relationships

### Running the Complete Pipeline

```bash
# Start Jupyter
uv run jupyter notebook

# Execute in order:
# 1. notebooks/vanguard.ipynb      (Extract Vanguard fund data)
# 2. notebooks/edgar.ipynb          (Extract iShares & Fidelity fund data)
# 3. notebooks/neo4j.ipynb          (Extract company filings + Ingest all data into Neo4j)
```

After the pipeline completes, the Neo4j database will contain the full knowledge graph ready for querying.


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
