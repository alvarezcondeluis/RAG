# SEC Filings Intelligence — Hybrid RAG System

Final Master's Thesis (TFM) project: a production-grade RAG system that combines a **Neo4j knowledge graph** with **LLM-powered natural language querying** over SEC financial data — mutual funds, ETFs, and company 10-K filings. Users ask questions in plain English; the system translates them into Cypher queries, executes them against the knowledge graph, and streams analyst-grade answers with source citations.

## Pipeline

```
User Query
  → SetFit Classifier (9 categories)
  → Schema Slice Selection
  → Entity Resolution (fuzzy name/ticker matching via rapidfuzz)
  → Few-Shot Example Retrieval (FAISS, nomic-embed-text-v1.5)
  → Text2Cypher LLM (Ollama / Groq / OpenAI-compatible)
  → Cypher Validation + Retry Loop (up to 3 attempts with error-specific feedback)
  → Neo4j Execution
  → Result Classification (table / scalar / text / chart / empty)
  → Context Enrichment (supplementary queries + document provenance)
  → Answer Generation LLM (Groq / Gemini / OpenRouter / Ollama)
  → Streamed Response with source citations
```

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **Package Manager** | `uv` (`pyproject.toml` + `uv.lock`) |
| **Graph Database** | Neo4j 5.27 Community (Docker) |
| **LLM Orchestration** | LangChain |
| **Text2Cypher Models** | Ollama (local), Groq API, OpenAI-compatible (llama.cpp / LM Studio / vLLM) |
| **Answer Generation** | Groq, OpenRouter, Google Gemini, Ollama (pluggable via `ProviderRegistry`) |
| **Embeddings** | `nomic-ai/nomic-embed-text-v1.5` (1536d) for queries; `all-MiniLM-L6-v2` for few-shot selector |
| **Query Classification** | SetFit (`sentence-transformers/paraphrase-mpnet-base-v2`) |
| **Data Source** | SEC EDGAR (10-K, DEF 14A, Form 4, N-CSR, N-PORT, 497K via `edgartools`) |
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
│   └── operations/              # CRUD ops (company, fund, holdings, providers)
├── rag/
│   ├── text2cypher.py           # NL → Cypher translation (multi-backend)
│   │                            #   • Validation + retry loop (up to 3 attempts)
│   │                            #   • EXPLAIN/PROFILE prefix stripping
│   │                            #   • Auto-fix undefined relationship variables
│   │                            #   • Groq 70B fallback after exhausted retries
│   ├── query_handler.py         # Orchestrates: classify → schema slice → translate → execute
│   ├── orchestrator.py          # Terminal REPL for interactive pipeline setup
│   ├── schema_definitions.py    # Full Neo4j schema (nodes, relationships, properties)
│   ├── schema_slices.py         # Per-category schema subsets for prompt injection
│   ├── prompt_templates.py      # Text2Cypher + retry prompt templates
│   ├── entity_resolver.py       # Fuzzy name/ticker matching (rapidfuzz)
│   ├── dynamic_few_shot.py      # FAISS-based example retrieval
│   ├── context_enrichment.py    # Supplementary Cypher queries + document provenance
│   ├── groq_wrapper.py          # Groq LangChain wrapper with rate-limit tracking
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
│   ├── query/                   # SetFit query classifier (train + inference)
│   │   ├── query_classification.py
│   │   └── retrain_classifier.py
│   └── post_processing/         # Cypher validation rules
│       └── cypher_validator.py  # 15+ rule checks (syntax, schema, anti-patterns)
├── extraction/                  # SEC filing parsers
│   ├── tenk_parser.py           # 10-K annual reports (XBRL income statement, sections)
│   ├── def14a_parser.py         # Proxy statements (CEO comp, Pay-vs-Performance table)
│   ├── form4_parser.py          # Insider transactions (Form 4)
│   ├── ncsr_parser.py           # Fund annual/semi-annual reports (N-CSR)
│   ├── nport.py                 # Portfolio holdings (NPORT-P)
│   └── prospectus_parser.py     # Fund summary prospectuses (497K)
├── evaluation/                  # Benchmarking
│   ├── benchmark.py             # Text2CypherBenchmark class
│   ├── test_set.json            # 600+ test cases with ground truth Cypher
│   └── run_benchmark.py         # CLI runner
├── models/                      # Pydantic data models
│   ├── company.py               # CompanyEntity, Filing10K, IncomeStatement, InsiderTransaction
│   └── fund.py                  # FundData, HoldingData, FinancialHighlight
├── embeddings/                  # Nomic embedding generation
├── streamlit_app/               # Streamlit web frontend
│   ├── app.py                   # Main entry point + routing
│   ├── pipeline_bridge.py       # UI ↔ RAG orchestrator bridge (streaming)
│   ├── config.py                # App config (models, sample queries, Neo4j settings)
│   ├── components/              # UI components (chat, sidebar, config page)
│   └── styles/main.css          # Bloomberg terminal aesthetic
└── utils/                       # Helpers (cache manager, financial concept mapping)
    ├── cache_manager.py         # EDGAR filing cache (warm + invalidate)
    └── financial_concept_map.py # XBRL concept → IncomeStatement field mapping
```

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker (for Neo4j)
- At least one LLM backend:
  - **Ollama** (local, free) — recommended for offline use
  - **Groq API key** — fast cloud inference, free tier available
  - **LM Studio / llama.cpp** — OpenAI-compatible local server

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
   # Neo4j
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=YourPassword
   NEO4J_DATABASE=neo4j

   # LLM API keys (add whichever you plan to use)
   GROQ_API_KEY=your_groq_key
   OPEN_ROUTER_API_KEY=your_openrouter_key
   GOOGLE_AI_STUDIO_API_KEY=your_gemini_key
   ```

### Neo4j Database

A Docker Compose configuration is provided for an APOC-enabled Neo4j instance:

```bash
cd neo4j
docker-compose up -d
```

- **Browser UI**: [http://localhost:7474](http://localhost:7474)
- **Bolt**: `bolt://localhost:7687`

---

## Data Extraction & Ingestion Workflow

Before querying, you need to extract SEC filing data and populate the Neo4j knowledge graph. This is done via the three Jupyter notebooks **in this exact order**.

```bash
uv run jupyter notebook
```

---

### Notebook 1 — `notebooks/vanguard.ipynb` (Vanguard Funds)

Extracts and ingests Vanguard fund data from SEC EDGAR.

**What it downloads:**

| Filing Type | Description | Data Extracted |
|---|---|---|
| **N-CSR** | Certified Shareholder Report | Year-by-year financial highlights: expense ratio, turnover rate, total return, net assets, NAV start/end |
| **497K** | Summary Prospectus | Fund name, ticker, share class, investment objective, principal strategies, risk factors, portfolio managers, performance commentary |
| **N-PORT** | Portfolio Holdings Report | Complete holding list with CUSIP/ISIN, shares, market value, fair value levels (1/2/3), issuer category, country |

**What it creates in Neo4j:**
- `Provider → Trust → Fund` hierarchy
- `ShareClass`, `Profile` nodes with sections (`Objective`, `Strategy`, `RiskFactor`, `PerformanceCommentary`)
- `FinancialHighlight` nodes with `HAS_FINANCIAL_HIGHLIGHT {year}` relationships
- `AverageReturns` nodes (1y, 5y, 10y, inception)
- `Sector` and `Region` allocation nodes with `weight` on relationships
- `Portfolio → Holding` graph with `HAS_HOLDING` relationship properties
- `EXTRACTED_FROM → Document` provenance edges

---

### Notebook 2 — `notebooks/edgar.ipynb` (iShares & Fidelity Funds)

Extracts and ingests fund data for iShares (BlackRock) and Fidelity, using the same filing types as above but with provider-specific parsers that handle each company's document structure.

**Auto-detects provider** via keyword matching in filing text:
- `VanguardProspectusExtractor`
- `iSharesProspectusExtractor`
- `FidelityProspectusExtractor`

The output schema in Neo4j is identical to Notebook 1 — same node types, same relationships.

---

### Notebook 3 — `notebooks/neo4j.ipynb` (Company 10-K Data + Full Ingestion)

The largest notebook, split into two phases.

#### Phase A — Company Filing Extraction

For each company ticker found in fund portfolio holdings, downloads and parses:

| Filing Type | Description | Data Extracted |
|---|---|---|
| **10-K** | Annual Report | Income statement (revenue + segments, gross profit, operating income, net income, EPS), balance sheet text, cash flow text, and five narrative sections: Business (Item 1), Risk Factors (Item 1A), Properties (Item 2), Legal Proceedings (Item 3), MD&A (Item 7) |
| **DEF 14A** | Proxy Statement | CEO name, total compensation, actually-paid compensation, shareholder return (Pay-vs-Performance table), fiscal year end |
| **Form 4** | Insider Trading | Transaction date, type (BUY/SELL/GRANT/GIFT/TAX/UNKNOWN), shares, price, value, remaining shares, insider name and position |

**Important notes on extraction:**
- Only original `10-K` filings are fetched — `10-K/A` amendments (which contain only Part III addenda and lack financials) are explicitly excluded
- XBRL income statement data supports multi-year extraction (typically 3 fiscal years per filing)
- Revenue and operating income segments are preserved with their dimensional axes (product, geography, business segment)
- Gross profit is computed from Revenue − Cost of Sales when not directly reported

#### Phase B — Full Neo4j Ingestion

Runs `db.ingest_companies_batch(companies, verbose=True)` which writes:

- `Company` nodes (ticker, name, CIK)
- `Filing10K` nodes linked via `REPORTS_IN {year}` — year is derived from `report_period_end`, not `filing_date`, ensuring companies that file in January for the prior fiscal year (GOOG, AMZN, TSLA, META) are matched correctly
- `Section` nodes for each narrative section (`RiskFactor`, `BusinessInformation`, `LegalProceeding`, `ManagemetDiscussion`, `Properties`)
- `Financials` nodes with `FinancialMetric` children and `Segment` breakdowns
- `Person` (CEO) with `HAS_CEO` relationship carrying compensation fields
- `CompensationPackage` nodes with `fiscalYearEnd` and shareholder return
- `InsiderTransaction` nodes linked to `Person` via `MADE_BY`
- `Document` provenance nodes for all source filings

Then runs `db.ingest_filing_chunks(companies)` which writes `SectionChunk` nodes with embeddings for vector similarity search.

---

## Running the Application

### Streamlit Web Interface (recommended)

```bash
uv run streamlit run src/simple_rag/streamlit_app/app.py
```

Opens a configuration page where you select:
- **Text2Cypher backend**: Ollama / Groq / OpenAI-compatible (LM Studio, llama.cpp, vLLM)
- **Answer generation provider**: Groq / Gemini / OpenRouter / Ollama
- Pipeline toggles: schema injection, entity resolution, few-shot examples, retry module

Once configured, use the chat interface to ask questions about funds, companies, and SEC filings.

### Terminal REPL (interactive)

```bash
uv run python -m simple_rag.rag.orchestrator
```

Same configuration wizard in the terminal. Useful for debugging and low-latency testing.

### Benchmark Runner

```bash
uv run python src/simple_rag/evaluation/run_benchmark.py
```

Runs 600+ test cases with ground truth Cypher against the full pipeline. Reports are saved to `src/simple_rag/evaluation/reports/`. Key metrics: accuracy, avg latency, retry module effectiveness, validator rule violation breakdown.

### Tests

```bash
uv run pytest tests/                          # All tests
uv run pytest tests/test_cypher_validator.py  # Cypher validation rules
uv run pytest tests/test_entity_resolver.py   # Entity resolution
uv run pytest tests/test_providers.py         # LLM provider connectivity
```

---

## Neo4j Schema Overview

The knowledge graph spans two domains:

**Fund Management**
```
Provider -[:MANAGES]-> Trust -[:ISSUES]-> Fund
Fund -[:HAS_SHARE_CLASS]-> ShareClass
Fund -[:DEFINED_BY]-> Profile -[:HAS_SECTION]-> {Objective, Strategy, RiskFactor, PerformanceCommentary}
Fund -[:HAS_FINANCIAL_HIGHLIGHT {year}]-> FinancialHighlight
Fund -[:HAS_AVERAGE_RETURNS]-> AverageReturns
Fund -[:HAS_SECTOR_ALLOCATION {weight}]-> Sector
Fund -[:HAS_REGION_ALLOCATION {weight}]-> Region
Fund -[:HAS_PORTFOLIO]-> Portfolio -[:HAS_HOLDING {shares, marketValue, weight, fairValueLevel}]-> Holding
Fund -[:MANAGED_BY]-> Person
Holding -[:REPRESENTS]-> Company
Holding -[:OF_ASSET_TYPE]-> AssetCategory
```

**Company Filings**
```
Company -[:REPORTS_IN {year}]-> Filing10K
Filing10K -[:HAS_SECTION]-> Section {RiskFactor, BusinessInformation, LegalProceeding, ManagemetDiscussion, Properties}
Section -[:HAS_CHUNK]-> SectionChunk  (with vector embeddings for similarity search)
Filing10K -[:HAS_FINANCIALS]-> Financials -[:HAS_METRIC]-> FinancialMetric -[:HAS_SEGMENT]-> Segment
Company -[:HAS_CEO]-> Person
Person -[:RECEIVED_COMPENSATION]-> CompensationPackage -[:AWARDED_BY]-> Company
Company -[:HAS_INSIDER_TRANSACTION]-> InsiderTransaction -[:MADE_BY]-> Person
```

**Provenance** — every data node traces to its source:
```
(:Node)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})
```

---

## Query Classification Categories

The SetFit classifier routes queries to the appropriate schema slice before Cypher generation:

| Category | Example Queries |
|---|---|
| `fund_basic` | expense ratio, turnover, net assets, financial highlights, returns |
| `fund_portfolio` | holdings, positions, sector/region weights, fair value levels |
| `fund_profile` | investment objective, strategy, risk factors, prospectus |
| `company_filing` | 10-K sections, financial metrics, MD&A, legal proceedings |
| `company_people` | CEO compensation, insider transactions, executives |
| `not_related` | out-of-scope questions |

---

## License

See [LICENSE](LICENSE).
