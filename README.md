# RAG Systems Study - Final Master's Thesis (TFM)

This repository contains the code and implementations for my Final Master's Thesis (TFM) focused on **Retrieval-Augmented Generation (RAG) systems**. The project involves building comprehensive RAG pipelines and locally augmenting Large Language Models (LLMs) using cutting-edge approaches and techniques.

## Project Overview

The study explores different RAG architectures and configurations, implementing various phases of the RAG pipeline using state-of-the-art methodologies:

- **Document Processing & Parsing**: Advanced text extraction and preprocessing for financial documentation. Includes extracting data from SEC EDGAR filings such as 497K prospectuses and N-CSR docs.
- **Embedding Generation**: Strategies combining semantic textual representations with structured data context.
- **Vector Storage & Graph Retrieval**: Efficient similarity search blended with robust Neo4j graph traversal (Hybrid RAG approach).
- **LLM Integration**: Local augmentation techniques powered by `llama.cpp` using optimized `.gguf` models to ensure security and privacy.
- **Pipeline Evaluation & Optimization**: Dedicated modules for benchmarking queries, models, and retrieval accuracy.
- **Interactive UI**: A fully-fledged Streamlit application for end-users to query and converse with the system.

## Tech Stack & Architecture

- **Language**: Python 3.10+
- **Environment Management**: `uv` (Fast Python dependency management)
- **Knowledge Graph & Vector DB**: Neo4j (Graph Database with Vector Indexing)
- **Local Inference Engine**: `llama.cpp`
- **Application Interface**: Streamlit

## Getting Started

### Prerequisites

Install the required system-level dependencies for document OCR and parsing capabilities:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr ghostscript poppler-utils

# macOS
brew install tesseract ghostscript poppler
```

### Environment Setup

This project strictly utilizes `uv` for reproducible and fast environment creation.

1. **Install `uv`** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Project Dependencies**:
   Inside the root directory of the project, initialize the environment and install dependencies from the lockfile:

   ```bash
   uv sync
   ```

3. **Run Code within the Environment**:
   Always prefix your execution commands with `uv run` to ensure you are operating within the isolated environment.
   ```bash
   uv run python <script.py>
   # or
   uv run streamlit run src/simple_rag/streamlit_app/app.py
   ```

## Infrastructure Setup

### Deploying the Neo4j Graph Database

The application depends on a running Neo4j Database instance. A Docker Compose configuration is provided to quickly launch the APOC-enabled Community Edition locally.

1. **Navigate to the `neo4j` Directory**:

   ```bash
   cd neo4j
   ```

2. **Configure Environment Variables**:
   Ensure you have a `.env` file within the `neo4j` folder containing your authentication variables. An example structure:

   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_PASSWORD=YourPassword123
   NEO4J_DATABASE=neo4j
   ```

3. **Deploy the Container**:
   Start the database in the background using docker-compose:

   ```bash
   docker-compose up -d
   ```

4. **Verify Deployment**:
   - The Neo4j Browser UI is accessible at: [http://localhost:7474](http://localhost:7474).
   - Sign in using the username `neo4j` and your configured `NEO4J_PASSWORD`.
