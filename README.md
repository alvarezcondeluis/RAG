# RAG Systems Study - Final Master's Thesis (TFM)

This repository contains the code and implementations for my Final Master's Thesis (TFM) focused on **Retrieval-Augmented Generation (RAG) systems**. The project involves building comprehensive RAG pipelines and locally augmenting Large Language Models (LLMs) using cutting-edge approaches and techniques.

## Project Overview

The study explores different RAG architectures and configurations, implementing various phases of the RAG pipeline using state-of-the-art methodologies:

- **Document Processing & Parsing**: Advanced text extraction and preprocessing
- **Embedding Generation**: Multiple embedding strategies and models
- **Vector Storage & Retrieval**: Efficient similarity search implementations  
- **LLM Integration**: Local augmentation techniques for enhanced generation
- **Pipeline Optimization**: Performance analysis and configuration tuning

## Environment Setup

This project uses **Poetry** for dependency management and **Conda** for environment isolation.

### Prerequisites

Install the following system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr ghostscript poppler-utils

# macOS
brew install tesseract ghostscript poppler
```

### Environment Installation

1. **Create Conda Environment**:
   ```bash
   conda create -n rag-env python=3.11
   conda activate rag-env
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install Project Dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the Environment**:
   ```bash
   conda activate rag-env
   poetry shell
   ```