# SEC Filings Intelligence Assistant

A professional Streamlit chatbot interface for querying SEC filings and fund data using RAG (Retrieval-Augmented Generation) with Neo4j knowledge graph integration.

## Features

### 🎨 Professional Finance-Focused Design

- Navy and gold color palette
- Formal typography (Playfair Display, Inter, JetBrains Mono)
- Clean, minimal interface with subtle shadows
- Responsive layout

### 💬 Chat Interface

- Real-time message streaming
- Source citations with SEC EDGAR links
- Query suggestions and examples
- Conversation history

### 🔍 Advanced Filtering

- Document type filters (10-K, DEF 14A, Form 4, N-CSR)
- Date range selection
- Company and fund filters
- Search mode selection (Hybrid, Semantic, Graph)

### 📊 Session Management

- Query statistics
- Conversation export (JSON)
- Clear history functionality
- Database connection status

## Installation

### Prerequisites

- Python 3.11+
- Neo4j database running
- Streamlit

### Setup

1. **Install dependencies:**

```bash
pip install streamlit
```

2. **Configure environment variables:**
   Create a `.env` file in the project root:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Groq API (optional, for production)
GROQ_API_KEY=your_groq_api_key

# Mock mode (for development)
MOCK_MODE=true
```

3. **Run the application:**

```bash
cd /home/luis/Desktop/code/RAG
streamlit run src/simple_rag/streamlit_app/app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### Sample Queries

**Financial Analysis:**

- "What were Apple's revenue segments in their latest 10-K?"
- "Compare Microsoft and Google's revenue growth over the past 3 years"

**Risk Assessment:**

- "Summarize the key risk factors for Tesla from their latest filing"
- "What legal proceedings is Amazon currently facing?"

**Fund Analysis:**

- "What are the top holdings of Vanguard Total Stock Market ETF?"
- "Which funds have the highest exposure to the technology sector?"

**Executive Compensation:**

- "Show me the CEO compensation for all companies in VTSAX's top 10 holdings"
- "How does Apple's CEO pay compare to shareholder returns?"

**Insider Trading:**

- "What insider transactions occurred at Meta in the last 6 months?"
- "Show me all insider sales above $1M for companies in my portfolio"

### Filters

Use the sidebar to refine your queries:

- **Document Types:** Filter by SEC filing type
- **Date Range:** Limit results to specific time periods
- **Companies:** Focus on specific tickers
- **Funds:** Filter by fund holdings
- **Search Mode:** Choose retrieval strategy
- **Number of Sources:** Control how many documents to retrieve

## Architecture

```
src/simple_rag/streamlit_app/
├── app.py                    # Main application
├── config.py                 # Configuration and settings
├── components/
│   ├── chat_interface.py     # Message display and sources
│   └── sidebar.py            # Filters and settings
├── styles/
│   └── main.css              # Professional CSS styling
└── utils/
    └── session_manager.py    # Conversation state management
```

## Development Mode

The application currently runs in **mock mode** for development:

- Mock responses are generated based on query keywords
- No actual LLM API calls are made
- Sample data demonstrates the interface and features

### Enabling Production Mode

To use real RAG with Groq API:

1. Set `GROQ_API_KEY` in your `.env` file
2. Set `MOCK_MODE=false`
3. Implement the RAG pipeline in `utils/rag_pipeline.py`

## Customization

### Styling

Edit `styles/main.css` to customize:

- Color palette
- Typography
- Component styling
- Animations

### Configuration

Edit `config.py` to modify:

- LLM settings (model, temperature, max tokens)
- RAG parameters (top-k results, search weights)
- Sample queries
- Application metadata

## Future Enhancements

- [ ] Real RAG pipeline with Groq API
- [ ] Vector database integration
- [ ] Data visualizations (charts, graphs)
- [ ] PDF export functionality
- [ ] Voice input support
- [ ] Multi-language support
- [ ] Advanced analytics dashboard

## Troubleshooting

### Neo4j Connection Issues

- Ensure Neo4j is running: `docker ps`
- Check credentials in `.env` file
- Verify URI format: `bolt://localhost:7687`

### Styling Not Loading

- Clear browser cache
- Check that `styles/main.css` exists
- Restart Streamlit server

### Import Errors

- Ensure you're running from the project root
- Check Python path includes `src/`
- Verify all dependencies are installed

## License

Part of the SEC Filings RAG project.
