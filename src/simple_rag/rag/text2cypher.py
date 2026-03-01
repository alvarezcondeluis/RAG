import subprocess
import time
import requests
import re
import logging
from typing import Optional, Literal
from datetime import datetime, timedelta
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from simple_rag.rag.dynamic_few_shot import DynamicFewShotSelector
from simple_rag.rag.entity_resolver import EntityResolver
from simple_rag.rag.groq_wrapper import GroqWrapper
from simple_rag.rag.post_processing.cypher_validator import CypherValidator

logger = logging.getLogger(__name__)

class CypherTranslator:
    def __init__(
        self,
        neo4j_driver,
        model_name: str = "llama3.1:8b",
        api_url: str = "http://localhost:11434",
        auto_start_ollama: bool = True,
        backend: Literal["ollama", "huggingface", "groq", "openai"] = "ollama",
        device: str = "cuda",
        temperature: float = 0.2,
        use_entity_resolver: bool = True,
        entity_resolver_debug: bool = False,
        groq_api_key: Optional[str] = None,
        llama_cpp_host: str = "localhost",
        llama_cpp_port: int = 8080,
    ):
        """
        LangChain-based LLM wrapper for text-to-Cypher translation.

        Args:
            neo4j_driver:        Neo4j driver instance for entity resolution.
            model_name:          Model identifier (Ollama model, HuggingFace path, Groq model,
                                 or any string for llama.cpp — the server ignores it).
            api_url:             Ollama base URL (only used when backend="ollama").
            auto_start_ollama:   Auto-start Ollama if not running (ignored for other backends).
            backend:             "ollama" | "huggingface" | "groq" | "openai".
                                 Use "openai" to target a local llama.cpp server.
            device:              Device for HuggingFace model ("cuda" or "cpu").
            temperature:         Sampling temperature (lower = more deterministic).
            use_entity_resolver: Enable EntityResolver for name/ticker pre-resolution.
            entity_resolver_debug: Enable debug output for entity resolution.
            groq_api_key:        Groq API key (reads GROQ_API_KEY env var if None).
            llama_cpp_host:      Hostname of the llama.cpp server (default: localhost).
            llama_cpp_port:      Port of the llama.cpp server (default: 8080).
        """
        self.llama_cpp_host = llama_cpp_host
        self.llama_cpp_port = llama_cpp_port
        self.model_name = model_name
        self.api_url = api_url
        self.ollama_process = None
        self.backend = backend
        self.device = device
        self.temperature = temperature
        self.groq_api_key = None
        self.selector = DynamicFewShotSelector(k=2)
        
        # Initialize HuggingFace attributes (will be set if using HF backend)
        self.hf_model = None
        self.hf_tokenizer = None
        
        # Groq token tracking (Free tier limits - model-specific)
        self.groq_tokens_used = 0
        self.groq_requests_count = 0
        self.groq_request_timestamps = []  # Track request times for RPM limiting
        self.groq_reset_time = datetime.now() + timedelta(days=1)
        
        # Set limits based on model (Free tier)
        if backend == "groq":
            # Check if using 70B model (stricter limits) or 8B model
            if "70b" in model_name.lower():
                self.groq_daily_limit = 1000  # 70B: 1,000 requests/day
                self.groq_tpm_limit = 6000    # 70B: 6,000 tokens/minute
                print(f"📊 Groq Free Tier (70B model): 1,000 requests/day, 6,000 tokens/min")
            else:
                self.groq_daily_limit = 14400  # 8B: 14,400 requests/day
                self.groq_tpm_limit = 6000     # 8B: 6,000 tokens/minute
                print(f"📊 Groq Free Tier (8B model): 14,400 requests/day, 6,000 tokens/min")
        else:
            self.groq_daily_limit = 14400
            self.groq_tpm_limit = 6000
        
        # Initialize EntityResolver if enabled
        self.use_entity_resolver = use_entity_resolver
        self.entity_resolver = None
        if use_entity_resolver and neo4j_driver:
            self.entity_resolver = EntityResolver(neo4j_driver, debug=entity_resolver_debug)
            print("✓ EntityResolver initialized")

        # Initialize Cypher validator
        self.validator = CypherValidator(neo4j_driver=neo4j_driver, block_writes=True)
        self.max_validation_retries = 3
        print("✓ CypherValidator initialized")

        # LangChain components
        self.llm = None
        self.chain = None
        self.llama_cpp_process = None  # managed llama.cpp subprocess (if launched by us)
        
        # Complete Neo4j schema
        self.optimized_schema = """
        // === GRAPH SCHEMA ===
        // 1. FUND MANAGEMENT & PROFILE
        (:Provider{name})-[:MANAGES]->(:Trust{name})-[:ISSUES]->(f:Fund{ticker,name,securityExchange,costsPer10k,advisoryFees,numberHoldings,expenseRatio,netAssets,turnoverRate})
        (f)-[:HAS_SHARE_CLASS]->(:ShareClass{name,description})
        (f)-[:DEFINED_BY{date}]->(p:Profile{summaryProspectus})-[:HAS_OBJECTIVE]->(:Objective{id,text,embedding})
        (p)-[:HAS_RISK_NODE]->(:RiskChunk{id,title,text,embedding})
        (p)-[:HAS_STRATEGY]->(:StrategyChunk{id,title,text,embedding})
        (p)-[:HAS_PERFORMANCE_COMMENTARY]->(:PerformanceCommentary{id,text,embedding})
        (f)-[:HAS_CHART{date}]->(:Image{id,title,category,svg})
        (f)-[:HAS_SECTOR_ALLOCATION{weight,date}]->(:Sector{name})
        (f)-[:HAS_REGION_ALLOCATION{weight,date}]->(:Region{name})

        // 2. PORTFOLIO & PERFORMANCE
        (f)-[:MANAGED_BY{date}]->(per:Person{name})
        (f)-[:HAS_PORTFOLIO]->(port:Portfolio{seriesId,date})-[:HAS_HOLDING{shares,marketValue,weight,fairValueLevel,isRestricted,payoffProfile}]->(h:Holding{name,ticker,isin,lei,category,category_desc,issuer_category,businessAddress})
        (f)-[:HAS_FINANCIAL_HIGHLIGHT{year}]->(:FinancialHighlight{turnover,expenseRatio,totalReturn,netAssets,netAssetsValueBeginning,netAssetsValueEnd,netIncomeRatio})
        (f)-[:HAS_TRAILING_PERFORMANCE{date}]->(:TrailingPerformance{return1y,return5y,return10y,returnInception})

        // 3. COMPANY, 10-K & FINANCIALS
        (h)-[:REPRESENTS]->(c:Company{ticker,name,cik})
        (c)-[:EMPLOYED_AS_CEO{ceoCompensation,ceoActuallyPaid,date}]->(per)
        (c)<-[:AWARDED_BY]-(comp:CompnesationPackage{totalCompensation,shareholderReturn,date})<-[:RECEIVED_COMPENSATION]-(per)
        (c)-[:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction{position,transactionType,shares,price,value,remainingShares})-[:MADE_BY]->(per)
        (c)-[:HAS_FILING{date}]->(filing:Filing10K)-[:HAS_RISK_FACTORS]->(:Section:RiskFactor{id,text,embedding})
        (filing)-[:HAS_BUSINESS_INFORMATION]->(:Section:BusinessInformation{id,text,embedding})
        (filing)-[:HAS_LEGAL_PROCEEDINGS]->(:Section:LegalProceeding{id,text,embedding})
        (filing)-[:HAS_MANAGEMENT_DISCUSSION]->(:Section:ManagemetDiscussion{id,text,embedding})
        (filing)-[:HAS_PROPERTIES_CHUNK]->(:Section:Properties{id,text,embedding})
        (filing)-[:HAS_FINACIALS]->(fin:Section:Financials{incomeStatement,balanceSheet,cashFlow,fiscalYeat})-[:HAS_METRIC]->(met:FinancialMetric{label,value})
        (met)-[:HAS_SEGMENT]->(:Segment{label,value,percentage})

        // 4. DOCUMENTS (Sources)
        (f)-[:EXTRACTED_FROM]->(d:Document{accessionNumber,url,type,filingDate,reportingDate})
        (p)-[:EXTRACTED_FROM]->(d)
        (port)-[:EXTRACTED_FROM]->(d)
        (filing)-[:EXTRACTED_FROM]->(d)
        (it)-[:DISCLOSED_IN]->(d)<-[:DISCLOSED_IN]-(comp)

        // === QUERY RULES ===
        1. INDEXES FIRST: For name searches use CALL db.index.fulltext.queryNodes('indexName', 'search_term') [providerNameIndex, trustNameIndex, fundNameIndex, personNameIndex].
        2. EXACT MATCH: For symbols use {ticker: 'VTI'} directly.
        3. Use f.numberHoldings to calculate the number of holdings a fund has.
        3. PROPERTY FILTERS: Never generate incomplete filters like (n:Label {name}). Only use explicit values e.g., (n:Label {name: 'Value'}). If value is unknown, omit the {} block.
        4. METRICS: netAssets is directly on Fund. numberHoldings is pre-calculated. turnoverRate is absolute (2 = 2%, not 0.02).
        5. ALWAYS return general information about the fund, ticker and name in the response.
    """

        
        self.schema = """
        # ============================================================
        # === FUND MANAGEMENT STRUCTURE ===
        # ============================================================
        (:Provider {name})-[:MANAGES]->(:Trust {name})-[:ISSUES]->(:Fund)
        # === FUND NODE PROPERTIES (Use these directly!) ===
        (:Fund {
            ticker,              # Symbol like 'VTI' (use for matching symbols)
            name,                # Full name like 'Vanguard Total Stock Market Index Fund'
            securityExchange,    # Exchange like 'NASDAQ', 'NYSE'
            costsPer10k,         # Costs per $10,000 invested (numeric)
            advisoryFees,        # Advisory fees (numeric)
            numberHoldings,      # Total number of holdings (integer) - USE THIS, not count(h)!
            expenseRatio,        # Expense ratio (numeric)
            netAssets,           # Net assets value (numeric) - DIRECTLY ON FUND NODE
            turnoverRate        # Portfolio Turnover Rate in ABSOLUTE terms (e.g., 2 means 2%, NOT 0.02) 
        })
        # FULL-TEXT INDEXES (use for fuzzy/partial name matching):
        # - Provider.name -> use CALL db.index.fulltext.queryNodes('providerNameIndex', 'search_term')
        # - Trust.name -> use CALL db.index.fulltext.queryNodes('trustNameIndex', 'search_term')
        # - Fund.name -> use CALL db.index.fulltext.queryNodes('fundNameIndex', 'search_term')
        # - Person.name -> use CALL db.index.fulltext.queryNodes('personNameIndex', 'search_term')
        # For exact ticker matching, use {ticker: 'VTI'} (no index needed)

        # === FUND RELATIONSHIPS ===
        # Share Classes
        (:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name, description})
        # Document node
        (:Fund)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})
        # Profile (versioned by date)
        (:Fund)-[:DEFINED_BY {date}]->(:Profile {id, summaryProspectus})
        (:Profile)-[:HAS_OBJECTIVE]->(:Objective {id, text, embedding})
        (:Profile)-[:HAS_PERFORMANCE_COMMENTARY]->(:PerformanceCommentary {id, text, embedding})
        (:Profile)-[:HAS_RISK_NODE]->(:RiskChunk {id, title, text, embedding})
        (:Profile)-[:HAS_STRATEGY]->(:StrategyChunk {id, title, text, embedding})
        (:Profile)-[:EXTRACTED_FROM]->(:Document)
        # Charts/Images (dated)
        (:Fund)-[:HAS_CHART {date}]->(:Image {id, title, category, svg})
        
        # Management Team
        (:Fund)-[:MANAGED_BY]->(:Person {name})
        
        # Allocations (by report date)
        (:Fund)-[h:HAS_SECTOR_ALLOCATION {weight, date}]->(:Sector {name})
        (:Fund)-[h:HAS_GEOGRAPHIC_ALLOCATION {weight, date}]->(:GeographicAllocation {name})
                    
        # Holdings Structure
        (:Fund)-[:HAS_PORTFOLIO]->(:Portfolio {id, ticker, date, count})
        (:Portfolio)-[:HAS_HOLDING {shares, marketValue, weight, currency, fairValueLevel, isRestricted, payoffProfile}]->(:Holding {
            id, name, ticker, cusip, isin, lei, country, sector, assetCategory, assetDesc, issuerCategory, issuerDesc})
        (:Portfolio)-[:EXTRACTED_FROM]->(:Document)
        # Financial Highlights
        (:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT {year}]->(:FinancialHighlight {turnover, expenseRatio, totalReturn, netAssets, netAssetsValueBeginning, netAssetsValueEnd, netIncomeRatio})

        # Trailing Performance
        (:Fund)-[:HAS_TRAILING_PERFORMANCE {date}]->(:TrailingPerformance {return1y, return5y, return10y, returnInception})

        # ============================================================
        # === COMPANY STRUCTURE (10-K Filings) ===
        # ============================================================
        
        # === COMPANY NODE PROPERTIES ===
        (:Company {
            ticker,              # Stock ticker symbol like 'AAPL', 'MSFT'
            name,                # Company name like 'Apple Inc.'
            cik                  # SEC Central Index Key
        })
        # === COMPANY RELATIONSHIPS ===
        # Document and Filing Structure
        # Funds can hold Company stocks
        (:Holding {ticker})-[:REPRESENTS]->(:Company {ticker})
        
        (:Company)-[:HAS_FILING]->(:Filing10K {id, filingDate, reportPeriodEnd, url})
        (:Filing10K)-[:EXTRACTED_FROM]->(:Document {accessionNumber, form, url, filingDate, reportingDate})
        
        # 10-K Section Types (all inherit from :Section base label)
        (:Filing10K)-[:HAS_RISK_FACTORS]->(:Section:RiskFactor {id, text, embedding})
        (:Filing10K)-[:HAS_BUSINESS_INFORMATION]->(:Section:BusinessInformation {id, text, embedding})
        (:Filing10K)-[:HAS_LEGAL_PROCEEDINGS]->(:Section:LegalProceeding {id, text, embedding})
        (:Filing10K)-[:HAS_MANAGEMENT_DISCUSSION]->(:Section:ManagemetDiscussion {id, text, embedding})
        (:Filing10K)-[:HAS_PROPERTIES_CHUNK]->(:Section:Properties {id, text, embedding})
        (:Filing10K)-[:HAS_FINANCIALS]->(:Section:Financials {id})
        
        # === COMPANY PEOPLE & COMPENSATION ===
        (:Company)-[:HAS_CEO]->(:Person {name})
        (:Person)-[:RECEIVED]->(:CompensationPackage {
            id, year, totalCompensation, salary, bonus, stockAwards, 
            optionAwards, nonEquityIncentive, changeInPensionValue, otherCompensation
        })
        (:Company)-[:HAS_INSIDER_TRANSACTION]->(:InsiderTransaction {
            id, date, transactionType, shares, pricePerShare, totalValue, 
            sharesOwnedAfter, directOrIndirect
        })
        (:InsiderTransaction)-[:MADE_BY]->(:Person {name})
        
        # === FINANCIAL METRICS & SEGMENTS ===
        (:Section:Financials)-[:HAS_METRIC]->(:FinancialMetric {
            id, label, value, unit, period, fiscalYear, fiscalQuarter
        })
        (:Section:Financials)-[:HAS_SEGMENT]->(:Segment {
            id, name, revenue, operatingIncome, assets
        })

        # === IMPORTANT NOTES ===
        # 1. netAssets is DIRECTLY on Fund node, not in a separate FinancialHighlight node
        # 2. numberHoldings property already contains the count - don't recalculate!
        # 3. turnoverRate is absolute (2 = 2%, not 0.02)
        # 4. Use 'ticker' for symbols (VTI), 'name' for full names
        # 5. Vector indexes exist on embedding properties for semantic search
        # 6. Fulltext indexes exist on name properties for fuzzy/partial name matchings USE THEM before the MATCH
        # 7. NEVER generate incomplete property filters like (n:Label {name}). Only use property filters with values like (n:Label {name: 'Value'}).
        # 8. If you do not know the value of a property, DO NOT include it in the curly braces {}.
        # 9. ALWAYS return general information about the fund, ticker and name in the response.
        # 10. Whenever it exists, return the document information from which it has been extracted.
        """
        
        # Define the prompt template
        self.prompt_template = """You are a Neo4j Cypher expert. Convert the natural language question to a valid Cypher query.

Neo4j Schema:
{schema}

Examples:
{examples}

{entity_context}

Rules:
1. Output ONLY the Cypher query, no explanations or markdown
2. Use proper Cypher syntax with MATCH, WHERE, RETURN, COUNT(*)
3. Use property names exactly as shown in schema
4. For numeric comparisons, use appropriate operators (>, <, =, etc.) just after the MATCH clause
5. For text search, use CONTAINS or regular expressions
6. Always return more properties than the ones asked. Include ticker, name, etc.
7. If entity names are provided in the Entity Context, use them EXACTLY as shown

Question: {question}

Cypher Query:"""
        
        self.prompt = PromptTemplate(
            input_variables=["schema", "examples", "entity_context", "question"],
            template=self.prompt_template
        )
        
        # Retry prompt for when validation fails
        self.retry_prompt_template = """You are a Neo4j Cypher expert. Your previous Cypher query had errors. Fix them.

Neo4j Schema:
{schema}

{entity_context}

Original Question: {question}

Your Previous (Invalid) Query:
{failed_query}

Validation Errors:
{validation_errors}

Rules:
1. Output ONLY the corrected Cypher query, no explanations or markdown
2. Fix ALL the validation errors listed above
3. Use ONLY node labels, relationship types, and property names that exist in the schema
4. Use proper Cypher syntax
5. Do NOT use MERGE, CREATE, SET, DELETE, or any write operations

Corrected Cypher Query:"""
        
        self.retry_prompt = PromptTemplate(
            input_variables=["schema", "entity_context", "question", "failed_query", "validation_errors"],
            template=self.retry_prompt_template
        )
        
        # Initialize based on backend
        if self.backend == "ollama":
            if auto_start_ollama:
                self._ensure_ollama_running()
            self._init_ollama_chain()
        elif self.backend == "huggingface":
            self._init_huggingface_chain()
        elif self.backend == "groq":
            self._init_groq_chain()
        elif self.backend == "openai":
            self._init_openai_chain()
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Use 'ollama', 'huggingface', 'groq', or 'openai'."
            )

        self._warm_up()

    def _warm_up(self):
        """
        Sends a lightweight dummy request to force the LLM to load into VRAM.
        This ensures the first real user query doesn't lag.
        """
        if self.backend in ["openai", "groq"]:
            return
            
        print("🔥 Warming up LLM (loading into memory)...")
        try:
            # We send a trivial request. The answer doesn't matter.
            self.chain.invoke({
                "schema": "",  # Empty schema is fine for warmup
                "examples": "",
                "question": "hi",
                "entity_context": ""
            })
            print("✓ LLM Warmed up and ready.")
        except Exception as e:
            # If warmup fails, we log it but don't crash the app
            print(f"⚠ Warm-up failed (non-critical): {e}")
    
    def _init_ollama_chain(self):
        """Initialize LangChain with Ollama backend."""
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.api_url,
                temperature=self.temperature,
                num_predict=512
            )
            
            # Build the chain: Prompt -> LLM -> Output Parser
            self.chain = self.prompt | self.llm | StrOutputParser()
            
            print(f"✓ Ollama LangChain initialized with model: {self.model_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama chain: {e}") from e
    
    def _init_huggingface_chain(self):
        """Initialize LangChain with HuggingFace backend."""
        try:
            # Import torch locally to avoid circular import issues
            import torch
            from unsloth import FastLanguageModel
            from transformers import pipeline
            
            print(f"Loading HuggingFace model: {self.model_name}")
            
            # Use Unsloth's FastLanguageModel
            max_seq_length = 2048
            dtype = None  # Auto-detect
            load_in_4bit = True  # Use 4bit quantization for efficiency
            
            self.hf_model, self.hf_tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
            
            # Enable inference mode (faster)
            FastLanguageModel.for_inference(self.hf_model)
            
            # Check CUDA availability
            use_cuda = self.device == "cuda" and torch.cuda.is_available()
            device_id = 0 if use_cuda else -1
            
            # Create HuggingFace pipeline
            hf_pipeline = pipeline(
                "text-generation",
                model=self.hf_model,
                tokenizer=self.hf_tokenizer,
                max_new_tokens=256,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                device=device_id
            )
            
            # Wrap in LangChain HuggingFacePipeline
            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
            
            # Build the chain
            self.chain = self.prompt | self.llm | StrOutputParser()
            
            # Check device and report
            if use_cuda:
                print(f"✓ HuggingFace model loaded on GPU with 4-bit quantization")
            else:
                print(f"✓ HuggingFace model loaded on CPU")
                
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend requires 'unsloth', 'transformers', and 'langchain-huggingface'. "
                "Install with: pip install unsloth transformers langchain-huggingface"
            ) from e
        except AttributeError as e:
            if "torch" in str(e) and "circular import" in str(e):
                raise RuntimeError(
                    "PyTorch circular import detected. This usually happens when:\n"
                    "1. PyTorch version is incompatible with other packages\n"
                    "2. Multiple PyTorch installations exist\n"
                    "3. Environment has conflicting packages\n\n"
                    "Try:\n"
                    "- pip install torch --upgrade --force-reinstall\n"
                    "- pip install unsloth transformers langchain-huggingface --upgrade\n"
                    "- Or use a different backend (backend='ollama' or backend='groq')"
                ) from e
            else:
                raise RuntimeError(f"Failed to initialize HuggingFace chain: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace chain: {e}") from e
    
    def _init_groq_chain(self):
        """Initialize LangChain with Groq backend."""
        try:
            # Initialize Groq wrapper
            groq_wrapper = GroqWrapper(
                model_name=self.model_name,
                api_key=self.groq_api_key,
                temperature=self.temperature,
                max_tokens=512
            )
            
            # Get the LangChain-compatible LLM
            self.llm = groq_wrapper.get_llm()
            
            # Build the chain: Prompt -> LLM -> Output Parser
            self.chain = self.prompt | self.llm | StrOutputParser()
            
            print(f"✓ Groq LangChain initialized with model: {self.model_name}")
            
        except ImportError as e:
            raise ImportError(
                "Groq backend requires 'langchain-groq'. "
                "Install with: pip install langchain-groq"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq chain: {e}") from e

    def _init_openai_chain(self):
        """Initialize LangChain with an OpenAI-compatible local llama.cpp server."""
        try:
            base_url = f"http://{self.llama_cpp_host}:{self.llama_cpp_port}/v1"
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_base=base_url,
                openai_api_key="llama-cpp",   # required by library; ignored by llama.cpp
                temperature=self.temperature,
                max_tokens=512,
            )
            self.chain = self.prompt | self.llm | StrOutputParser()
            print(f"✓ llama.cpp OpenAI-compatible backend → {base_url}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize llama.cpp (openai) chain: {e}") from e

    def _is_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def _start_ollama_server(self) -> bool:
        """Start Ollama server in a subprocess."""
        try:
            print("Starting Ollama server...")
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Wait for server to be ready (max 30 seconds)
            max_wait = 30
            wait_interval = 1
            elapsed = 0
            
            while elapsed < max_wait:
                if self._is_ollama_running():
                    print(f"✓ Ollama server started successfully (took {elapsed}s)")
                    return True
                time.sleep(wait_interval)
                elapsed += wait_interval
            
            print("⚠ Ollama server started but not responding yet")
            return False
            
        except FileNotFoundError:
            print("✗ Ollama command not found. Please install Ollama first.")
            print("  Visit: https://ollama.ai/download")
            return False
        except Exception as e:
            print(f"✗ Failed to start Ollama server: {e}")
            return False
    
    def _ensure_ollama_running(self) -> bool:
        """Ensure Ollama server is running, start it if not."""
        if self._is_ollama_running():
            print("✓ Ollama server is already running")
            return True
        
        print("Ollama server not detected, attempting to start...")
        return self._start_ollama_server()
    
    def _check_groq_rate_limits(self) -> bool:
        """
        Check if we're within Groq API rate limits.
        Returns True if safe to proceed, False if limits exceeded.
        
        Free Tier Limits:
        - Llama 3.1 8B: 14,400 requests/day, 6,000 tokens/min
        - Llama 3.3 70B: 1,000 requests/day, 6,000 tokens/min
        """
        if self.backend != "groq":
            return True
        
        now = datetime.now()
        
        # Check daily reset
        if now >= self.groq_reset_time:
            self.groq_tokens_used = 0
            self.groq_requests_count = 0
            self.groq_request_timestamps = []
            self.groq_reset_time = now + timedelta(days=1)
            print("🔄 Groq counters reset (new day)")
        
        # Check daily REQUEST limit (not token limit)
        if self.groq_requests_count >= self.groq_daily_limit:
            remaining_time = self.groq_reset_time - now
            hours = remaining_time.seconds // 3600
            minutes = (remaining_time.seconds % 3600) // 60
            print(f"⚠️  Groq daily request limit reached ({self.groq_requests_count}/{self.groq_daily_limit})")
            print(f"   Resets in {hours}h {minutes}m")
            return False
        
        # Check TPM (tokens per minute) limit
        one_minute_ago = now - timedelta(minutes=1)
        self.groq_request_timestamps = [ts for ts in self.groq_request_timestamps if ts > one_minute_ago]
        
        # Calculate tokens used in the last minute
        # Note: We'll track this after the API call, so this is a pre-check based on previous minute
        # For now, we just warn if we're close to the limit
        
        return True
    
    def _update_groq_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """Update Groq token usage tracking."""
        if self.backend != "groq":
            return
        
        total_tokens = prompt_tokens + completion_tokens
        self.groq_tokens_used += total_tokens
        self.groq_requests_count += 1
        self.groq_request_timestamps.append(datetime.now())
        
        # Calculate percentage used (based on requests, not tokens)
        usage_percent = (self.groq_requests_count / self.groq_daily_limit) * 100
        
        print(f"📊 Groq Usage: {self.groq_requests_count}/{self.groq_daily_limit} requests ({usage_percent:.1f}%) | "
              f"Tokens: {self.groq_tokens_used} total | "
              f"This call: {total_tokens} tokens (↑{prompt_tokens} ↓{completion_tokens})")
    
    def get_groq_usage_stats(self) -> dict:
        """Get current Groq usage statistics."""
        if self.backend != "groq":
            return {"error": "Not using Groq backend"}
        
        remaining_requests = self.groq_daily_limit - self.groq_requests_count
        usage_percent = (self.groq_requests_count / self.groq_daily_limit) * 100
        time_until_reset = self.groq_reset_time - datetime.now()
        
        return {
            "requests_used": self.groq_requests_count,
            "requests_remaining": remaining_requests,
            "daily_request_limit": self.groq_daily_limit,
            "tokens_used": self.groq_tokens_used,
            "tpm_limit": self.groq_tpm_limit,
            "usage_percent": round(usage_percent, 2),
            "reset_in_hours": round(time_until_reset.seconds / 3600, 2),
            "model": self.model_name
        }
    
    def reset_groq_usage(self):
        """Manually reset Groq usage counters (for testing or new day)."""
        if self.backend == "groq":
            self.groq_tokens_used = 0
            self.groq_requests_count = 0
            self.groq_request_timestamps = []
            self.groq_reset_time = datetime.now() + timedelta(days=1)
            print("✓ Groq usage counters reset")
    
    def _invoke_llm(self, prompt_text: str) -> str:
        """
        Invoke the LLM with a pre-formatted prompt string.
        Handles Groq token tracking and backend differences.
        
        Args:
            prompt_text: The fully formatted prompt to send to the LLM
            
        Returns:
            Raw LLM response string
        """
        if self.backend == "groq":
            from langchain_core.messages import HumanMessage
            
            llm_response = self.llm.invoke([HumanMessage(content=prompt_text)])
            response = llm_response.content
            
            # Track token usage
            if hasattr(llm_response, 'response_metadata') and 'token_usage' in llm_response.response_metadata:
                token_usage = llm_response.response_metadata['token_usage']
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)
                self._update_groq_token_usage(prompt_tokens, completion_tokens)
            else:
                estimated_prompt = len(prompt_text.split()) * 1.3
                estimated_completion = len(response.split()) * 1.3
                self._update_groq_token_usage(int(estimated_prompt), int(estimated_completion))
                print("⚠️  Token usage estimated (metadata not available)")
            
            return response
        else:
            # For ollama/huggingface, use the chain
            # We build a minimal dict — but since we have a pre-formatted prompt,
            # we invoke the LLM directly
            from langchain_core.messages import HumanMessage
            llm_response = self.llm.invoke([HumanMessage(content=prompt_text)])
            if hasattr(llm_response, 'content'):
                return llm_response.content
            return str(llm_response)

    def translate(self, user_query: str, temperature: Optional[float] = None) -> Optional[str]:
        """
        Translate natural language query to Cypher query using LangChain.
        
        Includes a validation loop: if the generated Cypher fails validation
        (syntax or schema errors), the error feedback is sent back to the LLM
        for up to 3 attempts to produce a valid query.
        
        Args:
            user_query: Natural language question
            temperature: Override temperature for this query (optional)
            
        Returns:
            Cypher query string or None if error
        """
        if self.chain is None:
            raise RuntimeError("LangChain not initialized. Check backend configuration.")
        
        # Check Groq rate limits before proceeding
        if self.backend == "groq":
            if not self._check_groq_rate_limits():
                print("❌ Cannot proceed: Groq rate limits exceeded")
                return None
        
        # Step 1: Resolve entities in the query
        entity_context = ""
        processed_query = user_query
        
        if self.entity_resolver:
            
            resolved_entities = self.entity_resolver.extract_entities(user_query)
            print("Resolved Entities: ", resolved_entities)
            if resolved_entities:
                # Build entity context with explicit instructions
                entity_context = "RESOLVED ENTITIES (Use these EXACT names in your Cypher query):\n"
                
                # Sort by entity type for consistent output
                entity_list = []
                for entity_name, entity_type in resolved_entities.items():
                    entity_list.append((entity_type, entity_name))
                    # Replace entity in query (case-insensitive)
                    import re
                    pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
                    processed_query = pattern.sub(f"'{entity_name}'", processed_query, count=1)
                
                # Add to context
                # Sort by string representation to handle non-comparable types (e.g., dicts)
                for entity_type, entity_name in sorted(entity_list, key=lambda x: (str(x[0]), str(x[1]))):
                    entity_context += f"  - {entity_type}: {entity_name}\n"
                
               
                entity_context += f"Processed Question: {processed_query}\n"
                entity_context += "\nIMPORTANT: Use the exact entity names shown above in your Cypher query.\n"
                
                # Compact entity resolution output
                # Sort by string representation to handle non-comparable types (e.g., dicts)
                entity_summary = ", ".join([f"{t}:{n}" for t, n in sorted(entity_list, key=lambda x: (str(x[0]), str(x[1])))])
                print(f"🔍 Entities: {entity_summary}")
        
        # Step 2: Get few-shot examples
        examples_str = self.selector.get_formatted_context(user_query)
        n_examples = len(examples_str.split('Example'))-1
        print(f"📚 Examples: {n_examples} retrieved")
        if examples_str.strip():
            print("--- Retrieved Examples ---")
            print(examples_str.strip())
            print("--------------------------")
        
        try:
            # Update temperature if provided
            if temperature is not None and self.backend in ["ollama", "groq"]:
                self.llm.temperature = temperature
            
            # Step 3: Initial LLM invocation
            full_prompt = self.prompt.format(
                schema=self.schema,
                examples=examples_str,
                entity_context=entity_context,
                question=processed_query
            )
            
            response = self._invoke_llm(full_prompt)
            cypher_query = self._clean_cypher(response)
            
            if not cypher_query:
                print("⚠ Empty Cypher query generated")
                return None
            
            print(f"✓ Generated Cypher (attempt 1): {cypher_query}")
            
            # Step 4: Validate + retry loop
            for attempt in range(1, self.max_validation_retries + 1):
                validation_result = self.validator.validate(cypher_query)
                
                if validation_result.is_valid:
                    print(f"✅ Validation passed (attempt {attempt})")
                    break
                
                # Validation failed
                error_summary = "\n".join(
                    [f"  - [SYNTAX] {e}" for e in validation_result.syntax_errors] +
                    [f"  - [SCHEMA] {e}" for e in validation_result.schema_errors]
                )
                print(f"⚠️  Validation failed (attempt {attempt}/{self.max_validation_retries}):")
                print(error_summary)
                
                # If we've exhausted retries, return the last query anyway
                if attempt >= self.max_validation_retries:
                    print(f"❌ Max validation retries ({self.max_validation_retries}) reached. Returning last query.")
                    break
                
                # Check Groq rate limits before retrying
                if self.backend == "groq" and not self._check_groq_rate_limits():
                    print("❌ Cannot retry: Groq rate limits exceeded")
                    break
                
                # Build retry prompt with error feedback
                retry_prompt_text = self.retry_prompt.format(
                    schema=self.schema,
                    entity_context=entity_context,
                    question=processed_query,
                    failed_query=cypher_query,
                    validation_errors=error_summary
                )
                
                print(f"🔄 Retrying with error feedback (attempt {attempt + 1})...")
                retry_response = self._invoke_llm(retry_prompt_text)
                cypher_query = self._clean_cypher(retry_response)
                
                if not cypher_query:
                    print("⚠ Empty Cypher query on retry")
                    break
                
                print(f"✓ Generated Cypher (attempt {attempt + 1}): {cypher_query}")
            
            # Restore original temperature
            if temperature is not None and self.backend in ["ollama", "groq"]:
                self.llm.temperature = self.temperature
            
            return cypher_query
            
        except Exception as e:
            print(f"✗ Error during translation: {e}")
            logger.error(f"Translation error: {e}", exc_info=True)
            return None
    
    
    def _clean_cypher(self, query: str) -> str:
        """Clean up generated Cypher query."""
        query = query.strip()
        
        # Remove markdown code blocks
        if query.startswith("```"):
            lines = query.split("\n")
            query = "\n".join(lines[1:-1]) if len(lines) > 2 else query
        
        query = query.replace("```cypher", "").replace("```", "").strip()
        
        # Remove any leading/trailing quotes
        query = query.strip('"\'')
        
        # Remove common prefixes
        prefixes = ["cypher:", "Cypher:", "CYPHER:", "Query:", "query:"]
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
        
        # === NEW FIX: Repair Syntax Errors ===
        
        # 1. Fix "Property without value" error: (n:Label {prop}) -> (n:Label)
        # Matches {word} that does NOT have a colon inside
        # This turns (t:Trust {name}) into (t:Trust )
        query = re.sub(r'\{\s*[a-zA-Z0-9_]+\s*\}', '', query)
        
        # 2. Fix empty braces error: (n:Label {}) -> (n:Label)
        query = re.sub(r'\{\s*\}', '', query)
        
        # 3. Clean up accidental double spaces created by the removal
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def stop_ollama_server(self):
        """Stop the Ollama server if it was started by this instance. No-op for other backends."""
        if self.backend != "ollama":
            return
        if self.ollama_process:
            print("Stopping Ollama server...")
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=5)
                print("✓ Ollama server stopped")
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
                print("✓ Ollama server force stopped")
            self.ollama_process = None
    
    def __del__(self):
        """Cleanup on deletion."""
        # Stop Ollama server if it was started
        if hasattr(self, 'ollama_process'):
            self.stop_ollama_server()
        
        # Clear GPU memory if using HuggingFace
        if hasattr(self, 'hf_model') and self.hf_model is not None:
            try:
                import torch
                del self.hf_model
                if hasattr(self, 'hf_tokenizer'):
                    del self.hf_tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                # Silently ignore cleanup errors
                pass