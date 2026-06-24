import os
import subprocess
import time
import requests
import re
import logging
import warnings
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
from simple_rag.rag.post_processing.cypher_validator import CypherValidator, ValidationResult
from simple_rag.rag.schema_definitions import DETAILED_SCHEMA, DETAILED_SCHEMA_V2
from simple_rag.rag.prompt_templates import CYPHER_GENERATION_TEMPLATE, CYPHER_RETRY_TEMPLATE, CYPHER_RETRY_TEMPLATE_LEAN, CYPHER_GENERATION_TEMPLATE2

logger = logging.getLogger(__name__)

# ============================================================
# === CONFIGURATION CONSTANTS ===
# ============================================================

# LLM Configuration
DEFAULT_MAX_TOKENS = 512
DEFAULT_NUM_PREDICT = 512
DEFAULT_VALIDATION_RETRIES = 3
DEFAULT_TEMPERATURE = 0.2

# Groq Rate Limits
GROQ_70B_DAILY_LIMIT = 1000
GROQ_8B_DAILY_LIMIT = 14400
GROQ_TPM_LIMIT = 6000

# Ollama Configuration
OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_MAX_STARTUP_WAIT = 30

# OpenAI-compatible Configuration
OPENAI_COMPATIBLE_DEFAULT_HOST = "localhost"
OPENAI_COMPATIBLE_DEFAULT_PORT = 1234  # LM Studio default

class CypherTranslator:
    def __init__(
        self,
        neo4j_driver,
        model_name: str = "llama3.1:8b",
        api_url: str = OLLAMA_DEFAULT_URL,
        auto_start_ollama: bool = True,
        backend: Literal["ollama", "huggingface", "groq", "openai", "openrouter"] = "ollama",
        device: str = "cuda",
        temperature: float = DEFAULT_TEMPERATURE,
        use_entity_resolver: bool = True,
        entity_resolver_debug: bool = False,
        groq_api_key: Optional[str] = None,
        openai_compatible_host: str = OPENAI_COMPATIBLE_DEFAULT_HOST,
        openai_compatible_port: int = OPENAI_COMPATIBLE_DEFAULT_PORT,
        ctx_size: int = 0,
        max_validation_retries: int = DEFAULT_VALIDATION_RETRIES,
        llama_cpp_host: Optional[str] = None,
        llama_cpp_port: Optional[int] = None,
        few_shot_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        retry_strategy: str = "full",
        use_few_shot: bool = True,
        use_groq_fallback: bool = True,
        openrouter_fallback_model: Optional[str] = None,
        main_llm_model_openai: Optional[str] = None,
        use_cypher_validator: bool = True,
        prompt_template_version: str = "v2",
    ):
        """
        LangChain-based LLM wrapper for text-to-Cypher translation.

        Args:
            neo4j_driver:        Neo4j driver instance for entity resolution.
            model_name:          Model identifier (Ollama model, HuggingFace path, Groq model,
                                 or any string for OpenAI-compatible servers).
            api_url:             Ollama base URL (only used when backend="ollama").
            auto_start_ollama:   Auto-start Ollama if not running (ignored for other backends).
            backend:             "ollama" | "huggingface" | "groq" | "openai".
                                 Use "openai" for LM Studio, llama.cpp, vLLM, or any OpenAI-compatible server.
            device:              Device for HuggingFace model ("cuda" or "cpu").
            temperature:         Sampling temperature (lower = more deterministic).
            use_entity_resolver: Enable EntityResolver for name/ticker pre-resolution.
            entity_resolver_debug: Enable debug output for entity resolution.
            groq_api_key:        Groq API key (reads GROQ_API_KEY env var if None).
            openai_compatible_host: Hostname for OpenAI-compatible server (LM Studio, llama.cpp, etc.).
            openai_compatible_port: Port for OpenAI-compatible server (default: 1234 for LM Studio).
            max_validation_retries: Maximum number of validation retry attempts (default: 3).
            llama_cpp_host:      DEPRECATED: Use openai_compatible_host instead.
            llama_cpp_port:      DEPRECATED: Use openai_compatible_port instead.
        """
        # Handle deprecated parameters
        if llama_cpp_host is not None:
            warnings.warn(
                "Parameter 'llama_cpp_host' is deprecated and will be removed in a future version. "
                "Use 'openai_compatible_host' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            openai_compatible_host = llama_cpp_host
        if llama_cpp_port is not None:
            warnings.warn(
                "Parameter 'llama_cpp_port' is deprecated and will be removed in a future version. "
                "Use 'openai_compatible_port' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            openai_compatible_port = llama_cpp_port
        
        self.openai_compatible_host = openai_compatible_host
        self.openai_compatible_port = openai_compatible_port
        self.ctx_size = ctx_size
        self.max_validation_retries = max_validation_retries
        self.use_cypher_validator = use_cypher_validator
        self.model_name = model_name
        self.api_url = api_url
        self.ollama_process = None
        self.backend = backend
        self.device = device
        self.temperature = temperature
        self.groq_api_key = None
        self.use_few_shot = use_few_shot
        self.selector = DynamicFewShotSelector(k=3, embedding_model=few_shot_embedding_model)
        
        # Initialize HuggingFace attributes 
        self.hf_model = None
        self.hf_tokenizer = None
        
        # Groq token tracking 
        self.groq_tokens_used = 0
        self.groq_requests_count = 0
        self.groq_request_timestamps = []  # Track request times for RPM limiting
        self.groq_reset_time = datetime.now() + timedelta(days=1)
        
        # Set limits based on model 
        if backend == "groq":
            # Check if using 70B model 8B model
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
        print("✓ CypherValidator initialized")

     
        self.fallback_llm = None
        self._fallback_label = "fallback"  # display name used in log messages
        if openrouter_fallback_model:
            _or_key = os.environ.get("OPEN_ROUTER_API_KEY")
            if _or_key:
                try:
                    self.fallback_llm = ChatOpenAI(
                        model=openrouter_fallback_model,
                        api_key=_or_key,
                        base_url="https://openrouter.ai/api/v1",
                        temperature=0.1,
                        max_tokens=512,
                        default_headers={
                            "HTTP-Referer": "https://github.com/sec-filings-intelligence",
                            "X-Title": "SEC Filings Intelligence",
                        },
                    )
                    self._fallback_label = f"OpenRouter ({openrouter_fallback_model})"
                    print(f"✓ OpenRouter fallback initialized: {openrouter_fallback_model} (last-resort after retries)")
                except Exception as e:
                    print(f"⚠ OpenRouter fallback unavailable: {e}")
            else:
                print("⚠ OPEN_ROUTER_API_KEY not set — OpenRouter fallback disabled")
        elif main_llm_model_openai:
            # Second local model on the same OpenAI-compatible server (e.g. LM Studio multi-model)
            try:
                _fallback_base_url = f"http://{openai_compatible_host}:{openai_compatible_port}/v1"
                self.fallback_llm = ChatOpenAI(
                    model=main_llm_model_openai,
                    base_url=_fallback_base_url,
                    api_key="not-needed",
                    temperature=0.1,
                    max_tokens=DEFAULT_MAX_TOKENS,
                )
                self._fallback_label = f"local:{main_llm_model_openai}"
                print(f"✓ Local fallback initialized: {main_llm_model_openai} @ {_fallback_base_url}")
            except Exception as e:
                print(f"⚠ Local fallback unavailable: {e}")
        elif use_groq_fallback:
            _fallback_key = os.environ.get("GROQ_API_KEY")
            _is_already_groq70b = (
                backend == "groq" and "70b" in model_name.lower()
            )
            if _fallback_key and not _is_already_groq70b:
                try:
                    from langchain_groq import ChatGroq
                    self.fallback_llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        api_key=_fallback_key,
                        temperature=0.1,
                        max_tokens=512,
                    )
                    self._fallback_label = "Groq 70B"
                    print("✓ Groq 70B fallback initialized (last-resort after retries)")
                except Exception as e:
                    print(f"⚠ Groq 70B fallback unavailable: {e}")
            elif _is_already_groq70b:
                print("ℹ  Primary model is already Groq 70B — fallback disabled")
            else:
                print("⚠ GROQ_API_KEY not set — Groq 70B fallback disabled")

        # LangChain components
        self.llm = None
        self.chain = None
        self.llama_cpp_process = None  # managed llama.cpp subprocess (if launched by us)
        
        self.detailed_schema = DETAILED_SCHEMA  # always the full schema — never overwritten by slicing
        self.schema = DETAILED_SCHEMA           # may be temporarily replaced by a schema slice
        self.last_initial_prompt: Optional[str] = None  # prompt used on the first LLM call (captured for debugging)
        self.last_retry_prompts: list = []             # prompts used on each retry attempt (only populated when _capture_prompts=True)
        self._capture_prompts: bool = False            # set True by benchmark.debug() only — keeps run() overhead-free
        self.last_validation_failures: list = []  # validator rule tags that fired during the last translate() call
        self.last_write_blocked_op: Optional[str] = None  # set when a write op is blocked mid-retry
        self.last_total_prompt_tokens: int = 0  # cumulative token estimate across all LLM calls (initial + retries)
        self.last_completion_tokens: int = 0    # estimated tokens in all LLM responses for this translate() call
        self.last_schema_tokens: int = 0        # estimated tokens consumed by the schema slice in the initial prompt
        
        # Use imported prompt templates
        _template = CYPHER_GENERATION_TEMPLATE if prompt_template_version == "v1" else CYPHER_GENERATION_TEMPLATE2
        self.prompt = PromptTemplate(
            input_variables=["schema", "examples", "entity_context", "question"],
            template=_template,
        )
        print(f"✓ Prompt template: {prompt_template_version}")
        
        self.retry_strategy = retry_strategy
        if retry_strategy == "lean":
            self.retry_prompt = PromptTemplate(
                input_variables=["examples", "failed_query", "validation_errors", "error_examples"],
                template=CYPHER_RETRY_TEMPLATE_LEAN,
            )
            print(f"✓ Retry strategy: lean (no schema, ~400 tokens per retry)")
        else:
            self.retry_prompt = PromptTemplate(
                input_variables=["schema", "entity_context", "examples", "question", "failed_query", "validation_errors", "error_examples"],
                template=CYPHER_RETRY_TEMPLATE,
            )
            print(f"✓ Retry strategy: full (schema injected, ~3k tokens per retry)")
        
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
        elif self.backend == "openrouter":
            self._init_openrouter_chain()
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
        if self.backend in ["openai", "groq", "openrouter"]:
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
                num_predict=DEFAULT_NUM_PREDICT,
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
        """Initialize LangChain with an OpenAI-compatible server (llama.cpp, vLLM, etc.)."""
        try:
            base_url = f"http://{self.openai_compatible_host}:{self.openai_compatible_port}/v1"
            # num_ctx is a llama.cpp per-request extension; ignored by other servers
            model_kwargs = {"num_ctx": self.ctx_size} if self.ctx_size > 0 else {}
            self.llm = ChatOpenAI(
                model=self.model_name,
                base_url=base_url,
                api_key="not-needed",
                temperature=self.temperature,
                max_tokens=DEFAULT_MAX_TOKENS,
                model_kwargs=model_kwargs,
            )
            self.chain = self.prompt | self.llm | StrOutputParser()
            ctx_info = f", ctx_size={self.ctx_size}" if self.ctx_size > 0 else ""
            print(f"✓ OpenAI-compatible backend initialized → {base_url}{ctx_info}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Failed to connect to OpenAI-compatible server at {base_url}\n"
                f"Please ensure LM Studio, llama.cpp, or another OpenAI-compatible server is running on port {self.openai_compatible_port}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI-compatible chain: {e}") from e

    def _init_openrouter_chain(self):
        """Initialize LangChain with OpenRouter as the primary text2cypher backend."""
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPEN_ROUTER_API_KEY environment variable not set.")
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=self.temperature,
                max_tokens=DEFAULT_MAX_TOKENS,
                default_headers={
                    "HTTP-Referer": "https://github.com/sec-filings-intelligence",
                    "X-Title": "SEC Filings Intelligence",
                },
            )
            self.chain = self.prompt | self.llm | StrOutputParser()
            print(f"✓ OpenRouter backend initialized → {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenRouter chain: {e}") from e

    def test_connection(self) -> bool:
        """
        Test connection to the configured backend.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            if self.backend == "ollama":
                return self._is_ollama_running()
            elif self.backend == "openai":
                base_url = f"http://{self.openai_compatible_host}:{self.openai_compatible_port}"
                response = requests.get(f"{base_url}/v1/models", timeout=2)
                if response.status_code == 200:
                    print(f"✓ OpenAI-compatible server is running at {base_url}")
                    return True
                else:
                    print(f"✗ OpenAI-compatible server returned status {response.status_code}")
                    return False
            elif self.backend in ("groq", "openrouter"):
                print(f"✓ {self.backend.capitalize()} backend configured (API-based, no connection test needed)")
                return True
            elif self.backend == "huggingface":
                print("✓ HuggingFace backend configured (local model, no connection test needed)")
                return True
            else:
                print(f"⚠ Unknown backend: {self.backend}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Failed to connect to {self.backend} backend")
            if self.backend == "openai":
                print(f"  Ensure LM Studio or llama.cpp is running on port {self.openai_compatible_port}")
            return False
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False
    
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
            
            # Wait for server to be ready
            max_wait = OLLAMA_MAX_STARTUP_WAIT
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
    
    def _update_groq_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
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
    
    def reset_groq_usage(self) -> None:
        """Manually reset Groq usage counters (for testing or new day)."""
        if self.backend == "groq":
            self.groq_tokens_used = 0
            self.groq_requests_count = 0
            self.groq_request_timestamps = []
            self.groq_reset_time = datetime.now() + timedelta(days=1)
            print("✓ Groq usage counters reset")
    
    def _invoke_with_llm(self, llm, prompt_text: str) -> str:
        """Invoke an arbitrary LangChain LLM instance with a pre-formatted prompt."""
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt_text)])
        return response.content if hasattr(response, "content") else str(response)

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
        resolved_entities = {}

        if self.entity_resolver:
            
            resolved_entities = self.entity_resolver.extract_entities(user_query)
            print("Resolved Entities: ", resolved_entities)
            if resolved_entities:
                # Build entity context with explicit filter instructions
                entity_context = "RESOLVED ENTITIES — use the most appropiate one:\n"

                # Sort by entity type for consistent output
                entity_list = []
                for entity_name, entity_type in resolved_entities.items():
                    entity_list.append((entity_type, entity_name))
                    # Replace entity in query (case-insensitive)
                    import re
                    pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
                    processed_query = pattern.sub(f"'{entity_name}'", processed_query, count=1)

                # Find the highest score across all resolved entities so we can mark the best one.
                best_score = max(
                    (info.get("score", 0) if isinstance(info, dict) else 0)
                    for info, _ in entity_list
                )

                covered_tickers: set = set()
                for entity_type, entity_name in sorted(entity_list, key=lambda x: (str(x[0]), str(x[1]))):
                    t = entity_type.get("type", "") if isinstance(entity_type, dict) else str(entity_type)
                    info = entity_type if isinstance(entity_type, dict) else {}
                    score = info.get("score", None) if isinstance(info, dict) else None
                    score_tag = f" (score: {score:.1f})" if score is not None else ""
                    strong = " ★ STRONG MATCH" if score is not None and score == best_score else ""

                    if t == "Fund":
                        ticker = info.get("ticker", entity_name)
                        covered_tickers.add(ticker)
                        entity_context += f"  - Fund '{entity_name}'{score_tag}{strong}: filter by .ticker = '{ticker}'\n"
                    elif t == "Company":
                        ticker = info.get("ticker", "")
                        covered_tickers.add(ticker)
                        if ticker:
                            entity_context += (
                                f"  - Company '{entity_name}' (ticker: '{ticker}'){score_tag}{strong}: "
                                f"filter Holding or Company by .ticker = '{ticker}'  [NOT by .name]\n"
                            )
                        else:
                            entity_context += f"  - Company '{entity_name}'{score_tag}{strong}\n"
                    elif t == "Ticker":
                        # Skip Ticker entries that are already covered by a Fund/Company line above
                        company = info.get("company", "")
                        fund = info.get("fund", "")
                        if not company and not fund:
                            entity_context += f"  - Ticker '{entity_name}'{score_tag}{strong}: filter by .ticker = '{entity_name}'\n"
                    elif t in ("Provider", "Trust"):
                        entity_context += f"  - {t}: '{entity_name}'{score_tag}{strong}\n"
                    else:
                        entity_context += f"  - {t}: '{entity_name}'{score_tag}{strong}\n"

                entity_context += f"Processed Question: {processed_query}\n"
                entity_context += "\nIMPORTANT: Use the filter values shown above — do not use company/fund names as property values.\n"
                
                # Compact entity resolution output
                # Sort by string representation to handle non-comparable types (e.g., dicts)
                entity_summary = ", ".join([f"{t}:{n}" for t, n in sorted(entity_list, key=lambda x: (str(x[0]), str(x[1])))])
                print(f"🔍 Entities: {entity_summary}")
        
        # Step 2: Get few-shot examples.
        # query_vector is NOT forwarded — the index uses intent-normalised vectors so
        # a raw pre-computed vector from Nomic/MiniLM would be in the wrong space.
        if self.use_few_shot:
            examples_str = self.selector.get_formatted_context(user_query)
            n_examples = len(examples_str.split('Example'))-1
            print(f"📚 Examples: {n_examples} retrieved")
            if examples_str.strip():
                print("--- Retrieved Examples ---")
                print(examples_str.strip())
                print("--------------------------")
        else:
            examples_str = ""
            print("📚 Examples: disabled")
        
        try:
            # Update temperature if provided
            if temperature is not None and self.backend in ["ollama", "groq"]:
                self.llm.temperature = temperature

            # Step 3: Initial LLM invocation — with context-overflow fallback.
            # If the assembled prompt is too long for the model's n_ctx, retry with
            # progressively fewer few-shot examples (k → k//2 → 0) before giving up.
            def _build_prompt(ex_str: str) -> str:
                return self.prompt.format(
                    schema=self.schema,
                    examples=ex_str,
                    entity_context=entity_context,
                    question=processed_query,
                )

            def _is_context_overflow(exc: Exception) -> bool:
                msg = str(exc).lower()
                return "context length" in msg or "n_keep" in msg or "too many tokens" in msg

            full_prompt = _build_prompt(examples_str)
            self.last_initial_prompt = full_prompt  # stored for benchmark error reporting
            self.last_completion_tokens = 0
            # Schema slice token contribution: measure only the schema section of the prompt
            schema_section = self.prompt.format(
                schema=self.schema, examples="", entity_context="", question=""
            )
            self.last_schema_tokens = len(schema_section) // 4

            response = None
            current_examples = examples_str
            fallback_ks = [self.selector.k // 2, 0]  # e.g. 4 → [2, 0]
            try:
                response = self._invoke_llm(full_prompt)
            except Exception as ctx_err:
                if not _is_context_overflow(ctx_err):
                    raise
                for fallback_k in fallback_ks:
                    if fallback_k > 0:
                        fallback_docs = self.selector.vector_store.similarity_search_with_score(
                            user_query, k=fallback_k
                        )
                        current_examples = self.selector.format_examples_as_string(
                            [d.metadata for d, _ in fallback_docs],
                            [self.selector._l2_to_similarity(s) for _, s in fallback_docs],
                        )
                        print(f"⚠ Context overflow — retrying with k={fallback_k} examples")
                    else:
                        current_examples = ""
                        print("⚠ Context overflow — retrying with 0 examples")
                    full_prompt = _build_prompt(current_examples)
                    self.last_initial_prompt = full_prompt
                    try:
                        response = self._invoke_llm(full_prompt)
                        break
                    except Exception as retry_err:
                        if not _is_context_overflow(retry_err):
                            raise
                if response is None:
                    print("❌ Prompt too long even with 0 examples — skipping query.")
                    return None

            cypher_query = self._clean_cypher(response)

            if not cypher_query:
                print("⚠ Empty Cypher query generated")
                return None

            if self.use_cypher_validator:
                _cypher_before_rules = cypher_query

                cypher_query, _replaced = CypherValidator.replace_fund_name_with_resolved_ticker(
                    cypher_query, resolved_entities
                )
                if _replaced:
                    print("⚡ Auto-replaced Fund {name: '...'} exact filter with resolved ticker")

                cypher_query, _stripped = CypherValidator.strip_filing10k_year_filter(cypher_query)
                if _stripped:
                    print("⚡ Auto-stripped Filing10K {year} filter (year lives on REPORTS_IN, not Filing10K)")

                cypher_query, _stripped = CypherValidator.strip_has_average_returns_year_filter(cypher_query)
                if _stripped:
                    print("⚡ Auto-stripped HAS_AVERAGE_RETURNS {year} filter (return1y/5y/10y already encode the period)")

                cypher_query, _stripped = CypherValidator.strip_portfolio_intermediary_for_allocation(cypher_query)
                if _stripped:
                    print("⚡ Auto-stripped [:HAS_PORTFOLIO]->(Portfolio) hop before HAS_SECTOR/REGION_ALLOCATION (allocation is direct Fund→Sector/Region)")

                cypher_query, _stripped = CypherValidator.strip_vector_where_filters(cypher_query)
                if _stripped:
                    print("⚡ Auto-stripped year/CONTAINS filters from vector query")

                cypher_query, _injected = CypherValidator.inject_chunk_id(cypher_query)
                if _injected:
                    print("⚡ Auto-injected chunk.id AS chunkId into vector query RETURN")

                cypher_query, _boosted = CypherValidator.boost_vector_candidate_count(cypher_query)
                if _boosted:
                    print("⚡ Auto-boosted vector candidate count to 50")

                cypher_query, _fixed = CypherValidator.fix_portfolio_count_ordering(cypher_query)
                if _fixed:
                    print("⚡ Auto-fixed ORDER BY p.count → ORDER BY r.weight DESC on holdings query")

                cypher_query, _injected = CypherValidator.inject_year_in_section_query(cypher_query)
                if _injected:
                    print("⚡ Auto-injected r.year into RETURN and ORDER BY for 10-K section query")

                if cypher_query != _cypher_before_rules:
                    print(f"  BEFORE: {_cypher_before_rules}")
                    print(f"  AFTER:  {cypher_query}")

            print(f"✓ Generated Cypher (attempt 1): {cypher_query}")

            # Step 4: Validate + retry loop
            self.last_validation_failures = []
            self.last_write_blocked_op = None
            if self._capture_prompts:
                self.last_retry_prompts = []
            self.last_retry_attempts = 0
            self.last_total_prompt_tokens = len(full_prompt) // 4  # initial call
            self.last_completion_tokens += len(response) // 4 if response else 0
            validation_result = ValidationResult(is_valid=True, original_query=cypher_query)
            for attempt in range(1, self.max_validation_retries + 1):
                self.last_retry_attempts = attempt
                if not self.use_cypher_validator:
                    break  # skip validation loop entirely
                validation_result = self.validator.validate(cypher_query)

                if validation_result.is_valid:
                    print(f"✅ Validation passed (attempt {attempt})")
                    break

                # Record which rule fired for benchmark analytics
                if validation_result.triggered_rule:
                    self.last_validation_failures.append(validation_result.triggered_rule)

                # Hard stop: write operations are never fixable by retrying
                if validation_result.triggered_rule == "WRITE_OPERATION":
                    upper = cypher_query.upper()
                    self.last_write_blocked_op = next(
                        (kw.strip() for kw in self.validator.BLOCKED_WRITE_KEYWORDS if kw.strip() in upper),
                        "WRITE",
                    )
                    print(f"🚨 WRITE OPERATION BLOCKED ({self.last_write_blocked_op}) — aborting all retries immediately")
                    return None

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

                # Short-circuit: if ALL errors are variable collisions, fix directly
                # without calling the LLM (the model consistently re-introduces this mistake).
                all_collision_errors = all(
                    "VARIABLE COLLISION" in e for e in validation_result.syntax_errors
                ) and validation_result.syntax_errors and not validation_result.schema_errors
                if all_collision_errors:
                    fixed = self._fix_variable_collisions(cypher_query)
                    if fixed != cypher_query:
                        print(f"⚡ Short-circuit: auto-fixed variable collision (skipping LLM retry)")
                        cypher_query = fixed
                        continue

                # Check Groq rate limits before retrying
                if self.backend == "groq" and not self._check_groq_rate_limits():
                    print("❌ Cannot retry: Groq rate limits exceeded")
                    break
                
                # Build retry prompt with error feedback — always use the full detailed
                # schema on retries so the LLM has the complete graph context to fix errors.
                # Add targeted examples based on error type
                error_specific_examples = self._get_error_specific_examples(error_summary)
                
                if error_specific_examples:
                    print(f"💡 Injecting error-specific examples for retry attempt {attempt + 1}")
                
                if self.retry_strategy == "lean":
                    retry_prompt_text = self.retry_prompt.format(
                        examples=current_examples,
                        failed_query=cypher_query,
                        validation_errors=error_summary,
                        error_examples=error_specific_examples,
                    )
                else:
                    retry_prompt_text = self.retry_prompt.format(
                        schema=self.detailed_schema,
                        entity_context=entity_context,
                        examples=current_examples,
                        question=processed_query,
                        failed_query=cypher_query,
                        validation_errors=error_summary,
                        error_examples=error_specific_examples,
                    )

                token_est = len(retry_prompt_text) // 4
                self.last_total_prompt_tokens += token_est
                if self._capture_prompts:
                    self.last_retry_prompts.append(retry_prompt_text)

                retry_response = self._invoke_llm(retry_prompt_text)
                self.last_completion_tokens += len(retry_response) // 4 if retry_response else 0
                cypher_query = self._clean_cypher(retry_response)

                if not cypher_query:
                    print("⚠ Empty Cypher query on retry")
                    break

                if self.use_cypher_validator:
                    _cypher_before_rules = cypher_query

                    cypher_query, _replaced = CypherValidator.replace_fund_name_with_resolved_ticker(
                        cypher_query, resolved_entities
                    )
                    if _replaced:
                        print("⚡ Auto-replaced Fund {name: '...'} exact filter with resolved ticker (retry)")

                    cypher_query, _stripped = CypherValidator.strip_filing10k_year_filter(cypher_query)
                    if _stripped:
                        print("⚡ Auto-stripped Filing10K {year} filter (retry)")

                    cypher_query, _stripped = CypherValidator.strip_has_average_returns_year_filter(cypher_query)
                    if _stripped:
                        print("⚡ Auto-stripped HAS_AVERAGE_RETURNS {year} filter (retry)")

                    cypher_query, _stripped = CypherValidator.strip_portfolio_intermediary_for_allocation(cypher_query)
                    if _stripped:
                        print("⚡ Auto-stripped [:HAS_PORTFOLIO]->(Portfolio) hop before allocation (retry)")

                    cypher_query, _stripped = CypherValidator.strip_vector_where_filters(cypher_query)
                    if _stripped:
                        print("⚡ Auto-stripped year/CONTAINS filters from vector query (retry)")

                    cypher_query, _injected = CypherValidator.inject_chunk_id(cypher_query)
                    if _injected:
                        print("⚡ Auto-injected chunk.id AS chunkId into vector query RETURN (retry)")

                    cypher_query, _boosted = CypherValidator.boost_vector_candidate_count(cypher_query)
                    if _boosted:
                        print("⚡ Auto-boosted vector candidate count to 50 (retry)")

                    if cypher_query != _cypher_before_rules:
                        print(f"  BEFORE: {_cypher_before_rules}")
                        print(f"  AFTER:  {cypher_query}")

                print(f"✓ Generated Cypher (attempt {attempt + 1}): {cypher_query}")
            
            # Fallback LLM — fires only if retries were exhausted and query is still invalid
            if self.fallback_llm is not None and not validation_result.is_valid:
                print(f"🆘 All retries exhausted — escalating to {self._fallback_label} fallback...")
                try:
                    from simple_rag.rag.prompt_templates import CYPHER_RETRY_TEMPLATE
                    from langchain_core.prompts import PromptTemplate
                    _full_retry_prompt = PromptTemplate(
                        input_variables=["schema", "entity_context", "examples", "question", "failed_query", "validation_errors", "error_examples"],
                        template=CYPHER_RETRY_TEMPLATE,
                    )
                    fallback_prompt = _full_retry_prompt.format(
                        schema=self.detailed_schema,
                        entity_context=entity_context,
                        examples=current_examples,
                        question=processed_query,
                        failed_query=cypher_query,
                        validation_errors=error_summary,
                        error_examples="",
                    )
                    fallback_response = self._invoke_with_llm(self.fallback_llm, fallback_prompt)
                    fallback_cypher = self._clean_cypher(fallback_response)
                    if fallback_cypher:
                        fallback_validation = self.validator.validate(fallback_cypher)
                        status = "✅ valid" if fallback_validation.is_valid else "⚠ still invalid"
                        print(f"🆘 {self._fallback_label} fallback result ({status}): {fallback_cypher}")
                        cypher_query = fallback_cypher
                except Exception as e:
                    print(f"⚠ {self._fallback_label} fallback failed: {e}")

            # Restore original temperature
            if temperature is not None and self.backend in ["ollama", "groq"]:
                self.llm.temperature = self.temperature

            return cypher_query
            
        except Exception as e:
            print(f"✗ Error during translation: {e}")
            logger.error(f"Translation error: {e}", exc_info=True)
            return None
    
    def _get_error_specific_examples(self, error_summary: str) -> str:
        """
        Analyze validation errors and return targeted examples to help fix them.
        
        Args:
            error_summary: String containing validation error messages
            
        Returns:
            String with relevant examples, or empty string if no specific pattern detected
        """
        error_lower = error_summary.lower()
        examples = []
        
        # Pattern 1: Undefined variable (e.g., "Variable `p` not defined")
        if "variable" in error_lower and "not defined" in error_lower:
            examples.append("""
ERROR PATTERN: Undefined Variable
WRONG: MATCH (f:Fund)-[:HAS_PORTFOLIO]->(:Portfolio) RETURN p.count
       (p is not defined because Portfolio is anonymous)
       
CORRECT: MATCH (f:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio) RETURN p.date
         (p is now defined and can be used in RETURN)

CORRECT: MATCH (f:Fund {ticker: 'VTI'})-[:HAS_PORTFOLIO]->(:Portfolio)-[:HAS_HOLDING]->(h:Holding) RETURN count(h)
         (Use count(h) to count holdings, not p.count which doesn't exist)
""")
        
        # Pattern 2: DISTINCT + ORDER BY scope issue
        if "distinct" in error_lower and ("order by" in error_lower or "not possible to access" in error_lower):
            examples.append("""
ERROR PATTERN: DISTINCT + ORDER BY Scope
WRONG: MATCH (f:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight) 
       RETURN DISTINCT f.ticker, f.name ORDER BY r.year DESC
       (r.year is not in RETURN, cannot ORDER BY it with DISTINCT)
       
CORRECT: MATCH (f:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight) 
         RETURN DISTINCT f.ticker, f.name, r.year ORDER BY r.year DESC
         (Include r.year in RETURN when using it in ORDER BY with DISTINCT)

CORRECT: MATCH (f:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight) 
         RETURN f.ticker, f.name ORDER BY r.year DESC
         (Remove DISTINCT if you don't need it)
""")
        
        # Pattern 2b: Aggregate function + ORDER BY scope issue
        # The Neo4j error says "DISTINCT or an aggregation" so "aggregation" is the signal
        # when the query uses AVG/COUNT/SUM/etc. without DISTINCT.
        if "aggregation" in error_lower and "not possible to access" in error_lower:
            examples.append("""
ERROR PATTERN: Aggregate Function + ORDER BY Scope
When AVG, COUNT, SUM, MIN, or MAX is in RETURN, all non-aggregated variables become
grouping keys and relationship variables (r, rel, etc.) become out of scope for ORDER BY.

WRONG: MATCH (f:Fund {ticker: 'VTI'})-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight)
       RETURN f.ticker, AVG(fh.turnover) AS averageTurnoverRate ORDER BY r.year DESC LIMIT 1
       (r is not in RETURN — AVG creates aggregation scope, r becomes inaccessible)

CORRECT (average across all years — no ORDER BY needed):
       MATCH (f:Fund {ticker: 'VTI'})-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight)
       RETURN f.ticker, AVG(fh.turnover) AS averageTurnoverRate

CORRECT (latest single value — drop AVG, use ORDER BY + LIMIT):
       MATCH (f:Fund {ticker: 'VTI'})-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight)
       RETURN f.ticker, fh.turnover AS turnoverRate ORDER BY r.year DESC LIMIT 1

RULE: Never mix an aggregate function in RETURN with ORDER BY on a relationship variable
      that is not also in the RETURN clause.
""")

        # Pattern 3: Property without value in curly braces
        if "property" in error_lower and ("without" in error_lower or "value" in error_lower):
            examples.append("""
ERROR PATTERN: Property Without Value
WRONG: MATCH (f:Fund {ticker}) RETURN f.name
       (ticker has no value assigned)
       
CORRECT: MATCH (f:Fund {ticker: 'VTI'}) RETURN f.name
         (ticker has a value 'VTI')

CORRECT: MATCH (f:Fund) WHERE f.ticker IS NOT NULL RETURN f.name
         (Use WHERE clause for property checks without specific values)
""")
        
        # Pattern 4: Pattern expressions introducing variables
        if "pattern" in error_lower and ("introduce" in error_lower or "new variable" in error_lower):
            examples.append("""
ERROR PATTERN: Pattern Expression Introducing Variables
WRONG: WHERE node IN (f)-[r:REL]->(n) 
       (Cannot introduce r or n inside pattern expression)
       
CORRECT: MATCH (f)-[r:REL]->(n) WHERE node = n RETURN f
         (Define variables in MATCH, use them in WHERE)
""")
        
        # Pattern 5: Fulltext score not yielded
        if "score not yielded" in error_lower or (
            "fulltext" in error_lower and "yield" in error_lower and "score" in error_lower
        ):
            examples.append("""
ERROR PATTERN: CALL db.index.fulltext.queryNodes — 'score' not in YIELD
When using CALL db.index.fulltext.queryNodes(), every variable you use later must be in YIELD.
If you reference 'score' in ORDER BY, WHERE, or RETURN, add it to the YIELD clause.

BAD:  CALL db.index.fulltext.queryNodes('fundNameIndex', 'Total Stock Market~') YIELD node ORDER BY score DESC LIMIT 1
      MATCH (node)-[:DEFINED_BY]->(p:Profile) RETURN node.name, p.summaryProspectus
      (score is used in ORDER BY but not yielded — causes "Variable `score` not defined")

GOOD: CALL db.index.fulltext.queryNodes('fundNameIndex', 'Total Stock Market~') YIELD node, score ORDER BY score DESC LIMIT 1
      MATCH (node)-[:DEFINED_BY]->(p:Profile) RETURN node.name, p.summaryProspectus
      (score is yielded alongside node, so ORDER BY score works correctly)
""")

        # Pattern 6: Accessing properties on wrong node/relationship
        if "property" in error_lower or "does not exist" in error_lower:
            examples.append("""
SCHEMA REMINDER:
- expenseRatio, turnover, netAssets are on FinancialHighlight node
- weight, marketValue, shares, payoffProfile are on HAS_HOLDING relationship
- year is on HAS_FINANCIAL_HIGHLIGHT and DEFINED_BY relationships (use r.year)
- Portfolio has: date, seriesId, count (use p.count for number of holdings)
- Do NOT use fh.numberHoldings — use p.count from Portfolio instead

CORRECT: MATCH (f:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight)
         RETURN fh.expenseRatio, r.year ORDER BY r.year DESC

CORRECT: MATCH (f:Fund {ticker: 'VTI'})-[:HAS_PORTFOLIO]->(p:Portfolio)
         RETURN p.count AS holdingsCount
""")
        
        # Pattern 7: Relationship direction written as -<[:REL] instead of <-[:REL]
        if "-<[" in error_summary or "invalid input '-'" in error_lower:
            examples.append("""
ERROR PATTERN: Reversed Relationship Direction Syntax
The arrow for an incoming relationship is <-[:REL]- NOT -<[:REL]-.
The dash and angle-bracket must be: <- before the bracket, or -> after it.

WRONG: MATCH (chunk)<-[:HAS_CHUNK]-(s:Section)-<[:HAS_SECTION]-(f:Filing10K)
       (The -< is not valid Cypher syntax)

CORRECT: MATCH (chunk)<-[:HAS_CHUNK]-(s:Section)<-[:HAS_SECTION]-(f:Filing10K)
         (Use <- before the bracket for incoming relationships)

CORRECT pattern for vector search traversal:
  CALL db.index.vector.queryNodes('chunkEmbeddingIndex', 10, $queryVector) YIELD node AS chunk, score
  MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:RiskFactor)<-[:HAS_SECTION]-(f:Filing10K)<-[r:REPORTS_IN]-(c:Company {ticker: 'AAPL'})
  RETURN chunk.id, chunk.text AS text, r.year AS filingYear, score ORDER BY score DESC LIMIT 5

RULE: Every incoming relationship arrow must be written as <-[:TYPE]- never as -<[:TYPE]-.
""")

        # Return formatted examples or empty string
        if examples:
            return "\n" + "="*60 + "\nRELEVANT EXAMPLES TO FIX YOUR ERROR:\n" + "="*60 + "\n".join(examples)
        return ""
    
    
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
        
        # Remove common prefixes (including EXPLAIN/PROFILE which the model sometimes adds)
        prefixes = ["cypher:", "Cypher:", "CYPHER:", "Query:", "query:", "EXPLAIN ", "PROFILE "]
        for prefix in prefixes:
            if query.upper().startswith(prefix.upper()):
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

        # 4. Fix variable collision: same name used for both a relationship and a node.
        # e.g. -[it:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction) → -[:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction)
        # Strategy: keep the node variable (used in RETURN/WHERE), anonymise the relationship.
        query = self._fix_variable_collisions(query)

        # 5. Auto-inject DISTINCT when multi-hop traversal could cause duplicate rows
        query = self._auto_distinct(query)

        # 6. Auto-fix undefined relationship variables.
        # If the query references r.property (or rel.property, h.property, etc.) in
        # RETURN/WHERE/ORDER BY but the relationship pattern is anonymous (e.g. [:REL_TYPE]),
        # inject the variable name into the first anonymous relationship pattern.
        # e.g. -[:HAS_REGION_ALLOCATION]-> with r.weight in RETURN → -[r:HAS_REGION_ALLOCATION]->
        query = self._fix_undefined_rel_vars(query)

        return query

    def _fix_variable_collisions(self, query: str) -> str:
        """
        Detect and fix variable name collisions where the same identifier is used
        for both a named relationship and a node in the same query.

        Example collision (fatal Cypher syntax error):
            -[it:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction)

        Fix: anonymise the relationship bracket — the node variable is preserved
        because it is the one referenced in RETURN/WHERE/ORDER BY:
            -[:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction)
        """
        # Named relationships: -[varName:REL_TYPE]-> or <-[varName:REL_TYPE]-
        named_rel_pattern = re.compile(r'-\[([a-zA-Z_]\w*):([A-Z_][A-Z_0-9]*)\]', re.IGNORECASE)
        # Named nodes: (varName:Label) or (varName {
        named_node_vars = set(re.findall(r'\(\s*([a-zA-Z_]\w*)\s*[:{]', query))

        def _anonymise_if_collision(m: re.Match) -> str:
            var, rel_type = m.group(1), m.group(2)
            if var in named_node_vars:
                return f'-[:{rel_type}]'
            return m.group(0)

        fixed = named_rel_pattern.sub(_anonymise_if_collision, query)
        if fixed != query:
            print(f"⚡ Auto-fixed variable collision (relationship variable anonymised)")
        return fixed

    def _fix_undefined_rel_vars(self, query: str) -> str:
        """
        Detect relationship variables used in RETURN/WHERE/ORDER BY that are not
        captured in the MATCH pattern, and auto-inject them into the first matching
        anonymous relationship bracket.
        """
        # Find all anonymous relationship patterns: [-[:REL] or -[:REL {
        anon_rel_pattern = re.compile(r'-\[(\s*):([A-Z_][A-Z_0-9]*)', re.IGNORECASE)

        # Find all var.property usages in RETURN/ORDER BY/WHERE clauses
        usage_pattern = re.compile(r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b')

        # Variables already captured in relationship patterns [varName:REL]
        defined_rel_vars = set(m.group(1) for m in re.finditer(r'\[\s*([a-zA-Z_]\w*)\s*:', query))

        # Relationship-property names that signal a rel-var usage (not a node property)
        rel_properties = {'weight', 'year', 'date', 'ceoCompensation', 'ceoActuallyPaid',
                          'shares', 'marketValue', 'currency', 'fairValueLevel',
                          'isRestricted', 'payoffProfile'}

        for match in usage_pattern.finditer(query):
            var_name = match.group(1)
            prop_name = match.group(2)
            if prop_name not in rel_properties:
                continue
            if var_name in defined_rel_vars:
                continue
            # var_name is used but not defined — try to inject it into the first
            # anonymous relationship pattern in the query
            if anon_rel_pattern.search(query):
                query = anon_rel_pattern.sub(
                    lambda m: f'-[{var_name}:{m.group(2)}',
                    query,
                    count=1  # only fix the first anonymous rel to avoid over-injecting
                )
                defined_rel_vars.add(var_name)

        return query

    def _auto_distinct(self, query: str) -> str:
        """
        Inject DISTINCT into RETURN when a multi-hop MATCH traverses intermediate nodes
        that are not in the RETURN clause, which would otherwise produce duplicate rows.

        Heuristic: if MATCH has ≥2 relationship arrows AND RETURN has no DISTINCT AND
        no aggregate functions are used, add RETURN DISTINCT.
        """
        if not re.search(r'\bMATCH\b', query, re.IGNORECASE):
            return query
        if not re.search(r'\bRETURN\b', query, re.IGNORECASE):
            return query

        # Already distinct — nothing to do
        if re.search(r'\bRETURN\s+DISTINCT\b', query, re.IGNORECASE):
            return query

        # Aggregates collapse duplicates on their own — leave them alone
        if re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|COLLECT)\s*\(', query, re.IGNORECASE):
            return query

        # ORDER BY may reference relationship variables (e.g. r.year) that are not in RETURN.
        # Injecting DISTINCT in that case causes a scope error — skip.
        if re.search(r'\bORDER\s+BY\b', query, re.IGNORECASE):
            return query

        # Count relationship hops in the MATCH portion only
        match_part = re.split(r'\bRETURN\b', query, maxsplit=1, flags=re.IGNORECASE)[0]
        hop_count = len(re.findall(r'->|<-', match_part))

        if hop_count >= 2:
            query = re.sub(r'\bRETURN\b', 'RETURN DISTINCT', query, count=1, flags=re.IGNORECASE)

        return query
    
    def refresh_entity_cache(self) -> None:
        """Reload the entity resolver cache from Neo4j. Call this if the database
        was updated after the translator was initialized."""
        if self.entity_resolver:
            self.entity_resolver._refresh_cache()
            print("✓ Entity cache refreshed")
        else:
            print("⚠ Entity resolver is not enabled")

    def stop_ollama_server(self) -> None:
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