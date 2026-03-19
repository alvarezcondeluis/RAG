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
from simple_rag.rag.post_processing.cypher_validator import CypherValidator
from simple_rag.rag.schema_definitions import DETAILED_SCHEMA
from simple_rag.rag.prompt_templates import CYPHER_GENERATION_TEMPLATE, CYPHER_RETRY_TEMPLATE

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
        backend: Literal["ollama", "huggingface", "groq", "openai"] = "ollama",
        device: str = "cuda",
        temperature: float = DEFAULT_TEMPERATURE,
        use_entity_resolver: bool = True,
        entity_resolver_debug: bool = False,
        groq_api_key: Optional[str] = None,
        openai_compatible_host: str = OPENAI_COMPATIBLE_DEFAULT_HOST,
        openai_compatible_port: int = OPENAI_COMPATIBLE_DEFAULT_PORT,
        max_validation_retries: int = DEFAULT_VALIDATION_RETRIES,
        llama_cpp_host: Optional[str] = None,
        llama_cpp_port: Optional[int] = None,
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
        self.max_validation_retries = max_validation_retries
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
        print("✓ CypherValidator initialized")

        # LangChain components
        self.llm = None
        self.chain = None
        self.llama_cpp_process = None  # managed llama.cpp subprocess (if launched by us)
        
        self.detailed_schema = DETAILED_SCHEMA  # always the full schema — never overwritten by slicing
        self.schema = DETAILED_SCHEMA           # may be temporarily replaced by a schema slice
        self.last_initial_prompt: Optional[str] = None  # prompt used on the first LLM call (captured for debugging)
        
        # Use imported prompt templates
        self.prompt = PromptTemplate(
            input_variables=["schema", "examples", "entity_context", "question"],
            template=CYPHER_GENERATION_TEMPLATE
        )
        
        self.retry_prompt = PromptTemplate(
            input_variables=["schema", "entity_context", "question", "failed_query", "validation_errors"],
            template=CYPHER_RETRY_TEMPLATE
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
        """Initialize LangChain with an OpenAI-compatible server (LM Studio, llama.cpp, vLLM, etc.)."""
        try:
            base_url = f"http://{self.openai_compatible_host}:{self.openai_compatible_port}/v1"
            self.llm = ChatOpenAI(
                model=self.model_name,
                base_url=base_url,
                api_key="not-needed",
                temperature=self.temperature,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            self.chain = self.prompt | self.llm | StrOutputParser()
            print(f"✓ OpenAI-compatible backend initialized → {base_url}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Failed to connect to OpenAI-compatible server at {base_url}\n"
                f"Please ensure LM Studio, llama.cpp, or another OpenAI-compatible server is running on port {self.openai_compatible_port}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI-compatible chain: {e}") from e

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
            elif self.backend == "groq":
                print("✓ Groq backend configured (API-based, no connection test needed)")
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
            self.last_initial_prompt = full_prompt  # stored for benchmark error reporting

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
                
                # Build retry prompt with error feedback — always use the full detailed
                # schema on retries so the LLM has the complete graph context to fix errors.
                retry_prompt_text = self.retry_prompt.format(
                    schema=self.detailed_schema,
                    entity_context=entity_context,
                    question=processed_query,
                    failed_query=cypher_query,
                    validation_errors=error_summary
                )

                token_est = len(retry_prompt_text) // 4
                print(f"🔄 Retrying with full schema + error feedback (attempt {attempt + 1}) — {token_est:,} tokens (est.):")
                print(f"{'─'*76}")
                for line in retry_prompt_text.splitlines():
                    print(f"  {line}")
                print(f"{'─'*76}")
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

        # 4. Auto-inject DISTINCT when multi-hop traversal could cause duplicate rows
        query = self._auto_distinct(query)

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

        # Count relationship hops in the MATCH portion only
        match_part = re.split(r'\bRETURN\b', query, maxsplit=1, flags=re.IGNORECASE)[0]
        hop_count = len(re.findall(r'->|<-', match_part))

        if hop_count >= 2:
            query = re.sub(r'\bRETURN\b', 'RETURN DISTINCT', query, count=1, flags=re.IGNORECASE)

        return query
    
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