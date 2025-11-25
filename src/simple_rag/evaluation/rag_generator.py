import json
import pandas as pd
import os
import time
from datasets import Dataset
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Set environment variables to increase RAGAS timeouts
os.environ["RAGAS_TIMEOUT"] = "600"  # 10 minutes
os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # Disable telemetry
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from .gemini import GeminiChatModel
from ..rag.rag import RAG
from ..retriever.enhanced_retriever import EnhancedRetriever
from ..embeddings.embedding import EmbedData
from ..database.qdrant import QdrantDatabase


class RAGEvaluator:
    """
    RAGEvaluator class for RAG systems.
    
    This class handles the complete evaluation pipeline including:
    - RAG system initialization
    - Benchmark dataset loading
    - RAGAS metric evaluation
    - Results analysis and export
    
    Supports both local Ollama models and Google Gemini API for evaluation.
    
    """
    
    def __init__(
        self,
        collection_name: str = "efficient_rag",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        llm_model: str = "qwen3:4b-instruct",
        judge_llm_model: str = "qwen3:4b-instruct",
        judge_llm_provider: str = "ollama",
        ragas_embedding_provider: str = "huggingface",
        ollama_embedding_model: str = "BAAI/bge-base-en-v1.5",
        cache_folder: str = "./cache",
        retriever_k: int = 15,
        rerank_k: int = 5,
    ):
        """
        Initialize the RAGAS Evaluator.
        
        Args:
            collection_name: Qdrant collection name
            embedding_model: HuggingFace embedding model name (for RAG system)
            llm_model: Ollama LLM model for RAG system
            judge_llm_model: LLM model name for RAGAS judge
                           - For Ollama: "llama3.1:8b", "qwen2.5:14b", etc.
                           - For Gemini: "gemini-1.5-flash", "gemini-1.5-pro", etc.
            judge_llm_provider: Provider for RAGAS judge LLM ("ollama" or "gemini")
            ragas_embedding_provider: Provider for RAGAS embeddings ("ollama" or "huggingface")
            ollama_embedding_model: Ollama embedding model name (e.g., "nomic-embed-text", "mxbai-embed-large")
            cache_folder: Cache folder for HuggingFace embedding models
            retriever_k: Number of contexts to retrieve
            rerank_k: Number of contexts to rerank
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.judge_llm_model = judge_llm_model
        self.judge_llm_provider = judge_llm_provider.lower()
        self.ragas_embedding_provider = ragas_embedding_provider.lower()
        self.ollama_embedding_model = ollama_embedding_model
        self.cache_folder = cache_folder
        
        # Initialize RAGAS judge LLM based on provider
        if self.judge_llm_provider == "gemini":
            # Use Google Gemini API
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found in environment variables. "
                    "Please add it to your .env file."
                )
            
            print(f"Using Gemini API for RAGAS judge: {judge_llm_model}")
            self.judge_llm = GeminiChatModel(
                model_name=judge_llm_model,
                api_key=google_api_key,
                temperature=0.0  # More deterministic for evaluation
            )
        elif self.judge_llm_provider == "ollama":
            
            print(f"Using Ollama for RAGAS judge: {judge_llm_model}")
            self.judge_llm = ChatOllama(
                model=judge_llm_model, 
                timeout=800.0,  # 13+ minute timeout
                temperature=0.0,  # More deterministic for evaluation
                num_ctx=4096,  # Increase context window
            )
        else:
            raise ValueError(
                f"Invalid judge_llm_provider: {judge_llm_provider}. "
                "Must be 'ollama' or 'gemini'."
            )
        
        # Initialize embeddings for RAGAS based on provider
        if self.ragas_embedding_provider == "ollama":
            print(f"Using Ollama embeddings for RAGAS: {ollama_embedding_model}")
            self.ragas_embeddings = OllamaEmbeddings(
                model=ollama_embedding_model
            )
        elif self.ragas_embedding_provider == "huggingface":
            print(f"Using HuggingFace embeddings for RAGAS: {embedding_model}")
            self.ragas_embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                cache_folder=cache_folder,
                model_kwargs={'trust_remote_code': True}
            )
        else:
            raise ValueError(
                f"Invalid ragas_embedding_provider: {ragas_embedding_provider}. "
                "Must be 'ollama' or 'huggingface'."
            )
        
        # RAG system components (initialized lazily)
        self.rag_system = None
        self.retriever = None
        self.retriever_k = retriever_k  # Number of contexts to retrieve
        # Note: self.judge_llm is already initialized above
        self.results_df = None
        self.evaluation_data = []
        self.rerank_k = rerank_k
    
    def _initialize_rag_system(self):
        """Initialize RAG system components."""
        if self.rag_system:
            return
            
        print(f"\n[Phase 1] Initializing RAG system...")
        print(f"  - Collection: {self.collection_name}")
        print(f"  - Embedding Model: {self.embedding_model}")
        print(f"  - LLM Model: {self.llm_model}")
        print(f"  - Retriever K: {self.retriever_k}")
        
        # (Using your project's class names)
        embed_model_instance = EmbedData(model_name=self.embedding_model)
        vector_db_instance = QdrantDatabase(collection_name=self.collection_name)

        self.retriever = EnhancedRetriever(
            vector_db=vector_db_instance,
            embed_model=embed_model_instance.model
        )
        self.rag_system = RAG(
            retriever=self.retriever, 
            llm_name=self.llm_model,
            retriever_k=self.retriever_k,
            rerank_k=self.rerank_k
        )
        
        print("  ✓ RAG system initialized successfully")
    
    def warmup_models(self):
        """
        Warmup the RAG system LLM.
        
        This ensures the model is loaded into memory and responsive before
        running the actual evaluation, which helps avoid timeouts on the
        first real query.
        """
        print("\n[Warmup] Warming up RAG system LLM...")
        
        # Warmup RAG system
        self._initialize_rag_system()
        
        try:
            warmup_response = self.rag_system.warmup()
            print(f"  ✓ RAG LLM response: {warmup_response}...")
            print("  ✓ Warmup complete!\n")
        except Exception as e:
            print(f"  ⚠ RAG LLM warmup failed: {e}\n")
    
    def generate_rag_outputs(self, benchmark_json_path: str, num_questions: int = None):
        """
        PHASE 1: Run RAG system on all questions and store outputs.
        """
        print(f"\n[Phase 1] Generating RAG outputs...")
        
        # Print current working directory and file path info
        import os
        print(f"  - Current working directory: {os.getcwd()}")
        print(f"  - Benchmark JSON path (provided): {benchmark_json_path}")
        
        # Check if file exists
        if not os.path.exists(benchmark_json_path):
            # Try to find the file with absolute path
            abs_path = os.path.abspath(benchmark_json_path)
            print(f"  - Absolute path: {abs_path}")
            raise FileNotFoundError(
                f"Benchmark JSON file not found!\n"
                f"  Provided path: {benchmark_json_path}\n"
                f"  Absolute path: {abs_path}\n"
                f"  Current directory: {os.getcwd()}"
            )
        
        print(f"  ✓ Found benchmark file: {os.path.abspath(benchmark_json_path)}")
        
        # Initialize RAG system
        self._initialize_rag_system()
        
        start_time = time.time()
        self.warmup_models()
        end_time = time.time()
        print(f"Warmup took {end_time - start_time:.2f} seconds")
        
        # Load benchmark data
        try:
            with open(benchmark_json_path, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading benchmark file: {e}")
        
        questions_list = benchmark_data.get("questions", [])
        if not questions_list:
            raise ValueError("No 'questions' field found in benchmark JSON or it's empty")
        
        if num_questions:
            questions_list = questions_list[:num_questions]
            print(f"  - Testing with first {num_questions} questions")
        else:
            num_questions = len(questions_list)
            print(f"  - Evaluating all {num_questions} questions")
        
        self.evaluation_data = [] # Reset data
        for idx, item in enumerate(questions_list):
            print(f"  Processing question {idx+1}/{num_questions}...")
            
            try:
                # Get the RAG system's output
                rag_output = self._query_rag(item["question"])
                
                # Store everything in our evaluation list
                # Start with all original fields from the JSON
                eval_item = dict(item)  # Copy all original fields
                
                # Rename 'contexts' to 'ground_truth_contexts' to avoid confusion
                if "contexts" in eval_item:
                    eval_item["ground_truth_contexts"] = eval_item.pop("contexts")
                
                # Add RAG outputs
                print("Anser for question", str(idx+1), "is:", rag_output["answer"])
                eval_item["answer"] = rag_output["answer"]
                eval_item["contexts"] = rag_output["contexts"]
                
                self.evaluation_data.append(eval_item)
            except Exception as e:
                print(f"    ⚠ Error processing question {idx+1}: {e}")
                # Continue with next question instead of failing completely
                continue
        
        if not self.evaluation_data:
            raise RuntimeError("No questions were successfully processed!")
        
        print(f"  ✓ RAG output generation complete. Processed {len(self.evaluation_data)}/{num_questions} questions.")
        print(f"\n  Sample output (first question):")
        print(f"    Question: {self.evaluation_data[0]['question'][:80]}...")
        print(f"    Answer: {self.evaluation_data[0]['answer'][:80]}...")
        print(f"    Contexts retrieved: {len(self.evaluation_data[0]['contexts'])}")
    
    def generate_rag_outputs_gemini(self, benchmark_json_path: str, gemini_api_key: str, 
                                     model_name: str = "gemini-2.5-flash", 
                                     temperature: float = 0.0,
                                     num_questions: int = None):
        """
        PHASE 1: Run RAG system on all questions using Gemini API with rate limiting.
        
        Rate limit: 10 requests per minute (6 seconds between requests).
        
        Args:
            benchmark_json_path: Path to benchmark JSON file
            gemini_api_key: Google API key for Gemini
            model_name: Gemini model to use (default: gemini-2.5-flash)
            temperature: Temperature for generation (default: 0.0)
            num_questions: Optional limit on number of questions to process
        """
        print(f"\n[Phase 1] Generating RAG outputs with Gemini API...")
        print(f"  - Model: {model_name}")
        print(f"  - Rate limit: 10 requests/minute (6s delay between calls)")
        
        # Print current working directory and file path info
        import os
        import time
        print(f"  - Current working directory: {os.getcwd()}")
        print(f"  - Benchmark JSON path (provided): {benchmark_json_path}")
        
        # Check if file exists
        if not os.path.exists(benchmark_json_path):
            # Try to find the file with absolute path
            abs_path = os.path.abspath(benchmark_json_path)
            print(f"  - Absolute path: {abs_path}")
            raise FileNotFoundError(
                f"Benchmark JSON file not found!\n"
                f"  Provided path: {benchmark_json_path}\n"
                f"  Absolute path: {abs_path}\n"
                f"  Current directory: {os.getcwd()}"
            )
        
        print(f"  ✓ Found benchmark file: {os.path.abspath(benchmark_json_path)}")
        
        # Initialize RAG system
        self._initialize_rag_system()
        
        # Load benchmark data
        try:
            with open(benchmark_json_path, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading benchmark file: {e}")
        
        questions_list = benchmark_data.get("questions", [])
        if not questions_list:
            raise ValueError("No 'questions' field found in benchmark JSON or it's empty")
        
        if num_questions:
            questions_list = questions_list[:num_questions]
            print(f"  - Testing with first {num_questions} questions")
        else:
            num_questions = len(questions_list)
            print(f"  - Evaluating all {num_questions} questions")
        
        # Calculate estimated time
        estimated_time_seconds = num_questions * 6  # 6 seconds per request
        estimated_time_minutes = estimated_time_seconds / 60
        print(f"  - Estimated time: {estimated_time_minutes:.1f} minutes ({estimated_time_seconds}s)")
        
        self.evaluation_data = [] # Reset data
        
        for idx, item in enumerate(questions_list):
            print(f"  Processing question {idx+1}/{num_questions}...")
            
            try:
                # Get the RAG system's output using Gemini
                rag_output = self._query_rag_gemini(
                    item["question"], 
                    gemini_api_key=gemini_api_key,
                    model_name=model_name,
                    temperature=temperature
                )
                
                # Store everything in our evaluation list
                # Start with all original fields from the JSON
                eval_item = dict(item)  # Copy all original fields
                
                # Rename 'contexts' to 'ground_truth_contexts' to avoid confusion
                if "contexts" in eval_item:
                    eval_item["ground_truth_contexts"] = eval_item.pop("contexts")
                
                # Add RAG outputs
                print(f"    Answer for question {idx+1}: {rag_output['answer'][:80]}...")
                eval_item["answer"] = rag_output["answer"]
                eval_item["contexts"] = rag_output["contexts"]
                
                self.evaluation_data.append(eval_item)
                
                # Rate limiting: Wait 6 seconds between requests (10 requests/minute)
                # Don't wait after the last request
                if idx < num_questions - 1:
                    print(f"    ⏳ Waiting 6 seconds before next request (rate limit)...")
                    time.sleep(6)
                    
            except Exception as e:
                print(f"    ⚠ Error processing question {idx+1}: {e}")
                # Continue with next question instead of failing completely
                continue
        
        if not self.evaluation_data:
            raise RuntimeError("No questions were successfully processed!")
        
        print(f"  ✓ RAG output generation complete. Processed {len(self.evaluation_data)}/{num_questions} questions.")
        print(f"\n  Sample output (first question):")
        print(f"    Question: {self.evaluation_data[0]['question'][:80]}...")
        print(f"    Answer: {self.evaluation_data[0]['answer'][:80]}...")
        print(f"    Contexts retrieved: {len(self.evaluation_data[0]['contexts'])}")
    
    def save_rag_outputs(self, output_json_path: str):
        """
        PHASE 2: Save RAG outputs back to JSON file.
        
        Args:
            output_json_path: Path to save the JSON with RAG outputs
        """
        if not self.evaluation_data:
            raise ValueError("No evaluation data to save. Run generate_rag_outputs() first.")
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved {len(self.evaluation_data)} questions with RAG outputs to '{output_json_path}'")

    def _initialize_judge_llm(self):
        """Initialize the LLM used for judging generation."""
        if self.judge_llm:
            return

        print(f"\n[Phase 3] Initializing Judge LLM...")
        if self.judge_llm_provider == "gemini":
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in .env file.")
            
            print(f"  - Using Gemini API for judge: {self.judge_llm_model}")
            self.judge_llm = GeminiChatModel(
                model_name=self.judge_llm_model,
                api_key=google_api_key,
                temperature=0.0 
            )
        elif self.judge_llm_provider == "ollama":
            print(f"  - Using Ollama for judge: {self.judge_llm_model}")
            self.judge_llm = ChatOllama(
                model=self.judge_llm_model, 
                timeout=800.0,
                temperature=0.0,
            )
        else:
            raise ValueError(f"Invalid judge_llm_provider: {self.judge_llm_provider}")
        
        print("  ✓ Judge LLM initialized successfully")

    def _query_rag(self, question: str) -> dict:
        """
        Query the RAG system and return answer with contexts.
        
        Args:
            question: Question to ask
            
        Returns:
            dict with 'answer' and 'contexts' keys
        """
        # Get answer from RAG system
        
        rag_response = self.rag_system.query_rerank_metadata(question)
        
        return {
            "answer": rag_response.get('answer', 'Error: No answer returned'),
            "contexts": rag_response.get('contexts', []) # Contexts from RAG
        }
    
    def _query_rag_gemini(self, question: str, gemini_api_key: str, 
                          model_name: str = "gemini-2.5-flash", 
                          temperature: float = 0.0) -> dict:
        """
        Query the RAG system using Gemini API with reranking and return answer with contexts.
        
        Args:
            question: Question to ask
            gemini_api_key: Google API key for Gemini
            model_name: Gemini model to use
            temperature: Temperature for generation
            
        Returns:
            dict with 'answer' and 'contexts' keys
        """
        # Get answer from RAG system using Gemini with reranking
        rag_response = self.rag_system.query_gemini_rerank(
            question=question,
            gemini_api_key=gemini_api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        return {
            "answer": rag_response.get('answer', 'Error: No answer returned'),
            "contexts": rag_response.get('contexts', []) # Contexts from RAG
        }
    
    def query_advanced_rag_ollama(self, question: str) -> dict:
        """
        Query the RAG system and return answer with contexts.
        
        Args:
            question: Question to ask
            
        Returns:
            dict with 'answer' and 'contexts' keys
        """
        # Get answer from RAG system
        self._initialize_rag_system()
        rag_response = self.rag_system.query_rerank_metadata(question)
        
        return {
            "answer": rag_response.get('answer', 'Error: No answer returned'),
            "contexts": rag_response.get('contexts', []) # Contexts from RAG
        }

    def save_evaluation_results(self, output_json_path: str, output_csv_path: str = None):
        """
        Save complete evaluation results with all metrics.
        
        Args:
            output_json_path: Path to save JSON with all data
            output_csv_path: Optional path to save CSV summary
        """
        if not self.evaluation_data:
            raise ValueError("No evaluation data to save.")
        
        print(f"\n[Phase 5] Saving evaluation results...")
        
        # Save complete JSON
        output_data = {
            "metadata": {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
                "judge_llm_model": self.ragas_llm_model,
                "judge_llm_provider": self.ragas_llm_provider,
                "retriever_k": self.retriever_k,
                "total_questions": len(self.evaluation_data)
            },
            "questions": self.evaluation_data
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved complete results to '{output_json_path}'")
        
        # Save CSV summary if requested
        if output_csv_path:
            df = pd.DataFrame(self.evaluation_data)
            df.to_csv(output_csv_path, index=False)
            print(f"  ✓ Saved CSV summary to '{output_csv_path}'")
        
        # Print summary statistics
        self._print_summary_statistics()
    

    
    
    