import json
import pandas as pd
import os
from datasets import Dataset
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Set environment variables to increase RAGAS timeouts
os.environ["RAGAS_TIMEOUT"] = "600"  # 10 minutes
os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # Disable telemetry

from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any
from .gemini import GeminiChatModel
from ..rag.rag import RAG
from ..retriever.retriever import Retriever
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
        collection_name: str = "simple_rag",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        llm_model: str = "llama3:8b",
        judge_llm_model: str = "llama3:8b",
        judge_llm_provider: str = "ollama",
        cache_folder: str = "./cache",
        retriever_k: int = 15,
        rerank_k: int = 3,
    ):
        """
        Initialize the RAGAS Evaluator.
        
        Args:
            collection_name: Qdrant collection name
            embedding_model: HuggingFace embedding model name
            llm_model: Ollama LLM model for RAG system
            judge_llm_model: LLM model name for RAGAS judge
                           - For Ollama: "llama3.1:8b", "qwen2.5:14b", etc.
                           - For Gemini: "gemini-1.5-flash", "gemini-1.5-pro", etc.
            judge_llm_provider: Provider for RAGAS judge LLM ("ollama" or "gemini")
            cache_folder: Cache folder for embedding models
            retriever_k: Number of contexts to retrieve
            rerank_k: Number of contexts to rerank
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.judge_llm_model = judge_llm_model
        self.judge_llm_provider = judge_llm_provider.lower()
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
        
        # Initialize embeddings for RAGAS
        self.ragas_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            cache_folder=cache_folder,
            model_kwargs={'trust_remote_code': True}
        )
        
        # RAG system components (initialized lazily)
        self.rag_system = None
        self.retriever = None
        self.retriever_k = retriever_k  # Number of contexts to retrieve
        self.judge_llm = None  # Separate judge LLM (initialized lazily)
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
        self.retriever = Retriever(
            vector_db=vector_db_instance,
            embed_model=embed_model_instance.model
        )
        self.rag_system = RAG(retriever=self.retriever, llm_name=self.llm_model)
        
        print("  ✓ RAG system initialized successfully")
    
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
        rag_response = self.rag_system.query_rerank(question)
        
        return {
            "answer": rag_response.get('answer', 'Error: No answer returned'),
            "contexts": rag_response.get('contexts', []) # Contexts from RAG
        }
    
    def compute_retrieval_metrics(self):
        """
        PHASE 3: Compute retrieval metrics (Context Precision & Context Recall).
        These metrics compare retrieved contexts with ground truth contexts.
        
        Returns:
            dict with average scores
        """
        if not self.evaluation_data:
            raise ValueError("No evaluation data. Run generate_rag_outputs() first.")
        
        print(f"\n[Phase 3] Computing retrieval metrics...")
        
        for idx, item in enumerate(self.evaluation_data):
            print(f"  Processing {idx+1}/{len(self.evaluation_data)}...")
            
            retrieved_contexts = item.get("contexts", [])
            ground_truth_contexts = item.get("ground_truth_contexts", [])
            
            # Context Precision: What fraction of retrieved contexts are relevant?
            # We check if retrieved context appears in ground truth contexts
            precision = self._calculate_context_precision(retrieved_contexts, ground_truth_contexts)
            
            # Context Recall: What fraction of ground truth contexts were retrieved?
            recall = self._calculate_context_recall(retrieved_contexts, ground_truth_contexts)
            
            # Store metrics
            item["context_precision"] = precision
            item["context_recall"] = recall
        
        # Calculate averages
        avg_precision = sum(item["context_precision"] for item in self.evaluation_data) / len(self.evaluation_data)
        avg_recall = sum(item["context_recall"] for item in self.evaluation_data) / len(self.evaluation_data)
        
        print(f"\n  ✓ Retrieval metrics computed")
        print(f"    - Avg Context Precision: {avg_precision:.4f}")
        print(f"    - Avg Context Recall: {avg_recall:.4f}")
        
        return {
            "context_precision": avg_precision,
            "context_recall": avg_recall
        }
    
    def _calculate_context_precision(self, retrieved_contexts: List[str], ground_truth_contexts: List[str]) -> float:
        """
        Calculate what fraction of retrieved contexts are relevant.
        A retrieved context is relevant if it has significant overlap with any ground truth context.
        """
        if not retrieved_contexts:
            return 0.0
        
        relevant_count = 0
        for ret_ctx in retrieved_contexts:
            # Check if this retrieved context overlaps with any ground truth context
            if self._has_context_overlap(ret_ctx, ground_truth_contexts):
                relevant_count += 1
        
        return relevant_count / len(retrieved_contexts)
    
    def _calculate_context_recall(self, retrieved_contexts: List[str], ground_truth_contexts: List[str]) -> float:
        """
        Calculate what fraction of ground truth contexts were retrieved.
        """
        if not ground_truth_contexts:
            return 1.0  # If no ground truth, consider it perfect
        
        retrieved_count = 0
        for gt_ctx in ground_truth_contexts:
            # Check if this ground truth context appears in retrieved contexts
            if self._has_context_overlap(gt_ctx, retrieved_contexts):
                retrieved_count += 1
        
        return retrieved_count / len(ground_truth_contexts)
    
    def _has_context_overlap(self, context: str, context_list: List[str], threshold: float = 0.5) -> bool:
        """
        Check if a context has significant overlap with any context in the list.
        Uses simple token overlap ratio.
        """
        context_tokens = set(context.lower().split())
        
        for other_context in context_list:
            other_tokens = set(other_context.lower().split())
            
            if not context_tokens or not other_tokens:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(context_tokens & other_tokens)
            union = len(context_tokens | other_tokens)
            
            if union > 0:
                similarity = intersection / union
                if similarity >= threshold:
                    return True
        
        return False
    
    def compute_llm_judged_metrics(self):
        """
        PHASE 4: Compute LLM-judged metrics (Faithfulness & Answer Correctness).
        Uses the judge LLM to evaluate answer quality.
        
        Returns:
            dict with average scores
        """
        if not self.evaluation_data:
            raise ValueError("No evaluation data. Run generate_rag_outputs() first.")
        
        print(f"\n[Phase 4] Computing LLM-judged metrics...")
        self._initialize_judge_llm()
        
        for idx, item in enumerate(self.evaluation_data):
            print(f"  Processing {idx+1}/{len(self.evaluation_data)}...")
            
            question = item["question"]
            answer = item["answer"]
            contexts = item["contexts"]
            ground_truth = item["ground_truth"]
            
            # Faithfulness: Is the answer faithful to the retrieved contexts?
            faithfulness_score = self._evaluate_faithfulness(question, answer, contexts)
            
            # Answer Correctness: How correct is the answer compared to ground truth?
            correctness_score = self._evaluate_answer_correctness(question, answer, ground_truth)
            
            # Store metrics
            item["faithfulness"] = faithfulness_score
            item["answer_correctness"] = correctness_score
        
        # Calculate averages
        avg_faithfulness = sum(item["faithfulness"] for item in self.evaluation_data) / len(self.evaluation_data)
        avg_correctness = sum(item["answer_correctness"] for item in self.evaluation_data) / len(self.evaluation_data)
        
        print(f"\n  ✓ LLM-judged metrics computed")
        print(f"    - Avg Faithfulness: {avg_faithfulness:.4f}")
        print(f"    - Avg Answer Correctness: {avg_correctness:.4f}")
        
        return {
            "faithfulness": avg_faithfulness,
            "answer_correctness": avg_correctness
        }
    
    def _evaluate_faithfulness(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if the answer is faithful to the retrieved contexts.
        Returns a score between 0 and 1.
        """
        contexts_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are an expert evaluator. Your task is to evaluate if an answer is faithful to the provided contexts.

        Question: {question}

        Retrieved Contexts:
        {contexts_text}

        Answer: {answer}

        Evaluate if the answer is faithful to the contexts. An answer is faithful if:
        1. All claims in the answer can be verified from the contexts
        2. The answer does not add information not present in the contexts
        3. The answer does not contradict the contexts

        Provide a faithfulness score between 0 and 1, where:
        - 0 = Completely unfaithful (contradicts or invents information)
        - 0.5 = Partially faithful (some claims supported, some not)
        - 1 = Completely faithful (all claims supported by contexts)

        Respond with ONLY a number between 0 and 1, nothing else."""
        
        try:
            response = self.judge_llm.invoke([HumanMessage(content=prompt)])
            score_text = response.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except Exception as e:
            print(f"    Warning: Faithfulness evaluation failed: {e}")
            return 0.5  # Default to neutral score
    
    def _evaluate_answer_correctness(self, question: str, answer: str, ground_truth: str) -> float:
        """
        Evaluate how correct the answer is compared to ground truth.
        Returns a score between 0 and 1.
        """
        prompt = f"""You are an expert evaluator. Your task is to evaluate the correctness of an answer compared to the ground truth.

        Question: {question}

        Ground Truth Answer: {ground_truth}

        Generated Answer: {answer}

        Evaluate how correct the generated answer is. Consider:
        1. Factual accuracy compared to ground truth
        2. Completeness of the answer
        3. Semantic similarity (same meaning even if different words)

        Provide a correctness score between 0 and 1, where:
        - 0 = Completely incorrect
        - 0.5 = Partially correct
        - 1 = Completely correct

        Respond with ONLY a number between 0 and 1, nothing else."""
        
        try:
            response = self.judge_llm.invoke([HumanMessage(content=prompt)])
            score_text = response.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except Exception as e:
            print(f"    Warning: Answer correctness evaluation failed: {e}")
            return 0.5  # Default to neutral score
    
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
    
    def _print_summary_statistics(self):
        """Print summary statistics of all metrics."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        metrics = ["context_precision", "context_recall", "faithfulness", "answer_correctness"]
        
        for metric in metrics:
            if metric in self.evaluation_data[0]:
                scores = [item[metric] for item in self.evaluation_data if metric in item]
                if scores:
                    avg = sum(scores) / len(scores)
                    min_score = min(scores)
                    max_score = max(scores)
                    print(f"\n{metric.replace('_', ' ').title()}:")
                    print(f"  Average: {avg:.4f}")
                    print(f"  Min: {min_score:.4f}")
                    print(f"  Max: {max_score:.4f}")
        
        print(f"\n{'='*60}")