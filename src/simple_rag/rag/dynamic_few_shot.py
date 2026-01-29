import re
import shutil
import time
from typing import List, Dict
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import os
import json
import hashlib
class DynamicFewShotSelector:
    """
    Semantic similarity-based example selector for Cypher query generation.
    Embeds Q&A examples and retrieves the most similar ones for a given query.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", k: int = 2):
        """
        Args:
            embedding_model: Ollama model to use for embeddings (ensure you have pulled it: `ollama pull all-minilm`)
            k: Number of similar examples to retrieve (default: 3)
        """
        self.k = k
        self.embedding_model_name = embedding_model
        
        # 1. Define Paths relative to this file
        base_dir = Path(__file__).parent
        self.examples_path = base_dir / "examples" / "examples.json"
        self.index_path = base_dir / "examples" / "faiss_index"
        self.hash_file_path = self.index_path / "hash.md5"

        # 2. Setup Embeddings
        # Note: If you want to use the local HF model as discussed before, 
        # swap this for HuggingFaceEmbeddings(model_name=embedding_model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        self.vector_store = None
        
        # 3. Smart Load: Check hash, then Load or Build
        self._load_or_build_index()


    def _calculate_file_hash(self, filepath: Path) -> str:
        """Generates a unique MD5 fingerprint of the file content."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def _load_or_build_index(self):
        """
        Logic: Compare current file hash with the hash stored in the .md5 file.
        """
        if not self.examples_path.exists():
            raise FileNotFoundError(f"Could not find examples file at: {self.examples_path}")

        # A. Calculate current fingerprint
        current_hash = self._calculate_file_hash(self.examples_path)
        
        # B. Check if we have a valid cache
        if self.index_path.exists() and self.hash_file_path.exists():
            try:
                # CHANGE: Read the hash directly from the .md5 file
                with open(self.hash_file_path, 'r') as f:
                    stored_hash = f.read().strip()
                
                if stored_hash == current_hash:
                    print(f"✓ Cache valid (Hash: {current_hash[:8]}...). Loading vector index...")
                    self.vector_store = FAISS.load_local(
                        folder_path=str(self.index_path), 
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    return # SUCCESS
                else:
                    print(f"⚠ Content changed (Old: {stored_hash[:8]}... New: {current_hash[:8]}...). Rebuilding...")
            except Exception as e:
                print(f"⚠ Cache read error ({e}). Rebuilding...")
        
        # C. Rebuild if missing or mismatched
        self._build_index(current_hash)

    def _build_index(self, current_hash: str):
        """Reads JSON, computes embeddings, and saves index + .md5 file."""
        print(f"⟳ Computing embeddings for examples in {self.examples_path.name}...")
        
        # 1. Read JSON
        with open(self.examples_path, "r") as f:
            data = json.load(f)
            
        texts = [item["question"] for item in data]
        metadatas = [{"question": item["question"], "cypher": item["cypher"]} for item in data]
        
        # 2. Clear old index folder safely
        if self.index_path.exists():
            shutil.rmtree(self.index_path)

        # 3. Create Vector Store
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # 4. Save Index (creates the folder)
        self.vector_store.save_local(str(self.index_path))
        
        # CHANGE: Write the hash directly to a .md5 file
        with open(self.hash_file_path, 'w') as f:
            f.write(current_hash)
            
        print(f"✓ New index and hash saved to '{self.index_path}'")

    def get_similar_examples(self, query: str) -> List[Dict[str, str]]:
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")

        # Use MMR for diversity
        docs = self.vector_store.max_marginal_relevance_search(
            query, k=self.k, fetch_k=10
        )
        return [doc.metadata for doc in docs]
    
    def format_examples_as_string(self, examples: List[Dict[str, str]]) -> str:
        """
        Format retrieved examples as a string for inclusion in prompts.
        
        Args:
            examples: List of example dicts
            
        Returns:
            Formatted string with examples
        """
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Question: {ex['question']}")
            formatted.append(f"Cypher: {ex['cypher']}")
            formatted.append("")
        
        return "\n".join(formatted)

    def get_formatted_context(self, query: str) -> str:
        """
        Retrieves the top-k most relevant examples and formats them into a string.
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Index loading failed.")

        # 1. SEARCH: Use MMR (Maximal Marginal Relevance) for diversity
        start_time = time.time()
        docs = self.vector_store.max_marginal_relevance_search(
            query, 
            k=self.k, 
            fetch_k=10
        )
        search_time = time.time() - start_time
        print(f"⏱ Example search took {search_time:.3f}s")
        
        # 2. Extract metadata from Document objects
        examples = [doc.metadata for doc in docs]
        
        # 3. FORMAT: Iterate through the results and build the string
        return self.format_examples_as_string(examples)