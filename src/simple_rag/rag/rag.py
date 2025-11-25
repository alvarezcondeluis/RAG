from llama_index.llms.ollama import Ollama
import subprocess
import time
import requests
import threading
import os
from typing import List
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from google import genai
from google.genai import types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'evaluation'))
from gemini import GeminiChatModel
from langchain_core.messages import HumanMessage
import instructor
from openai import OpenAI
from ..utils.schemas import QueryEntities  
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import ValidationError
from qdrant_client import models

class RAG:
    def __init__(self, retriever, llm_name, auto_start_ollama=True, retriever_k=15, rerank_k=5):
        self.retriever = retriever
        self.llm_name = llm_name
        self.ollama_process = None
        self.retriever_k = retriever_k
        self.rerank_k = rerank_k
        reranker_model_name = "mixedbread-ai/mxbai-rerank-base-v1" 
    
        print(f"Loading reranker model '{reranker_model_name}'...")
        print("This may take a moment on first run...")
        
        try:
            self.reranker = SentenceTransformerRerank(
                model=reranker_model_name,
                top_n=self.rerank_k # The reranker will sort, our query func will slice
            )
            print("Reranker loaded successfully.")
        except Exception as e:
            print(f"--- WARNING: Failed to load reranker model ---")
            print(f"Error: {e}")
            print("Proceeding without reranker. Check if 'pip install llama-index-postprocessor-sbert-rerank' is run.")
            self.reranker = None


        # Check and start Ollama if needed
        if auto_start_ollama:
            self._ensure_ollama_running()

        # Set up the client to point to your local Ollama
        self.client = instructor.patch(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
        )
        
        self.llm = self._load_llm()
        self.prompt_tmpl = """
            Context information is below.
            ---------------------
            {context}
            ---------------------

            Given the context information above I want you
            to think step by step to answer the query in a
            crisp manner, incase case you can not answer with the given context
            say that the information is not included in the document.
            Do not say that you are using the context,
            just answer more or less with a summary using the information of the context.

            ---------------------
            Query: {query}
            ---------------------
            Answer:
                    """

    def _is_ollama_running(self):
        """Check if Ollama server is running."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def _start_ollama_server(self):
        """Start Ollama server in a subprocess."""
        try:
            print("Starting Ollama server...")
            # Start ollama serve in background
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Detach from parent process
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
    
    def _ensure_ollama_running(self):
        """Ensure Ollama server is running, start it if not."""
        if self._is_ollama_running():
            print("✓ Ollama server is already running")
            return True
        
        print("Ollama server not detected, attempting to start...")
        return self._start_ollama_server()
    
    def stop_ollama_server(self):
        """Stop the Ollama server if it was started by this instance."""
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

    def _load_llm(self):
        try:
             llm = Ollama(
                model=self.llm_name,
                request_timeout=800.0,
                timeout=120.0,  # 2 minutes for connection
                additional_kwargs={
                    "num_ctx": 4096,  # Smaller context = less memory
                    "num_predict": 1024,  # Limit output tokens
                }  
            )
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            return None
        print("LLM loaded successfully")
        return llm
    
    def warmup(self):
        """
        Warmup the Ollama model with a simple query.
        
        This loads the model into memory and ensures it's responsive,
        helping to avoid timeouts on the first real query.
        """
        try:
            print(f"Warming up Ollama model '{self.llm_name}'...")
            response = self.llm.chat(
                [ChatMessage(
                    role=MessageRole.USER,
                    content="Hello, respond with 'Ready'",
                )]
            )
            print(f"✓ Model warmed up. Response: {response}...")
            return response
        except Exception as e:
            print(f"⚠ Model warmup failed: {e}")
            return None
        

    def generate_context(self, query):
        results = self.retriever.search(query)
        combined_context = []             
        
        for result in results:
            # Extract text from the payload based on your Qdrant structure
            
           
            if hasattr(result, 'payload') and result.payload:
                text = result.payload.get('text', '')
                source = result.payload.get('source_document', 'Unknown')
                page = result.payload.get('page_number', 'N/A')
                section = result.payload.get('section_title', '')
                score = result.score
                
                print(f"Section: {section} score: {score}")
                
                # Format the context with metadata
                context_entry = f"[Source: {source}, Page: {page}]"
                if section:
                    context_entry += f" [Section: {section}]"
                context_entry += f"\n{text}"
                
                combined_context.append(context_entry)
        
        return "\n\n".join(combined_context)
    
    def query(self, question):
        # Generate context from retrieval
        results = self.retriever.search(question)
        combined_context = []
        context_texts = []
        
        for result in results:
            if hasattr(result, 'payload') and result.payload:
                text = result.payload.get('text', '')
                source = result.payload.get('source_document', 'Unknown')
                page = result.payload.get('page_number', 'N/A')
                section = result.payload.get('section_title', '')
                score = result.score
                
                print(f"Section: {section} score: {score}")
                
                context_entry = f"\n{text}"
                
                combined_context.append(context_entry)
                # Store raw text for evaluation
                context_texts.append(text)
        
        context = "\n\n".join(combined_context)
        
        # Format the prompt
        formatted_prompt = self.prompt_tmpl.format(
            context=context,
            query=question
        )
        
        # Generate response using LLM
        response = self.llm.chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content=formatted_prompt,
                )
            ]
        )
        
        return {
            'answer': response.text,
            'context': context,
            'contexts': context_texts  # Return list of raw context texts for evaluation
        }
    
    def query_gemini(self, question, gemini_api_key, model_name="gemini-2.5-flash", temperature=0.0):
        """
        Query method using Google Gemini API via GeminiChatModel wrapper.
        
        Args:
            question: The user's question
            gemini_api_key: Google API key for Gemini
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
            temperature: Temperature for generation (default: 0.0)
        
        Returns:
            Dictionary with 'answer', 'context', and 'contexts' keys
        """
        # Generate context from retrieval (same as original query method)
        results = self.retriever.search(question)
        combined_context = []
        context_texts = []
        
        for result in results:
            if hasattr(result, 'payload') and result.payload:
                text = result.payload.get('text', '')
                source = result.payload.get('source_document', 'Unknown')
                page = result.payload.get('page_number', 'N/A')
                section = result.payload.get('section_title', '')
                score = result.score
                
                print(f"Section: {section} score: {score}")
                
                context_entry = f"\n{text}"
                
                combined_context.append(context_entry)
                # Store raw text for evaluation
                context_texts.append(text)
        
        context = "\n\n".join(combined_context)
        
        # Format the prompt
        formatted_prompt = self.prompt_tmpl.format(
            context=context,
            query=question
        )
        
        # Generate response using GeminiChatModel wrapper
        try:
            gemini_model = GeminiChatModel(
                model_name=model_name,
                api_key=gemini_api_key,
                temperature=temperature
            )
            
            # Use the wrapper's _generate method with HumanMessage
            messages = [HumanMessage(content=formatted_prompt)]
            chat_result = gemini_model._generate(messages)
            answer = chat_result.generations[0].message.content
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            answer = f"Error generating response: {str(e)}"
        
        return {
            'answer': answer,
            'context': context,
            'contexts': context_texts  # Return list of raw context texts for evaluation
        }

    def _convert_results_to_nodes(self, qdrant_results) -> List[NodeWithScore]:
        """Converts Qdrant search results to LlamaIndex NodeWithScore objects."""
        nodes = []
        for result in qdrant_results:
            if hasattr(result, 'payload') and result.payload:
                # Re-create the TextNode from the payload
                node = TextNode(
                    text=result.payload.get('text', ''),
                    metadata=result.payload 
                )
                nodes.append(NodeWithScore(node=node, score=result.score))
        return nodes


    def query_rerank(self, question):
        # Generate context from retrieval
        start_total = time.time()
        
        print(f"\n{'='*60}")
        print(f"RERANKING QUERY: {question[:80]}...")
        print(f"{'='*60}")
        
        # Phase 1: Retrieval
        start_retrieval = time.time()
        results = self.retriever.search(question, limit=self.retriever_k)
        retrieval_time = time.time() - start_retrieval
        
        # Phase 2: Node conversion
        start_conversion = time.time()
        combined_context = []
        context_texts = []
        nodes = self._convert_results_to_nodes(results)
        conversion_time = time.time() - start_conversion
        
        print("")
        print(f"📥 Retrieved {len(nodes)} nodes from vector database (⏱️ {retrieval_time:.3f}s)")
        print(f"🔄 Converted to nodes (⏱️ {conversion_time:.3f}s)")
        
        # Show first 3 retrieved texts
        print(f"\n📄 Top 3 Retrieved Texts (before reranking):")
        for i, node in enumerate(nodes[:3]):
            text_preview = node.node.get_content()[:150].replace('\n', ' ')
            print(f"  [{i+1}] Score: {node.score:.4f}")
            print(f"      Text: {text_preview}...")
            print()
        
        # Phase 3: Reranking
        if self.reranker and nodes:
            print(f"🔄 Reranking {len(nodes)} nodes...")
            start_rerank = time.time()
            query_bundle = QueryBundle(question)
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes, 
                query_bundle=query_bundle
            )
            rerank_time = time.time() - start_rerank
            print(f"✓ Reranking complete. Top {self.rerank_k} nodes selected (⏱️ {rerank_time:.3f}s)")
            final_nodes = reranked_nodes[:self.rerank_k]
            
            # Show score comparison
            print(f"\n📊 Score comparison (before → after reranking):")
            for i, (orig, reranked) in enumerate(zip(nodes[:self.rerank_k], final_nodes)):
                print(f"  Position {i+1}: {orig.score:.4f} → {reranked.score:.4f}")
        else:
            if not self.reranker:
                print(f"⚠️  No reranker available, using top {self.rerank_k} from retrieval")
            final_nodes = nodes[:self.rerank_k]
            rerank_time = 0.0

        # Phase 4: Context building
        start_context = time.time()
        print(f"\n📝 Final context built from {len(final_nodes)} nodes")
        
        # Show final 3 reranked texts
        print(f"\n🎯 Top 3 Final Texts (after reranking):")
        for i, node in enumerate(final_nodes[:3]):
            text_preview = node.node.get_content()[:150].replace('\n', ' ')
            print(f"  [{i+1}] Score: {node.score:.4f}")
            print(f"      Text: {text_preview}...")
            print()
        
        context_texts = [node.node.get_content() for node in final_nodes]
        context_str = "\n\n---\n\n".join(context_texts)
        
        formatted_prompt = self.prompt_tmpl.format(
            context=context_str,
            query=question
        )
        context_time = time.time() - start_context
        print(f"✓ Context built (⏱️ {context_time:.3f}s)")
        
        # Phase 5: LLM generation
        print(f"\n🤖 Generating answer with LLM...")
        start_llm = time.time()
        response = self.llm.chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content=formatted_prompt,
                )
            ]
        )
        llm_time = time.time() - start_llm
        print(f"✓ Answer generated (⏱️ {llm_time:.3f}s)")
        
        # Total time
        total_time = time.time() - start_total
        print(f"\n{'='*60}")
        print(f"⏱️  TOTAL PIPELINE TIME: {total_time:.3f}s")
        print(f"   - Retrieval: {retrieval_time:.3f}s ({retrieval_time/total_time*100:.1f}%)")
        print(f"   - Conversion: {conversion_time:.3f}s ({conversion_time/total_time*100:.1f}%)")
        print(f"   - Reranking: {rerank_time:.3f}s ({rerank_time/total_time*100:.1f}%)")
        print(f"   - Context: {context_time:.3f}s ({context_time/total_time*100:.1f}%)")
        print(f"   - LLM: {llm_time:.3f}s ({llm_time/total_time*100:.1f}%)")
        print(f"{'='*60}")
        
        return {
            'answer': response.text,
            'context': context_str,
            'contexts': context_texts
        }


    
    def extract_entities(self, query: str) -> QueryEntities:
        """
        Uses a fast, local model to extract entities from a raw query.
        """
        print(f"--- 1. Processing query: '{query}' ---")
        try:
            # Explicitly prompt for JSON without markdown
            prompt = f"""You are an expert financial query analyst. Extract key entities from the user's query.

                You MUST respond with *only* a valid JSON object.
                Do NOT add any text, explanations, or markdown formatting (like ```json) before or after the JSON object.
                Omit the Vanguard part of the name of the fund or ETF.

                The JSON schema must be:
                {{
                "tickers": ["...", "..."] or null,
                "fund_names": ["...", "..."] or null,
                "metric": "..." or null,
                "timespan": "..." or null
                }}

                Query: {query}
                JSON Response:"""
            
            response = self.client.chat.completions.create(
                model="qwen3:4b-instruct",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract and clean the raw JSON
            raw_json = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            raw_json = raw_json.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON into Pydantic model
            entities = QueryEntities.model_validate_json(raw_json)
            return entities
            
        except ValidationError as ve:
            print(f"JSON validation error during entity extraction: {ve}")
            return QueryEntities()
        except Exception as e:
            print(f"Error during entity extraction: {e}")
            return QueryEntities()


    def extract_entities_small(self, query: str) -> QueryEntities:
        """
        Uses a fast, local model to extract entities from a raw query.
        """
        print(f"--- 1. Processing query: '{query}' ---")
        try:

            JSON_PROMPT = """
                You are an expert financial query analyst. Your task is to extract key entities from the user's query.

                You MUST respond with *only* a valid JSON object.
                Do NOT add any text, explanations, or markdown formatting (like ```json) before or after the JSON object.

                The JSON schema must be:
                {
                "tickers": ["...", "..."_] or null,
                "fund_names": ["...", "..."_] or null,
                "metric": "..." or null,
                "timespan": "..." or null
                }

                Query: {query}
                JSON Response:
                """
            entities = self.client.chat.completions.create(
                model="phi3:3.8b",  # Your fast local model
                messages=[
                    {"role": "system", "content": "You are an expert financial query analyst. Extract key entities (tickers (e.g., VHYAX), fund names, metrics, timespans) from the user's query and return valid JSON."},
                    {"role": "user", "content": query}
                ]
            )
            raw_text_response = response.choices[0].message.content
            print(f"--- Raw LLM response: {raw_text_response} ---")
            return entities
            
        except Exception as e:
            print(f"Error during entity extraction: {e}")
            return QueryEntities() 


    def query_rerank_metadata(self, question, small=False):
        # Generate context from retrieval


        start_total = time.time()
        
        print(f"\n{'='*60}")
        print(f"RERANKING QUERY: {question[:80]}...")
        print(f"{'='*60}")


        # Phase 1: Query Preprocessing
        start_preprocessing = time.time()
        if small:
            entities = self.extract_entities_small(question)
        else:
            entities = self.extract_entities(question)
        preprocessing_time = time.time() - start_preprocessing
        print("\n Entities extracted:")
        print(f"Preprocessing time: {preprocessing_time:.3f}s")
        print(entities.model_dump_json(indent=2))
        
        
        # Phase 1: Retrieval

        try:
            if entities:
                name = entities.fund_names[0] if entities.fund_names else None
                
                if name is not None:
                    if len(name) > 7:
                        name_filter = models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.name",
                                    match=models.MatchValue(value=name)
                                )
                            ]
                        )
                    else:
                        name_filter = None
                else:
                        name_filter = None

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return {
                "answer": "The information is not included in the document.",
                "contexts": []
            }
        start_retrieval = time.time()
        results = self.retriever.search(question, limit=self.retriever_k, metadata_filter=name_filter)
        retrieval_time = time.time() - start_retrieval
        
        # Phase 2: Node conversion
        start_conversion = time.time()
        combined_context = []
        context_texts = []
        nodes = self._convert_results_to_nodes(results)
        conversion_time = time.time() - start_conversion
        
        print("")
        print(f"📥 Retrieved {len(nodes)} nodes from vector database (⏱️ {retrieval_time:.3f}s)")
        print(f"🔄 Converted to nodes (⏱️ {conversion_time:.3f}s)")
        
        # Show first 3 retrieved texts
        print(f"\n📄 Top 3 Retrieved Texts (before reranking):")
        for i, node in enumerate(nodes[:3]):
            text_preview = node.node.get_content()[:150].replace('\n', ' ')
            print(f"  [{i+1}] Score: {node.score:.4f}")
            print(f"      Text: {text_preview}...")
            print()
        
        # Phase 3: Reranking
        if self.reranker and nodes:
            print(f"🔄 Reranking {len(nodes)} nodes...")
            start_rerank = time.time()
            query_bundle = QueryBundle(question)
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes, 
                query_bundle=query_bundle
            )
            rerank_time = time.time() - start_rerank
            print(f"✓ Reranking complete. Top {self.rerank_k} nodes selected (⏱️ {rerank_time:.3f}s)")
            final_nodes = reranked_nodes[:self.rerank_k]
            
            # Show score comparison
            print(f"\n📊 Score comparison (before → after reranking):")
            for i, (orig, reranked) in enumerate(zip(nodes[:self.rerank_k], final_nodes)):
                print(f"  Position {i+1}: {orig.score:.4f} → {reranked.score:.4f}")
        else:
            if not self.reranker:
                print(f"⚠️  No reranker available, using top {self.rerank_k} from retrieval")
            final_nodes = nodes[:self.rerank_k]
            rerank_time = 0.0

        # Phase 4: Context building
        start_context = time.time()
        print(f"\n📝 Final context built from {len(final_nodes)} nodes")
        
        # Show final 3 reranked texts
        print(f"\n🎯 Top 3 Final Texts (after reranking):")
        for i, node in enumerate(final_nodes[:3]):
            text_preview = node.node.get_content()[:150].replace('\n', ' ')
            print(f"  [{i+1}] Score: {node.score:.4f}")
            print(f"      Text: {text_preview}...")
            print()
        
        context_texts = [node.node.get_content() for node in final_nodes]
        context_str = "\n\n---\n\n".join(context_texts)
        
        formatted_prompt = self.prompt_tmpl.format(
            context=context_str,
            query=question
        )
        
        context_time = time.time() - start_context
        print(f"✓ Context built (⏱️ {context_time:.3f}s)")
        
        # Phase 5: LLM generation
        print(f"\n🤖 Generating answer with LLM...")
        start_llm = time.time()
        response = self.llm.chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content=formatted_prompt,
                )
            ]
        )
        llm_time = time.time() - start_llm
        print(f"✓ Answer generated (⏱️ {llm_time:.3f}s)")
        
        # Total time
        total_time = time.time() - start_total
        print(f"\n{'='*60}")
        print(f"⏱️  TOTAL PIPELINE TIME: {total_time:.3f}s")
        print(f"   - Preprocessing: {preprocessing_time:.3f}s ({preprocessing_time/total_time*100:.1f}%)")
        print(f"   - Retrieval: {retrieval_time:.3f}s ({retrieval_time/total_time*100:.1f}%)")
        print(f"   - Conversion: {conversion_time:.3f}s ({conversion_time/total_time*100:.1f}%)")
        print(f"   - Reranking: {rerank_time:.3f}s ({rerank_time/total_time*100:.1f}%)")
        print(f"   - Context: {context_time:.3f}s ({context_time/total_time*100:.1f}%)")
        print(f"   - LLM: {llm_time:.3f}s ({llm_time/total_time*100:.1f}%)")
        print(f"{'='*60}")
        

        print(response)
        return {
            'answer': response.message.content,
            'context': context_str,
            'contexts': context_texts
        }
    
    def query_gemini_rerank(self, question, gemini_api_key, model_name="gemini-2.5-flash", temperature=0.0):
        """
        Query method using Google Gemini API with reranking.
        
        Args:
            question: The user's question
            gemini_api_key: Google API key for Gemini
            model_name: Gemini model to use (default: gemini-2.5-flash)
            temperature: Temperature for generation (default: 0.0)
        
        Returns:
            Dictionary with 'answer', 'context', and 'contexts' keys
        """
        # Generate context from retrieval
        start_total = time.time()
        
        print(f"\n{'='*60}")
        print(f"GEMINI RERANKING QUERY: {question[:80]}...")
        print(f"{'='*60}")
        
        # Phase 1: Retrieval
        start_retrieval = time.time()
        results = self.retriever.search(question, limit=self.retriever_k)
        retrieval_time = time.time() - start_retrieval
        
        # Phase 2: Node conversion
        start_conversion = time.time()
        combined_context = []
        context_texts = []
        nodes = self._convert_results_to_nodes(results)
        conversion_time = time.time() - start_conversion
        
        print("")
        print(f"📥 Retrieved {len(nodes)} nodes from vector database (⏱️ {retrieval_time:.3f}s)")
        print(f"🔄 Converted to nodes (⏱️ {conversion_time:.3f}s)")
        
        # Show first 3 retrieved texts
        print(f"\n📄 Top 3 Retrieved Texts (before reranking):")
        for i, node in enumerate(nodes[:3]):
            text_preview = node.node.get_content()[:150].replace('\n', ' ')
            print(f"  [{i+1}] Score: {node.score:.4f}")
            print(f"      Text: {text_preview}...")
            print()
        
        # Phase 3: Reranking
        if self.reranker and nodes:
            print(f"🔄 Reranking {len(nodes)} nodes...")
            start_rerank = time.time()
            query_bundle = QueryBundle(question)
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes, 
                query_bundle=query_bundle
            )
            rerank_time = time.time() - start_rerank
            print(f"✓ Reranking complete. Top {self.rerank_k} nodes selected (⏱️ {rerank_time:.3f}s)")
            final_nodes = reranked_nodes[:self.rerank_k]
            
            # Show score comparison
            print(f"\n📊 Score comparison (before → after reranking):")
            for i, (orig, reranked) in enumerate(zip(nodes[:self.rerank_k], final_nodes)):
                print(f"  Position {i+1}: {orig.score:.4f} → {reranked.score:.4f}")
        else:
            if not self.reranker:
                print(f"⚠️  No reranker available, using top {self.rerank_k} from retrieval")
            final_nodes = nodes[:self.rerank_k]
            rerank_time = 0.0

        # Phase 4: Context building
        start_context = time.time()
        print(f"\n📝 Final context built from {len(final_nodes)} nodes")
        
        # Show final 3 reranked texts
        print(f"\n🎯 Top 3 Final Texts (after reranking):")
        for i, node in enumerate(final_nodes[:3]):
            text_preview = node.node.get_content()[:150].replace('\n', ' ')
            print(f"  [{i+1}] Score: {node.score:.4f}")
            print(f"      Text: {text_preview}...")
            print()
        
        context_texts = [node.node.get_content() for node in final_nodes]
        context_str = "\n\n---\n\n".join(context_texts)
        
        formatted_prompt = self.prompt_tmpl.format(
            context=context_str,
            query=question
        )
        context_time = time.time() - start_context
        print(f"✓ Context built (⏱️ {context_time:.3f}s)")
        
        # Phase 5: Gemini LLM generation
        print(f"\n🤖 Generating answer with Gemini ({model_name})...")
        start_llm = time.time()
        
        try:
            gemini_model = GeminiChatModel(
                model_name=model_name,
                api_key=gemini_api_key,
                temperature=temperature
            )
            
            # Use the wrapper's _generate method with HumanMessage
            messages = [HumanMessage(content=formatted_prompt)]
            chat_result = gemini_model._generate(messages)
            answer = chat_result.generations[0].message.content
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            answer = f"Error generating response: {str(e)}"
        
        llm_time = time.time() - start_llm
        print(f"✓ Answer generated (⏱️ {llm_time:.3f}s)")
        
        # Total time
        total_time = time.time() - start_total
        print(f"\n{'='*60}")
        print(f"⏱️  TOTAL PIPELINE TIME: {total_time:.3f}s")
        print(f"   - Retrieval: {retrieval_time:.3f}s ({retrieval_time/total_time*100:.1f}%)")
        print(f"   - Conversion: {conversion_time:.3f}s ({conversion_time/total_time*100:.1f}%)")
        print(f"   - Reranking: {rerank_time:.3f}s ({rerank_time/total_time*100:.1f}%)")
        print(f"   - Context: {context_time:.3f}s ({context_time/total_time*100:.1f}%)")
        print(f"   - Gemini LLM: {llm_time:.3f}s ({llm_time/total_time*100:.1f}%)")
        print(f"{'='*60}")
        
        return {
            'answer': answer,
            'context': context_str,
            'contexts': context_texts
        }
        
        
    
    def __del__(self):
        """Cleanup: Stop Ollama server if it was started by this instance."""
        self.stop_ollama_server()