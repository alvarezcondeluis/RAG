"""
Groq API wrapper for LangChain integration.
Provides access to Groq's fast inference API with models like Llama 3.3 70B.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file (search in parent directories)
dotenv_path = Path("../.env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)
    print(f"✓ Loaded .env from: {dotenv_path}")
else:
    # Try to find .env in project root (go up from current file)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent  # Go up to RAG directory
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded .env from: {env_file}")
    else:
        print("⚠️  No .env file found. Will use environment variables or require api_key parameter.")


class GroqWrapper:
    """
    Wrapper for Groq API integration with LangChain.
    
    Supports models like:
    - llama-3.3-70b-versatile (Llama 3.3 70B)
    - llama-3.1-70b-versatile (Llama 3.1 70B)
    - mixtral-8x7b-32768 (Mixtral 8x7B)
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048
    ):
        """
        Initialize Groq API wrapper.
        
        Args:
            model_name: Groq model identifier
            api_key: Groq API key (if None, will load from .env file or GROQ_API_KEY env var)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get API key from parameter, .env file, or environment variable
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Groq API key not found. Please either:\n"
                    "1. Add GROQ_API_KEY to your .env file, or\n"
                    "2. Set GROQ_API_KEY environment variable, or\n"
                    "3. Pass api_key parameter directly"
                )
        
        # Initialize ChatGroq
        self.llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        print(f"✓ Groq API initialized with model: {model_name}")
    
    def get_llm(self):
        """Return the LangChain-compatible LLM instance."""
        return self.llm
    
    @staticmethod
    def list_available_models():
        """List commonly available Groq models."""
        return [
            "llama-3.3-70b-versatile",  # Llama 3.3 70B (recommended)
            "llama-3.1-70b-versatile",  # Llama 3.1 70B
            "llama-3.1-8b-instant",     # Llama 3.1 8B (faster)
            "mixtral-8x7b-32768",       # Mixtral 8x7B
            "gemma2-9b-it",             # Gemma 2 9B
        ]
