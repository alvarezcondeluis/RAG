"""
Pluggable LLM provider modules for answer generation.

Supports multiple backends: Groq, OpenRouter, Google Gemini, and Ollama.
Each provider implements a common interface for listing models, generating
completions, and streaming responses.
"""

from simple_rag.rag.llm_providers.base import LLMProvider, LLMResponse
from simple_rag.rag.llm_providers.groq_provider import GroqProvider
from simple_rag.rag.llm_providers.openrouter_provider import OpenRouterProvider
from simple_rag.rag.llm_providers.gemini_provider import GeminiProvider
from simple_rag.rag.llm_providers.ollama_provider import OllamaProvider
from simple_rag.rag.llm_providers.registry import ProviderRegistry

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "GroqProvider",
    "OpenRouterProvider",
    "GeminiProvider",
    "OllamaProvider",
    "ProviderRegistry",
]
