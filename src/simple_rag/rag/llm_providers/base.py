"""
Abstract base class for LLM providers.

All providers (Groq, OpenRouter, Gemini, Ollama) implement this interface,
enabling pluggable model selection for answer generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Dict, Optional


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class ModelInfo:
    """Metadata for an available model."""
    id: str
    name: str
    context_window: int = 0
    description: str = ""


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """Fetch available models from the provider.

        Returns:
            List of ModelInfo with id, name, context_window, and description.
        """
        ...

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a completion from a list of messages.

        Args:
            messages: List of dicts with 'role' (system/user/assistant) and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with content and usage metadata.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Stream tokens one by one.

        Args:
            messages: List of dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Text chunks as they are generated.
        """
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Verify the provider is reachable and the API key is valid.

        Returns:
            True if the connection succeeds, False otherwise.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'Groq', 'OpenRouter')."""
        ...
