"""
Groq API provider for fast inference.

Uses the Groq cloud API with models like Llama 3.3 70B, Gemma 2 9B, etc.
Free tier includes rate limits tracked automatically.
"""

import os
import time
import logging
from typing import Iterator, List, Dict, Optional

import httpx
from dotenv import load_dotenv
from groq import Groq

from simple_rag.rag.llm_providers.base import LLMProvider, LLMResponse, ModelInfo

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq cloud API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY in .env or pass api_key."
            )
        self.client = Groq(api_key=self.api_key)
        self.model_id = model_id or "llama-3.3-70b-versatile"

    @property
    def provider_name(self) -> str:
        return "Groq"

    def list_models(self) -> List[ModelInfo]:
        """Fetch available models from the Groq API."""
        try:
            resp = httpx.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            models = []
            for m in data.get("data", []):
                # Only include active chat models
                if not m.get("active", True):
                    continue
                models.append(
                    ModelInfo(
                        id=m["id"],
                        name=m.get("id", ""),
                        context_window=m.get("context_window", 0),
                        description=m.get("owned_by", ""),
                    )
                )

            models.sort(key=lambda x: x.id)
            return models

        except Exception as e:
            logger.warning(f"Failed to fetch Groq models: {e}")
            # Fallback to known models
            return [
                ModelInfo(id="llama-3.3-70b-versatile", name="Llama 3.3 70B", context_window=128_000),
                ModelInfo(id="llama-3.1-8b-instant", name="Llama 3.1 8B", context_window=128_000),
                ModelInfo(id="gemma2-9b-it", name="Gemma 2 9B", context_window=8_192),
                ModelInfo(id="mixtral-8x7b-32768", name="Mixtral 8x7B", context_window=32_768),
            ]

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        latency = (time.time() - start) * 1000

        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model_id,
            provider=self.provider_name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency,
        )

    def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def test_connection(self) -> bool:
        try:
            resp = httpx.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False
