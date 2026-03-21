"""
OpenRouter API provider.

Provides access to a wide range of models (OpenAI, Anthropic, Meta, Google, etc.)
through a single unified API compatible with the OpenAI SDK.
"""

import os
import time
import logging
from typing import Iterator, List, Dict, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from simple_rag.rag.llm_providers.base import LLMProvider, LLMResponse, ModelInfo

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider — access many models through one API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPEN_ROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPEN_ROUTER_API_KEY in .env or pass api_key."
            )
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/sec-filings-intelligence",
                "X-Title": "SEC Filings Intelligence",
            },
        )
        self.model_id = model_id or "meta-llama/llama-3.3-70b-instruct:free"

    @property
    def provider_name(self) -> str:
        return "OpenRouter"

    def list_models(self) -> List[ModelInfo]:
        """Fetch available models from the OpenRouter API."""
        try:
            resp = httpx.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            models = []
            for m in data.get("data", []):
                pricing = m.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "0") or "0")
                completion_price = float(pricing.get("completion", "0") or "0")

                # Flag free models
                is_free = prompt_price == 0 and completion_price == 0
                desc = m.get("name", "")
                if is_free:
                    desc += " [FREE]"

                models.append(
                    ModelInfo(
                        id=m["id"],
                        name=m.get("name", m["id"]),
                        context_window=m.get("context_length", 0),
                        description=desc,
                    )
                )

            # Sort: free models first, then by name
            models.sort(key=lambda x: (not x.description.endswith("[FREE]"), x.name))
            return models

        except Exception as e:
            logger.warning(f"Failed to fetch OpenRouter models: {e}")
            return [
                ModelInfo(
                    id="meta-llama/llama-3.3-70b-instruct:free",
                    name="Llama 3.3 70B (Free)",
                    context_window=128_000,
                    description="Free tier",
                ),
                ModelInfo(
                    id="google/gemini-2.0-flash-exp:free",
                    name="Gemini 2.0 Flash (Free)",
                    context_window=1_048_576,
                    description="Free tier",
                ),
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
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def test_connection(self) -> bool:
        try:
            resp = httpx.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False
