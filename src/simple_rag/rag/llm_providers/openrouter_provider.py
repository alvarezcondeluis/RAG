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
        base_url: Optional[str] = None,
    ):
        load_dotenv()
        self._base_url = base_url or OPENROUTER_BASE_URL
        is_local = not self._base_url.startswith("https://openrouter")
        self.api_key = api_key or (None if is_local else os.getenv("OPEN_ROUTER_API_KEY"))
        if not is_local and not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPEN_ROUTER_API_KEY in .env or pass api_key."
            )
        client_kwargs: dict = {"base_url": self._base_url, "api_key": self.api_key or "local"}
        if not is_local:
            client_kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/sec-filings-intelligence",
                "X-Title": "SEC Filings Intelligence",
            }
        self.client = OpenAI(**client_kwargs)
        self.model_id = model_id or ("meta-llama/llama-3.3-70b-instruct:free" if not is_local else self._detect_model())

    def _detect_model(self) -> str:
        models = self.list_models()
        if models:
            logger.info("Auto-selected model '%s' from %s", models[0].id, self._base_url)
            return models[0].id
        return "local-model"

    @property
    def provider_name(self) -> str:
        return "OpenRouter" if self._base_url == OPENROUTER_BASE_URL else f"OpenAI-compat ({self._base_url})"

    def list_models(self) -> List[ModelInfo]:
        """Fetch available models from the API."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key and self.api_key != "local" else {}
            resp = httpx.get(
                f"{self._base_url}/models",
                headers=headers,
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
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key and self.api_key != "local" else {}
            resp = httpx.get(f"{self._base_url}/models", headers=headers, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False
