"""
Ollama provider for local LLM inference.

Connects to a locally running Ollama server. Note that if the text2cypher
model is already loaded, available VRAM may be limited.
"""

import time
import logging
from typing import Iterator, List, Dict, Optional

import httpx

from simple_rag.rag.llm_providers.base import LLMProvider, LLMResponse, ModelInfo

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model_id: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id or "llama3.1:8b"

    @property
    def provider_name(self) -> str:
        return "Ollama"

    def list_models(self) -> List[ModelInfo]:
        """Fetch locally available models from the Ollama server."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()

            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                size_gb = m.get("size", 0) / (1024**3)
                desc = f"{size_gb:.1f} GB"
                if m.get("details", {}).get("parameter_size"):
                    desc = f"{m['details']['parameter_size']} — {desc}"

                models.append(
                    ModelInfo(
                        id=name,
                        name=name,
                        context_window=m.get("details", {}).get("context_length", 0),
                        description=desc,
                    )
                )

            models.sort(key=lambda x: x.id)
            return models

        except httpx.ConnectError:
            logger.warning("Ollama server not running at %s", self.base_url)
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
            return []

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        start = time.time()
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_id,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = (time.time() - start) * 1000

        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self.model_id,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency,
        )

    def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_id,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        ) as resp:
            import json as _json

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = _json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except _json.JSONDecodeError:
                    continue

    def test_connection(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
