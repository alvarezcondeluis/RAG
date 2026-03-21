"""
Google AI Studio (Gemini) provider.

Uses the google-genai SDK for access to Gemini models.
"""

import os
import time
import logging
from typing import Iterator, List, Dict, Optional

from dotenv import load_dotenv

from simple_rag.rag.llm_providers.base import LLMProvider, LLMResponse, ModelInfo

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google AI Studio (Gemini) provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google AI Studio API key not found. "
                "Set GOOGLE_AI_STUDIO_API_KEY in .env or pass api_key."
            )

        from google import genai

        self.client = genai.Client(api_key=self.api_key)
        self.model_id = model_id or "gemini-2.0-flash"

    @property
    def provider_name(self) -> str:
        return "Gemini"

    def list_models(self) -> List[ModelInfo]:
        """Fetch available Gemini models from the API."""
        try:
            raw_models = self.client.models.list()
            models = []
            for m in raw_models:
                # Filter to generative models only
                name = m.name  # e.g. "models/gemini-2.0-flash"
                model_id = name.replace("models/", "") if name.startswith("models/") else name

                # Skip embedding and legacy models
                if "embedding" in model_id or "aqa" in model_id:
                    continue

                display_name = m.display_name if hasattr(m, "display_name") else model_id

                input_limit = 0
                if hasattr(m, "input_token_limit"):
                    input_limit = m.input_token_limit or 0

                models.append(
                    ModelInfo(
                        id=model_id,
                        name=display_name,
                        context_window=input_limit,
                        description=m.description if hasattr(m, "description") else "",
                    )
                )

            models.sort(key=lambda x: x.id)
            return models

        except Exception as e:
            logger.warning(f"Failed to fetch Gemini models: {e}")
            return [
                ModelInfo(id="gemini-2.0-flash", name="Gemini 2.0 Flash", context_window=1_048_576),
                ModelInfo(id="gemini-2.5-flash-preview-05-20", name="Gemini 2.5 Flash Preview", context_window=1_048_576),
                ModelInfo(id="gemini-2.5-pro-preview-05-06", name="Gemini 2.5 Pro Preview", context_window=1_048_576),
            ]

    def _build_contents(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], str]:
        """Convert messages list to system instruction + user prompt for Gemini."""
        system_instruction = None
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return system_instruction, "\n\n".join(parts)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        from google.genai import types

        system_instruction, prompt = self._build_contents(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        start = time.time()
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=config,
        )
        latency = (time.time() - start) * 1000

        # Extract token counts from usage metadata
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return LLMResponse(
            content=response.text or "",
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
        from google.genai import types

        system_instruction, prompt = self._build_contents(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self.client.models.generate_content_stream(
            model=self.model_id,
            contents=prompt,
            config=config,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def test_connection(self) -> bool:
        try:
            models = self.client.models.list()
            # Check we got at least one model
            return any(True for _ in models)
        except Exception:
            return False
