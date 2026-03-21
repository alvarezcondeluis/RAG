"""
Provider registry — discovers available providers and handles interactive model selection.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Type

from dotenv import load_dotenv

from simple_rag.rag.llm_providers.base import LLMProvider, ModelInfo

logger = logging.getLogger(__name__)

# Provider name -> (class, env var for API key, requires key)
_PROVIDER_SPECS: Dict[str, Tuple[str, Optional[str], bool]] = {
    "groq": ("simple_rag.rag.llm_providers.groq_provider.GroqProvider", "GROQ_API_KEY", True),
    "openrouter": ("simple_rag.rag.llm_providers.openrouter_provider.OpenRouterProvider", "OPEN_ROUTER_API_KEY", True),
    "gemini": ("simple_rag.rag.llm_providers.gemini_provider.GeminiProvider", "GOOGLE_AI_STUDIO_API_KEY", True),
    "ollama": ("simple_rag.rag.llm_providers.ollama_provider.OllamaProvider", None, False),
}


def _import_class(dotted_path: str) -> Type[LLMProvider]:
    """Import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ProviderRegistry:
    """Manages available LLM providers and interactive selection."""

    def __init__(self):
        load_dotenv()

    def available_providers(self) -> List[Dict[str, str]]:
        """Return providers with their API key status.

        Returns:
            List of dicts: {"name", "display", "key_status"} where key_status
            is "ok", "missing", or "n/a" (for providers that don't need a key).
        """
        result = []
        for name, (_, env_var, requires_key) in _PROVIDER_SPECS.items():
            if requires_key and env_var:
                key = os.getenv(env_var, "")
                status = "ok" if key else "missing"
            else:
                status = "n/a"

            display_map = {
                "groq": "Groq",
                "openrouter": "OpenRouter",
                "gemini": "Google Gemini",
                "ollama": "Ollama (local)",
            }
            result.append({
                "name": name,
                "display": display_map.get(name, name),
                "key_status": status,
            })
        return result

    def get_provider(self, name: str, model_id: Optional[str] = None, **kwargs) -> LLMProvider:
        """Instantiate a provider by name.

        Args:
            name: Provider key (groq, openrouter, gemini, ollama).
            model_id: Optional model ID to pre-select.

        Returns:
            Configured LLMProvider instance.
        """
        if name not in _PROVIDER_SPECS:
            raise ValueError(f"Unknown provider '{name}'. Available: {list(_PROVIDER_SPECS.keys())}")

        dotted_path, _, _ = _PROVIDER_SPECS[name]
        cls = _import_class(dotted_path)

        init_kwargs = {**kwargs}
        if model_id:
            init_kwargs["model_id"] = model_id

        return cls(**init_kwargs)

    def interactive_select(
        self,
        prompt_label: str = "LLM provider",
        filter_available: bool = True,
    ) -> Tuple[LLMProvider, str]:
        """Interactive terminal-based provider and model selection.

        Args:
            prompt_label: Label shown in the selection prompt.
            filter_available: If True, only show providers with valid API keys.

        Returns:
            Tuple of (provider_instance, selected_model_id).
        """
        providers = self.available_providers()
        if filter_available:
            providers = [p for p in providers if p["key_status"] != "missing"]

        if not providers:
            raise RuntimeError("No providers available. Check your API keys in .env")

        # Step 1: Select provider
        print(f"\n─── {prompt_label} ───")
        for i, p in enumerate(providers, 1):
            status_icon = {
                "ok": "\033[92m✓\033[0m",      # green check
                "n/a": "\033[93m●\033[0m",     # yellow dot
                "missing": "\033[91m✗\033[0m",  # red x
            }.get(p["key_status"], "?")
            extra = ""
            if p["name"] == "ollama":
                extra = " \033[93m(⚠ VRAM limited)\033[0m"
            print(f"  [{i}] {p['display']:<20s} (API key: {status_icon}){extra}")

        while True:
            try:
                choice = input(f"> Select {prompt_label}: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(providers):
                    break
                print(f"  Please enter a number between 1 and {len(providers)}")
            except (ValueError, EOFError):
                print(f"  Please enter a number between 1 and {len(providers)}")

        selected = providers[idx]
        provider = self.get_provider(selected["name"])

        # Step 2: Fetch and select model
        print(f"\n  Fetching models from {selected['display']}...")
        models = provider.list_models()

        if not models:
            print("  \033[91mNo models available from this provider.\033[0m")
            raise RuntimeError(f"No models available from {selected['display']}")

        # Show up to 30 models
        display_models = models[:30]
        for i, m in enumerate(display_models, 1):
            ctx = f" ({m.context_window:,} ctx)" if m.context_window else ""
            desc = f" — {m.description}" if m.description else ""
            print(f"  [{i}] {m.id}{ctx}{desc}")

        if len(models) > 30:
            print(f"  ... and {len(models) - 30} more")

        while True:
            try:
                choice = input("> Select model: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(display_models):
                    break
                print(f"  Please enter a number between 1 and {len(display_models)}")
            except (ValueError, EOFError):
                print(f"  Please enter a number between 1 and {len(display_models)}")

        selected_model = display_models[idx]
        provider.model_id = selected_model.id
        print(f"  ✓ Selected: {selected_model.id}")

        return provider, selected_model.id
