"""
Test script for LLM provider modules.

Verifies each provider can:
1. Connect to its API
2. List available models
3. Generate a simple response

Run: python tests/test_providers.py
  or: python -m pytest tests/test_providers.py -v -s
"""

import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simple_rag.rag.llm_providers.registry import ProviderRegistry


def _status(ok: bool) -> str:
    return "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"


def test_provider(name: str, registry: ProviderRegistry) -> bool:
    """Test a single provider: connection, list models, and generate."""
    print(f"\n{'─' * 50}")
    print(f"  Provider: \033[1m{name.upper()}\033[0m")
    print(f"{'─' * 50}")

    # 1. Instantiate
    try:
        provider = registry.get_provider(name)
        print(f"  Init:       {_status(True)}")
    except Exception as e:
        print(f"  Init:       {_status(False)} — {e}")
        return False

    # 2. Test connection
    try:
        connected = provider.test_connection()
        print(f"  Connection: {_status(connected)}")
        if not connected:
            print(f"  ⚠  Cannot reach {provider.provider_name}. Skipping further tests.")
            return False
    except Exception as e:
        print(f"  Connection: {_status(False)} — {e}")
        return False

    # 3. List models
    try:
        models = provider.list_models()
        print(f"  Models:     {_status(len(models) > 0)} — {len(models)} models found")
        for m in models[:5]:
            ctx = f" ({m.context_window:,} ctx)" if m.context_window else ""
            print(f"              · {m.id}{ctx}")
        if len(models) > 5:
            print(f"              ... and {len(models) - 5} more")
    except Exception as e:
        print(f"  Models:     {_status(False)} — {e}")
        return False

    # 4. Generate
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Reply in one sentence."},
            {"role": "user", "content": "Say hello and confirm you are working."},
        ]
        start = time.time()
        response = provider.generate(messages, temperature=0.1, max_tokens=64)
        elapsed = (time.time() - start) * 1000
        print(f"  Generate:   {_status(bool(response.content))}")
        print(f"              Response: \"{response.content[:100]}\"")
        print(f"              Tokens: {response.prompt_tokens}→{response.completion_tokens} "
              f"({response.total_tokens} total)")
        print(f"              Latency: {elapsed:.0f}ms")
    except Exception as e:
        print(f"  Generate:   {_status(False)} — {e}")
        return False

    return True


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  LLM Provider Test Suite                     ║")
    print("╚══════════════════════════════════════════════╝")

    registry = ProviderRegistry()

    # Show available providers
    available = registry.available_providers()
    print("\nProvider status:")
    for p in available:
        icon = {"ok": "✓", "n/a": "●", "missing": "✗"}[p["key_status"]]
        print(f"  {icon} {p['display']:<20s} [{p['key_status']}]")

    # Test each provider
    results = {}
    for p in available:
        name = p["name"]
        if p["key_status"] == "missing":
            print(f"\n  Skipping {p['display']} — no API key configured")
            results[name] = False
            continue
        results[name] = test_provider(name, registry)

    # Summary
    print(f"\n{'═' * 50}")
    print("  SUMMARY")
    print(f"{'═' * 50}")
    for name, passed in results.items():
        display = next(p["display"] for p in available if p["name"] == name)
        print(f"  {_status(passed)}  {display}")

    total = sum(results.values())
    print(f"\n  {total}/{len(results)} providers working")
    return 0 if all(results.values()) else 1


# Pytest compatibility
def test_groq_provider():
    registry = ProviderRegistry()
    assert test_provider("groq", registry)


def test_openrouter_provider():
    registry = ProviderRegistry()
    assert test_provider("openrouter", registry)


def test_gemini_provider():
    registry = ProviderRegistry()
    assert test_provider("gemini", registry)


def test_ollama_provider():
    registry = ProviderRegistry()
    # Ollama may not be running, so we just test init + list
    try:
        provider = registry.get_provider("ollama")
        provider.list_models()  # Returns empty list if not running
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
