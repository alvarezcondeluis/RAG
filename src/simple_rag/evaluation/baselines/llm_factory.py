"""
LLM factory helpers for baseline evaluation.

Both baselines (VectorRAGBaseline, DocumentAgent) accept any
langchain_core BaseChatModel. Use these helpers to build a configured
model and attach a DetailedTracker to inspect every call.

Quick-start:

    import os
    from simple_rag.evaluation.baselines.llm_factory import (
        create_openrouter_llm, POWERFUL_MODELS, SEARCH_MODELS, DetailedTracker
    )

    # Standard model
    llm = create_openrouter_llm(POWERFUL_MODELS["deepseek-r1"])

    # Model with web search
    llm = create_openrouter_llm(SEARCH_MODELS["perplexity-pro"])

    # Attach tracker to any run
    tracker = DetailedTracker()
    response = llm.invoke("...", config={"callbacks": [tracker]})
    tracker.print_trace()
"""
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

POWERFUL_MODELS: Dict[str, str] = {
    "claude-opus":      "anthropic/claude-opus-4",
    "claude-sonnet":    "anthropic/claude-sonnet-4-5",
    "gpt-4o":           "openai/gpt-4o",
    "gemini-pro":       "google/gemini-2.5-pro-preview",
    # Reasoning models — expose their chain of thought via reasoning_content
    "deepseek-r1":      "deepseek/deepseek-r1",
    "o3-mini":          "openai/o3-mini",
    # Free tier
    "llama-free":       "meta-llama/llama-3.3-70b-instruct:free",
}

# Models with built-in web search (no extra config needed)
SEARCH_MODELS: Dict[str, str] = {
    "perplexity-sonar":     "perplexity/sonar",
    "perplexity-pro":       "perplexity/sonar-pro",
    "perplexity-reasoning": "perplexity/sonar-reasoning-pro",
    "gpt-4o-online":        "openai/gpt-4o:online",
    "gemini-online":        "google/gemini-2.0-flash-001:online",
    "claude-online":        "anthropic/claude-sonnet-4-5:online",
}

# ---------------------------------------------------------------------------
# DetailedTracker
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """Full record of one LLM call."""
    call_index:       int
    messages:         List[Dict[str, str]]  # [{"role": ..., "content": ...}]
    response:         str
    reasoning:        Optional[str]         # DeepSeek R1 / thinking-model chain-of-thought
    latency_seconds:  float
    token_usage:      Dict[str, int]        # prompt / completion / total
    model:            Optional[str] = None


class DetailedTracker(BaseCallbackHandler):
    """
    Callback that records every LLM call: prompt, response, reasoning
    tokens (DeepSeek R1, o3-mini, Perplexity Reasoning), latency, and
    token counts.

    Attach it per-run:
        tracker = DetailedTracker()
        llm.invoke("...", config={"callbacks": [tracker]})

    Or share a single instance across a full agent session to accumulate
    all calls in order.

    Attributes:
        calls      — list of CallRecord, one per LLM call
        total_*    — aggregate counters across all calls
    """

    def __init__(self):
        super().__init__()
        self.calls: List[CallRecord] = []
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_latency_seconds: float = 0.0
        # Internal state keyed by run_id to handle concurrent calls
        self._pending: Dict[UUID, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs,
    ):
        formatted = []
        for msg in messages[0]:   # messages[0] is the current call's message list
            role = type(msg).__name__.replace("Message", "").lower()
            formatted.append({"role": role, "content": str(msg.content)})

        self._pending[run_id] = {
            "start": time.time(),
            "messages": formatted,
            "model": serialized.get("kwargs", {}).get("model_name")
                     or serialized.get("name"),
        }

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        pending = self._pending.pop(run_id, {})
        latency = time.time() - pending.get("start", time.time())

        # Extract text and reasoning from the first generation
        response_text = ""
        reasoning_text = None
        if response.generations:
            gen = response.generations[0][0]
            response_text = gen.text or ""
            # Reasoning content — present in DeepSeek R1, Perplexity Reasoning,
            # and OpenAI o-series models when reasoning is not excluded.
            extra = getattr(getattr(gen, "message", None), "additional_kwargs", {})
            reasoning_text = extra.get("reasoning_content") or extra.get("thinking")

        # Token usage — handle OpenAI and Gemini formats
        out = response.llm_output or {}
        if "token_usage" in out:
            tu = out["token_usage"]
            prompt_tok = tu.get("prompt_tokens", 0) or 0
            compl_tok  = tu.get("completion_tokens", 0) or 0
        else:
            prompt_tok = out.get("prompt_token_count", 0) or 0
            compl_tok  = out.get("candidates_token_count", 0) or 0

        self.total_prompt_tokens     += prompt_tok
        self.total_completion_tokens += compl_tok
        self.total_latency_seconds   += latency

        self.calls.append(CallRecord(
            call_index      = len(self.calls),
            messages        = pending.get("messages", []),
            response        = response_text,
            reasoning       = reasoning_text,
            latency_seconds = round(latency, 3),
            token_usage     = {
                "prompt":     prompt_tok,
                "completion": compl_tok,
                "total":      prompt_tok + compl_tok,
            },
            model = pending.get("model"),
        ))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def summary(self) -> Dict[str, Any]:
        """Aggregate statistics across all calls."""
        return {
            "total_calls":             len(self.calls),
            "total_latency_seconds":   round(self.total_latency_seconds, 3),
            "avg_latency_seconds":     round(
                self.total_latency_seconds / len(self.calls), 3
            ) if self.calls else 0,
            "token_usage": {
                "prompt":     self.total_prompt_tokens,
                "completion": self.total_completion_tokens,
                "total":      self.total_tokens,
            },
            "calls_with_reasoning": sum(
                1 for c in self.calls if c.reasoning is not None
            ),
        }

    def print_trace(self, max_content_len: int = 400):
        """
        Print a human-readable trace of all recorded calls.

        Args:
            max_content_len: Truncate prompt/response/reasoning text beyond
                             this length (set to 0 to disable truncation).
        """
        def _clip(text: str) -> str:
            if max_content_len and len(text) > max_content_len:
                return text[:max_content_len] + f"… [{len(text)} chars]"
            return text

        sep = "─" * 72
        print(f"\n{'═' * 72}")
        print(f"  LLM CALL TRACE  ({len(self.calls)} call(s))")
        print(f"{'═' * 72}")

        for rec in self.calls:
            print(f"\n{sep}")
            model_tag = f" [{rec.model}]" if rec.model else ""
            print(f"  Call #{rec.call_index + 1}{model_tag}  |  "
                  f"{rec.latency_seconds}s  |  "
                  f"{rec.token_usage['total']} tokens "
                  f"(↑{rec.token_usage['prompt']} ↓{rec.token_usage['completion']})")
            print(sep)

            print("\n  ── PROMPT ──")
            for msg in rec.messages:
                role = msg["role"].upper()
                print(f"  [{role}]\n  {_clip(msg['content'])}")

            if rec.reasoning:
                print("\n  ── REASONING / THINKING ──")
                print(f"  {_clip(rec.reasoning)}")

            print("\n  ── RESPONSE ──")
            print(f"  {_clip(rec.response)}")

        print(f"\n{sep}")
        s = self.summary()
        print(f"  TOTALS  |  {s['total_calls']} calls  |  "
              f"{s['total_latency_seconds']}s  |  "
              f"{s['token_usage']['total']} tokens  |  "
              f"{s['calls_with_reasoning']} with reasoning")
        print(f"{'═' * 72}\n")

    def reset(self):
        """Clear all recorded calls and reset counters."""
        self.calls.clear()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_seconds = 0.0
        self._pending.clear()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_openrouter_llm(
    model: str = POWERFUL_MODELS["claude-sonnet"],
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    search: bool = False,
    reasoning_effort: Optional[str] = None,
    http_referer: str = "https://github.com/sec-filings-intelligence",
    site_name: str = "SEC Filings Intelligence RAG",
) -> ChatOpenAI:
    """
    Build a ChatOpenAI instance pointed at OpenRouter.

    Args:
        model:            OpenRouter model ID. Use POWERFUL_MODELS or SEARCH_MODELS
                          presets, or any OpenRouter model string.
        api_key:          OpenRouter key. Falls back to OPEN_ROUTER_API_KEY env var.
        temperature:      Sampling temperature (0.0 = deterministic).
        max_tokens:       Maximum completion tokens.
        search:           If True, appends `:online` to the model ID to enable
                          web search (only works for models that support it —
                          use SEARCH_MODELS presets for reliable options).
        reasoning_effort: For o3-mini / o1 models: "low", "medium", or "high".
                          Ignored for other models. DeepSeek R1 returns reasoning
                          automatically — no flag needed.
        http_referer / site_name: Sent as headers for OpenRouter dashboard tracking.

    Returns:
        A ChatOpenAI instance ready to use in VectorRAGBaseline, DocumentAgent,
        or directly with a DetailedTracker.

    Examples:
        # Fastest reasoning model
        llm = create_openrouter_llm(POWERFUL_MODELS["deepseek-r1"])

        # Web-search augmented
        llm = create_openrouter_llm(SEARCH_MODELS["perplexity-pro"])

        # o3-mini with high reasoning effort
        llm = create_openrouter_llm(POWERFUL_MODELS["o3-mini"], reasoning_effort="high")

        # Claude with search (online plugin)
        llm = create_openrouter_llm(POWERFUL_MODELS["claude-sonnet"], search=True)
    """
    resolved_key = api_key or os.getenv("OPEN_ROUTER_API_KEY")
    if not resolved_key:
        raise ValueError(
            "OpenRouter API key not found. "
            "Pass api_key= or set the OPEN_ROUTER_API_KEY environment variable."
        )

    # Append :online suffix for web search if not already present
    if search and ":online" not in model and model not in SEARCH_MODELS.values():
        model = f"{model}:online"

    # Extra body parameters forwarded to OpenRouter (e.g. reasoning effort for o-series)
    extra_body: Dict[str, Any] = {}
    if reasoning_effort and any(p in model for p in ("o1", "o3", "o4")):
        extra_body["reasoning"] = {"effort": reasoning_effort}

    return ChatOpenAI(
        model=model,
        openai_api_key=resolved_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body if extra_body else None,
        default_headers={
            "HTTP-Referer": http_referer,
            "X-Title": site_name,
        },
    )


# Common local server base URLs
LOCAL_SERVERS: Dict[str, str] = {
    "lmstudio":  "http://localhost:1234/v1",
    "ollama":    "http://localhost:11434/v1",
    "llamacpp":  "http://localhost:8080/v1",
}


def create_local_llm(
    base_url: str = LOCAL_SERVERS["lmstudio"],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ChatOpenAI:
    """
    Build a ChatOpenAI pointed at a local OpenAI-compatible server.

    Works with LM Studio, Ollama (openai compat mode), llama.cpp server,
    or any other server that serves the /v1/chat/completions endpoint.

    Args:
        base_url:    Server URL. Use LOCAL_SERVERS presets or pass your own.
        model:       Model name as the server expects it. If None, uses
                     "local-model" (LM Studio ignores this field anyway).
        temperature: Sampling temperature.
        max_tokens:  Maximum tokens in the completion.

    Examples:
        # LM Studio (default port)
        llm = create_local_llm()

        # Ollama with a specific model
        llm = create_local_llm(LOCAL_SERVERS["ollama"], model="llama3.2")

        # Custom port
        llm = create_local_llm("http://localhost:5000/v1")
    """
    return ChatOpenAI(
        model=model or "local-model",
        openai_api_key="local",          # local servers don't validate the key
        openai_api_base=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def compare_models(
    question: str,
    models: Dict[str, BaseChatModel],
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    """
    Run the same question against multiple models and return a list of results
    ready for a pandas DataFrame or notebook display.

    Args:
        question:      The question to ask every model.
        models:        {"label": llm, ...} — any BaseChatModel instances.
        system_prompt: Optional shared system prompt.

    Returns:
        List of dicts, one per model:
        [
            {
                "model":            "DeepSeek R1",
                "answer":           "...",
                "reasoning":        "..." or None,
                "latency_seconds":  4.2,
                "prompt_tokens":    150,
                "completion_tokens": 300,
                "total_tokens":     450,
            },
            ...
        ]

    Example:
        import pandas as pd
        rows = compare_models(question, {"GPT-4o": llm1, "DeepSeek R1": llm2})
        df = pd.DataFrame(rows)[["model","latency_seconds","total_tokens","answer"]]
    """
    from simple_rag.evaluation.baselines.direct_llm import DirectLLMBaseline

    rows = []
    for label, llm in models.items():
        print(f"  Querying {label}…", end=" ", flush=True)
        baseline = DirectLLMBaseline(llm=llm, system_prompt=system_prompt)
        result = baseline.query(question)
        print(f"{result['latency_seconds']}s | {result['token_usage']['total']} tokens")
        rows.append({
            "model":             label,
            "answer":            result["answer"],
            "reasoning":         result["reasoning"],
            "latency_seconds":   result["latency_seconds"],
            "prompt_tokens":     result["token_usage"]["prompt"],
            "completion_tokens": result["token_usage"]["completion"],
            "total_tokens":      result["token_usage"]["total"],
        })
    return rows


def create_gemini_llm(
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Build the custom GeminiChatModel (google-genai SDK).

    Args:
        model:   Gemini model name.
        api_key: Google AI Studio key. Falls back to GOOGLE_AI_STUDIO_API_KEY.
    """
    from simple_rag.evaluation.gemini import GeminiChatModel

    resolved_key = api_key or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Gemini API key not found. "
            "Pass api_key= or set the GOOGLE_AI_STUDIO_API_KEY environment variable."
        )
    return GeminiChatModel(
        model_name=model,
        api_key=resolved_key,
        temperature=temperature,
    )
