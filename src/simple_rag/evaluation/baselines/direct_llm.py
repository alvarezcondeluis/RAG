"""
Direct LLM baseline: question → model → answer.

No retrieval, no agent loop. The model answers using only its parametric
knowledge — or live web data if using an online/search model.

Useful as the simplest possible comparison point against the RAG pipeline.

Usage:
    from simple_rag.evaluation.baselines import (
        DirectLLMBaseline, create_openrouter_llm,
        POWERFUL_MODELS, SEARCH_MODELS,
    )

    # Parametric knowledge only
    result = DirectLLMBaseline(
        create_openrouter_llm(POWERFUL_MODELS["deepseek-r1"])
    ).query("What are Apple's main business risks?")

    # With live web search
    result = DirectLLMBaseline(
        create_openrouter_llm(SEARCH_MODELS["perplexity-pro"])
    ).query("What did Apple report in their latest 10-K?")

    # result keys: answer, reasoning, latency_seconds, token_usage
"""
import time
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.simple_rag.evaluation.baselines._token_tracker import TokenTracker
from src.simple_rag.evaluation.baselines.llm_factory import DetailedTracker

_DEFAULT_SYSTEM = (
    "You are a financial analyst expert in SEC filings, mutual funds, ETFs, "
    "and public company financials. Answer questions accurately and concisely."
)


class DirectLLMBaseline:
    """
    Wraps any LangChain chat model for direct question-answering with
    full observability: answer, reasoning chain, latency, and token counts.

    Parameters
    ----------
    llm           : Any LangChain BaseChatModel. Use create_openrouter_llm()
                    to build one quickly.
    system_prompt : Custom system prompt. Pass None to use no system message.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: Optional[str] = _DEFAULT_SYSTEM,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self._tracker = DetailedTracker()

    def query(self, question: str) -> dict:
        """
        Send a question directly to the model and return the full result.

        Returns
        -------
        {
            "answer":           str,
            "reasoning":        str | None,   # thinking chain (DeepSeek R1, o3, …)
            "latency_seconds":  float,
            "token_usage":      {"prompt": int, "completion": int, "total": int},
        }
        """
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=question))

        self._tracker.reset()
        start = time.time()
        response = self.llm.invoke(
            messages,
            config={"callbacks": [self._tracker]},
        )
        latency = time.time() - start

        # Reasoning is captured by DetailedTracker; also check additional_kwargs directly
        reasoning = None
        extra = getattr(response, "additional_kwargs", {})
        reasoning = extra.get("reasoning_content") or extra.get("thinking")
        # Fall back to what the tracker captured
        if not reasoning and self._tracker.calls:
            reasoning = self._tracker.calls[-1].reasoning

        usage = (
            self._tracker.calls[-1].token_usage
            if self._tracker.calls
            else {"prompt": 0, "completion": 0, "total": 0}
        )

        return {
            "answer":          response.content,
            "reasoning":       reasoning,
            "latency_seconds": round(latency, 3),
            "token_usage":     usage,
        }

    def print_last_trace(self):
        """Pretty-print the last call's prompt, reasoning, and response."""
        self._tracker.print_trace()
