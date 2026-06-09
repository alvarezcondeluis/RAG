"""Shared token usage callback — handles both OpenAI and Gemini llm_output formats."""
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenTracker(BaseCallbackHandler):
    """
    Accumulates token counts across all LLM calls in a baseline run.

    Supports:
    - ChatOpenAI / OpenRouter: llm_output['token_usage'] with prompt_tokens /
      completion_tokens keys
    - GeminiChatModel: llm_output with prompt_token_count / candidates_token_count keys
    """

    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.llm_calls: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs):
        out = response.llm_output or {}

        if "token_usage" in out:
            # OpenAI / OpenRouter format
            tu = out["token_usage"]
            self.prompt_tokens += tu.get("prompt_tokens", 0) or 0
            self.completion_tokens += tu.get("completion_tokens", 0) or 0
        else:
            # Gemini format
            self.prompt_tokens += out.get("prompt_token_count", 0) or 0
            self.completion_tokens += out.get("candidates_token_count", 0) or 0

        self.llm_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
            "total": self.total_tokens,
        }

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
