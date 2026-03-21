"""
Answer generation — transforms raw Neo4j results into natural language answers.
"""

from simple_rag.rag.answer_generation.result_classifier import ResultType, ResultClassifier
from simple_rag.rag.answer_generation.prompt_templates import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
)

__all__ = [
    "ResultType",
    "ResultClassifier",
    "ANSWER_SYSTEM_PROMPT",
    "build_answer_prompt",
]
