"""Generate synthetic (query, relevant_chunk_id) pairs from a corpus.

For each chunk we ask an LLM to write 1+ realistic user questions that
the chunk would answer. We then keep (question, chunk_id) as the labeled
gold pair. The chunk is the only positive; every other chunk in the
corpus is treated as a negative for retrieval metrics.

Generated test sets are cached as JSON so re-running the benchmark
across multiple embedding models reuses the same questions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from simple_rag.evaluation.embeddings.corpus_loader import CorpusChunk
from simple_rag.rag.llm_providers.openrouter_provider import OpenRouterProvider

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    query: str
    relevant_chunk_id: str
    category: str
    source_text: str

    def to_dict(self) -> dict:
        return asdict(self)


_PROMPT_TEMPLATE = """Read the passage below and write exactly {n_questions} question(s) a financial analyst would ask to find this passage.

Rules:
- Output ONLY the question(s), nothing else.
- One question per line, no numbering, no bullets, no preamble.
- Each question must end with a question mark.
- Questions must be specific to the content (mention companies, funds, figures, or topics from the text).
- Do NOT copy sentences from the passage verbatim.

Context: {category} — {parent}

Passage:
{text}

Question(s):"""


_META_PATTERNS = re.compile(
    r"^(okay|alright|sure|let'?s|the user wants|i need to|i'll|first,|so,|"
    r"note:|answer:|here (is|are)|based on|looking at|this passage|"
    r"the passage|according to)",
    re.IGNORECASE,
)


def _parse_questions(raw: str, max_questions: int) -> List[str]:
    # Strip <think>...</think> blocks (Qwen3 chain-of-thought)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    questions: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\d\.\-\*\)\s]+", "", line).strip()
        line = line.strip(' "\'')
        if not line or len(line) < 8:
            continue
        if _META_PATTERNS.match(line):
            continue
        if not line.endswith("?"):
            line = line + "?"
        questions.append(line)
        if len(questions) >= max_questions:
            break
    return questions


LM_STUDIO_URL = "http://127.0.0.1:1234/v1"


def generate_test_set(
    corpus: List[CorpusChunk],
    provider: Optional[OpenRouterProvider] = None,
    questions_per_chunk: int = 1,
    max_chunks: Optional[int] = None,
    cache_path: Optional[Path] = None,
    truncate_text: int = 2000,
) -> List[QAPair]:
    """Build a synthetic test set, optionally caching to disk."""

    if cache_path and cache_path.exists():
        logger.info("Loading cached test set: %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            return [QAPair(**row) for row in json.load(f)]

    provider = provider or OpenRouterProvider(base_url=LM_STUDIO_URL)
    selected = corpus if max_chunks is None else corpus[:max_chunks]
    pairs: List[QAPair] = []

    for i, chunk in enumerate(selected, 1):
        text = chunk.text[:truncate_text]
        prompt = _PROMPT_TEMPLATE.format(
            n_questions=questions_per_chunk,
            category=chunk.category,
            parent=chunk.parent or "N/A",
            text=text,
        )
        try:
            resp = provider.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial analyst assistant. "
                            "Respond with ONLY the requested questions — "
                            "no thinking, no preamble, no explanations."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=256,
            )
        except Exception as e:
            logger.warning("LLM error on chunk %s: %s", chunk.chunk_id, e)
            continue

        questions = _parse_questions(resp.content, questions_per_chunk)
        for q in questions:
            pairs.append(
                QAPair(
                    query=q,
                    relevant_chunk_id=chunk.chunk_id,
                    category=chunk.category,
                    source_text=text,
                )
            )

        if i % 25 == 0:
            logger.info("Generated questions for %d/%d chunks", i, len(selected))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in pairs], f, indent=2, ensure_ascii=False)
        logger.info("Cached test set → %s", cache_path)

    return pairs
