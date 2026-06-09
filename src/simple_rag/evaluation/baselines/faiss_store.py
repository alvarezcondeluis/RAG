"""
Lightweight FAISS-backed vector store for baseline evaluation.

No server, no Docker — just in-memory C++ vectors + a JSON sidecar for
chunk metadata. Save/load is a single call so you never re-embed the same
document twice.

Usage:
    from simple_rag.evaluation.baselines.faiss_store import FaissVectorStore

    store = FaissVectorStore(vector_size=1024)
    store.add(chunks, embeddings)
    store.save("baseline/example/aapl")        # writes aapl.index + aapl.meta.json

    store = FaissVectorStore.load("baseline/example/aapl")
    results = store.search(query_embedding, k=5)
    # [{"text": ..., "score": 0.87, "chunk_idx": 3, ...}, ...]
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np


class FaissVectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP index with JSON metadata storage.

    Uses inner-product search on L2-normalised vectors which is equivalent
    to cosine similarity — the standard for sentence embeddings.

    Parameters
    ----------
    vector_size : Dimensionality of the embedding vectors (must match the
                  model used to produce them).
    """

    def __init__(self, vector_size: int):
        self.vector_size = vector_size
        self._index = faiss.IndexFlatIP(vector_size)
        self._payloads: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: List[str],
        embeddings: List[Optional[List[float]]],
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add chunks and their embeddings to the index.

        Args:
            chunks:        List of text strings.
            embeddings:    Parallel list of embedding vectors (None entries
                           are skipped so bad batches don't crash the build).
            extra_payload: Extra metadata merged into every chunk's record
                           (e.g. {"source": "AAPL_10K", "year": 2024}).

        Returns:
            Number of vectors actually added.
        """
        vecs, metas = [], []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            if emb is None:
                continue
            vecs.append(emb)
            payload = {"text": chunk, "chunk_idx": i}
            if extra_payload:
                payload.update(extra_payload)
            metas.append(payload)

        if not vecs:
            return 0

        arr = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(arr)          # cosine similarity via inner product
        self._index.add(arr)
        self._payloads.extend(metas)
        return len(vecs)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k most similar chunks.

        Args:
            query_embedding: Raw embedding vector from your embedding model.
            k:               Number of results to return.

        Returns:
            List of payload dicts ordered by similarity descending.
            Each dict has at least ``text``, ``chunk_idx``, and ``score``.
        """
        if self._index.ntotal == 0:
            return []

        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, indices = self._index.search(q, min(k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = dict(self._payloads[idx])
            record["score"] = float(score)
            results.append(record)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path_prefix: str) -> None:
        """
        Persist the index and metadata to disk.

        Writes two files:
            <path_prefix>.index      — FAISS binary index
            <path_prefix>.meta.json  — chunk payloads (JSON)

        Args:
            path_prefix: Path without extension, e.g. ``"baseline/example/aapl"``.
        """
        base = Path(path_prefix)
        base.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(base.with_suffix(".index")))
        base.with_suffix(".meta.json").write_text(
            json.dumps(self._payloads, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved {self._index.ntotal} vectors → {base}.index + .meta.json")

    @classmethod
    def load(cls, path_prefix: str) -> "FaissVectorStore":
        """
        Load a previously saved store from disk.

        Args:
            path_prefix: Same prefix passed to :meth:`save`.

        Returns:
            A fully initialised :class:`FaissVectorStore`.
        """
        base = Path(path_prefix)
        index = faiss.read_index(str(base.with_suffix(".index")))
        payloads = json.loads(base.with_suffix(".meta.json").read_text(encoding="utf-8"))

        store = cls.__new__(cls)
        store.vector_size = index.d
        store._index = index
        store._payloads = payloads
        print(f"Loaded {index.ntotal} vectors from {base}.index")
        return store

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of indexed vectors."""
        return self._index.ntotal
