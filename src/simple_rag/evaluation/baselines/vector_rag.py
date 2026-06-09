"""Simple vector RAG baseline: chunk → embed → Qdrant → retrieve → LLM answer."""
import time
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models.chat_models import BaseChatModel

from src.simple_rag.embeddings.embedding import EmbedData
from src.simple_rag.database.qdrant import QdrantDatabase
from src.simple_rag.retriever.retriever import Retriever
from src.simple_rag.evaluation.baselines._token_tracker import TokenTracker


class VectorRAGBaseline:
    """
    Baseline that answers questions via pure vector retrieval over chunked documents.

    The LLM is injected — use llm_factory helpers to build it:

        from simple_rag.evaluation.baselines.llm_factory import create_openrouter_llm
        llm = create_openrouter_llm("anthropic/claude-sonnet-4-5")

        baseline = VectorRAGBaseline(
            llm=llm,
            embed_model_name="nomic-ai/nomic-embed-text-v1.5",
            collection_name="baseline_vector_rag",
        )
        baseline.index_documents([{"text": full_text, "source": "AAPL_10K"}])
        result = baseline.query("What are Apple's main risk factors?")
    """

    ANSWER_PROMPT = (
        "You are a financial analyst assistant. Answer the question using ONLY the context below. "
        "If the context does not contain enough information, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        llm: BaseChatModel,
        embed_model_name: str,
        collection_name: str,
        vector_size: int = 768,
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        """
        Args:
            llm:               Any LangChain BaseChatModel (ChatOpenAI, GeminiChatModel, …).
            embed_model_name:  HuggingFace embedding model (must match vector_size).
            collection_name:   Qdrant collection name for this baseline run.
            vector_size:       Embedding dimensionality.
            top_k:             Chunks retrieved per query.
            chunk_size:        Characters per chunk.
            chunk_overlap:     Overlap between consecutive chunks.
        """
        self.llm = llm
        self.embed_data = EmbedData(model_name=embed_model_name)
        self.vector_db = QdrantDatabase(
            collection_name=collection_name,
            vector_size=vector_size,
            auto_start_qdrant=True,
        )
        self.retriever = Retriever(
            vector_db=self.vector_db,
            embed_model=self.embed_data.model,
        )
        self.top_k = top_k
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._tracker = TokenTracker()
        self._indexed = False

    def index_documents(self, documents: List[dict], reset: bool = False) -> int:
        """
        Chunk and index documents into Qdrant.

        Args:
            documents: list of {"text": str, "source": str, **extra_metadata}
            reset:     if True, delete existing collection before indexing

        Returns:
            Number of chunks indexed.
        """
        if reset:
            self.vector_db.delete_collection()

        self.vector_db.create_collection()

        all_entries = []
        for doc in documents:
            chunks = self._splitter.split_text(doc["text"])
            source = doc.get("source", "unknown")
            extra = {k: v for k, v in doc.items() if k not in ("text", "source")}
            for i, chunk in enumerate(chunks):
                all_entries.append({"text": chunk, "source": source, "chunk_idx": i, **extra})

        texts = [e["text"] for e in all_entries]
        embeddings = self.embed_data.embed(texts, description="Indexing chunks")

        embed_entries = [
            {"vector": emb, "payload": entry}
            for emb, entry in zip(embeddings, all_entries)
            if emb is not None
        ]
        self.vector_db.batch_upsert(embed_entries)
        self._indexed = True
        print(f"Indexed {len(embed_entries)} chunks from {len(documents)} document(s).")
        return len(embed_entries)

    def query(self, question: str) -> dict:
        """
        Retrieve relevant chunks and generate an answer.

        Returns:
            {
                "answer": str,
                "latency_seconds": float,
                "retrieved_chunks": int,
                "sources": List[str],
                "token_usage": {"prompt": int, "completion": int, "total": int},
            }
        """
        if not self._indexed:
            raise RuntimeError("Call index_documents() before querying.")

        self._tracker.reset()
        start = time.time()

        results = self.retriever.search(question, limit=self.top_k)
        context = "\n\n---\n\n".join(r.payload.get("text", "") for r in results)
        sources = list({r.payload.get("source", "unknown") for r in results})

        prompt = self.ANSWER_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(
            prompt,
            config={"callbacks": [self._tracker]},
        )

        latency = time.time() - start

        return {
            "answer": response.content,
            "latency_seconds": round(latency, 3),
            "retrieved_chunks": len(results),
            "sources": sources,
            "token_usage": self._tracker.as_dict(),
        }
