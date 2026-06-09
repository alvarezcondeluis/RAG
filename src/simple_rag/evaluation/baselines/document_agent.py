"""
Document-based ReAct agent baseline.

Each indexed document becomes a search tool. The agent decides which
document(s) to search and synthesises the final answer. Token usage is
tracked across all LLM calls (reasoning + retrieval + final answer).

The LLM is injected — use llm_factory helpers to build it:

    from simple_rag.evaluation.baselines.llm_factory import create_openrouter_llm
    llm = create_openrouter_llm("anthropic/claude-sonnet-4-5")

    agent = DocumentAgent(
        llm=llm,
        embed_model_name="nomic-ai/nomic-embed-text-v1.5",
    )
    agent.index_document("AAPL", apple_10k_text)
    agent.index_document("MSFT", msft_10k_text)

    # All documents available
    result = agent.query("Compare Apple and Microsoft R&D spending.")

    # Only a subset
    result = agent.query("What are Apple's risk factors?", selected_docs=["AAPL"])
"""
import re
import time
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.simple_rag.embeddings.embedding import EmbedData
from src.simple_rag.database.qdrant import QdrantDatabase
from src.simple_rag.retriever.retriever import Retriever
from src.simple_rag.evaluation.baselines._token_tracker import TokenTracker


_SYSTEM_PROMPT = """\
You are a financial analyst assistant. Answer the user's question by searching the available documents.

Available tools:
{tool_descriptions}

Use EXACTLY this format on every turn:

Thought: <your reasoning about what to search next>
Action: <tool name — must be one of: {tool_names}>
Action Input: <specific search query>

After each Observation, keep reasoning until you have enough information, then write:

Thought: I now know the final answer
Final Answer: <complete, well-structured answer>

Rules:
- Always write "Final Answer:" when you are done.
- Action must be exactly one of the listed tool names.
- Never make up information — only use what appears in Observations.
"""

_ACTION_RE = re.compile(r"Action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*([\s\S]+)", re.IGNORECASE)


class DocumentAgent:
    """
    ReAct agent with one search tool per indexed document.

    Uses a text-based Thought/Action/Observation loop that works with any
    chat model — no native tool-calling support required.

    Parameters
    ----------
    llm                 : Any LangChain BaseChatModel (ChatOpenAI, GeminiChatModel, …).
    embed_model_name    : HuggingFace model for chunking + retrieval.
    vector_size         : Embedding dimensionality (must match embed_model).
    top_k               : Chunks retrieved per tool call.
    chunk_size          : Characters per chunk.
    chunk_overlap       : Overlap between consecutive chunks.
    collection_prefix   : Qdrant collection name prefix (avoids collisions).
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embed_model_name: str,
        vector_size: int = 768,
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        collection_prefix: str = "doc_agent",
    ):
        self.llm = llm
        self.embed_data = EmbedData(model_name=embed_model_name)
        self.vector_size = vector_size
        self.top_k = top_k
        self.collection_prefix = collection_prefix
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._tracker = TokenTracker()
        # doc_name → (QdrantDatabase, Retriever)
        self._doc_retrievers: dict[str, tuple[QdrantDatabase, Retriever]] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_document(self, doc_name: str, text: str, reset: bool = False) -> int:
        """
        Chunk, embed, and store one document in its own Qdrant collection.

        Args:
            doc_name: Logical name used as the tool name (e.g. "AAPL", "MSFT_10K").
            text:     Full document text.
            reset:    If True, delete the existing collection first.

        Returns:
            Number of chunks indexed.
        """
        safe_name = re.sub(r"[^a-z0-9_]", "_", doc_name.lower())
        collection_name = f"{self.collection_prefix}_{safe_name}"

        vector_db = QdrantDatabase(
            collection_name=collection_name,
            vector_size=self.vector_size,
            auto_start_qdrant=True,
        )

        if reset:
            vector_db.delete_collection()

        vector_db.create_collection()

        chunks = self._splitter.split_text(text)
        embeddings = self.embed_data.embed(chunks, description=f"Indexing {doc_name}")

        embed_entries = [
            {
                "vector": emb,
                "payload": {"text": chunk, "source": doc_name, "chunk_idx": i},
            }
            for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
            if emb is not None
        ]
        vector_db.batch_upsert(embed_entries)

        retriever = Retriever(
            vector_db=vector_db,
            embed_model=self.embed_data.model,
        )
        self._doc_retrievers[doc_name] = (vector_db, retriever)
        print(f"Indexed {len(embed_entries)} chunks for '{doc_name}'.")
        return len(embed_entries)

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def _build_tool_map(self, doc_names: List[str]) -> dict:
        """Return {tool_name: callable(query) -> str} for the given docs."""
        tool_map = {}
        for name in doc_names:
            if name not in self._doc_retrievers:
                raise ValueError(
                    f"Document '{name}' is not indexed. Call index_document() first."
                )
            _, retriever = self._doc_retrievers[name]
            safe = re.sub(r"[^a-z0-9_]", "_", name.lower())
            tool_name = f"search_{safe}"

            # Capture by default arg to avoid closure-over-loop-variable bug.
            def _make_search(retr=retriever, doc=name):
                def search(query: str) -> str:
                    results = retr.search(query, limit=self.top_k)
                    passages = [r.payload.get("text", "") for r in results]
                    return "\n\n---\n\n".join(passages) or "No relevant passages found."

                search.__doc__ = f"Search the {doc} document. Input: specific search query."
                return search

            tool_map[tool_name] = _make_search()
        return tool_map

    def _react_loop(
        self,
        question: str,
        tool_map: dict,
        max_iterations: int,
    ) -> tuple[str, int]:
        """
        Run the ReAct text loop.

        Returns:
            (final_answer: str, iterations_used: int)
        """
        tool_descriptions = "\n".join(
            f"  {name}: {fn.__doc__}" for name, fn in tool_map.items()
        )
        tool_names = ", ".join(tool_map.keys())

        messages = [
            SystemMessage(
                content=_SYSTEM_PROMPT.format(
                    tool_descriptions=tool_descriptions,
                    tool_names=tool_names,
                )
            ),
            HumanMessage(content=question),
        ]

        for iteration in range(1, max_iterations + 1):
            response = self.llm.invoke(
                messages,
                config={"callbacks": [self._tracker]},
            )
            text = response.content
            messages.append(AIMessage(content=text))

            # Check for final answer first
            final_match = _FINAL_RE.search(text)
            if final_match:
                return final_match.group(1).strip(), iteration

            # Parse action + input
            action_match = _ACTION_RE.search(text)
            input_match = _INPUT_RE.search(text)

            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()

                if action in tool_map:
                    observation = tool_map[action](action_input)
                else:
                    observation = (
                        f"Unknown tool '{action}'. Available: {tool_names}"
                    )

                messages.append(HumanMessage(content=f"Observation: {observation}"))
            else:
                # Model didn't follow format — treat full output as the answer
                return text.strip(), iteration

        return "Maximum iterations reached without a final answer.", max_iterations

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        selected_docs: Optional[List[str]] = None,
        max_iterations: int = 8,
    ) -> dict:
        """
        Run the ReAct agent and return the answer with metadata.

        Args:
            question:       Natural-language question.
            selected_docs:  Documents the agent may search.
                            None → all indexed documents.
            max_iterations: Cap on reasoning iterations.

        Returns:
            {
                "answer": str,
                "latency_seconds": float,
                "llm_calls": int,
                "iterations": int,
                "token_usage": {"prompt": int, "completion": int, "total": int},
            }
        """
        doc_names = (
            selected_docs
            if selected_docs is not None
            else list(self._doc_retrievers.keys())
        )
        if not doc_names:
            raise RuntimeError("No documents indexed. Call index_document() first.")

        tool_map = self._build_tool_map(doc_names)
        self._tracker.reset()
        start = time.time()

        answer, iterations = self._react_loop(question, tool_map, max_iterations)
        latency = time.time() - start

        return {
            "answer": answer,
            "latency_seconds": round(latency, 3),
            "llm_calls": self._tracker.llm_calls,
            "iterations": iterations,
            "token_usage": self._tracker.as_dict(),
        }

    @property
    def indexed_documents(self) -> List[str]:
        """Names of all currently indexed documents."""
        return list(self._doc_retrievers.keys())
