"""
Semantic Search Benchmark

Compares two retrieval modes for semantic (vector) queries:
  1. Oracle  — directly embeds the question and queries the vector index (ground truth)
  2. Pipeline — full text2cypher pipeline generates the Cypher; checks if it uses vector search

Saves reports to: src/simple_rag/evaluation/reports/semantic/
"""

import sys
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Allow running as a script
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simple_rag.database.neo4j.neo4j import Neo4jDatabase
from simple_rag.rag.query_handler import QueryHandler


# ── Test set ──────────────────────────────────────────────────────────────────

TEST_QUESTIONS: List[Dict[str, Any]] = [
    # ── RiskFactor ────────────────────────────────────────────────────────────
    {
        "id": 1,
        "question": "What does Apple say about artificial intelligence and machine learning risks in their 10-K?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "RiskFactor",
    },
    {
        "id": 2,
        "question": "How does Microsoft describe cybersecurity and data privacy risks in their annual filing?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "RiskFactor",
    },
    {
        "id": 3,
        "question": "What regulatory and antitrust risks does Google disclose in its 10-K?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "RiskFactor",
    },
    {
        "id": 4,
        "question": "Which companies mention supply chain concentration or manufacturing risks in their filings?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "RiskFactor",
    },
    # ── BusinessInformation ───────────────────────────────────────────────────
    {
        "id": 5,
        "question": "How does Microsoft describe its cloud computing and Azure business segments?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "BusinessInformation",
    },
    {
        "id": 6,
        "question": "What does Tesla say about its electric vehicle manufacturing and energy business?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "BusinessInformation",
    },
    {
        "id": 7,
        "question": "How do companies describe their advertising revenue model in their 10-K filings?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "BusinessInformation",
    },
    # ── ManagementDiscussion ──────────────────────────────────────────────────
    {
        "id": 8,
        "question": "What does Apple management say about revenue growth drivers and gross margin trends?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "ManagementDiscussion",
    },
    {
        "id": 9,
        "question": "How does Amazon management explain operating income and segment profitability?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "ManagementDiscussion",
    },
    # ── LegalProceeding ───────────────────────────────────────────────────────
    {
        "id": 10,
        "question": "What antitrust or competition lawsuits are disclosed in Google's 10-K filings?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "LegalProceeding",
    },
    {
        "id": 11,
        "question": "What patent disputes or intellectual property litigation does Apple disclose?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "LegalProceeding",
    },
    # ── Properties ────────────────────────────────────────────────────────────
    {
        "id": 12,
        "question": "What data centers and office facilities does Microsoft own or lease?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "Properties",
    },
    {
        "id": 13,
        "question": "What fulfillment centers and warehouses does Amazon disclose in its properties section?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "Properties",
    },
    {
        "id": 14,
        "question": "What manufacturing plants and Gigafactories does Tesla describe in its 10-K?",
        "category": "company_filing",
        "index": "chunks",
        "section_label": "Properties",
    },
]


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class SemanticRecord:
    """A single retrieved chunk or objective."""
    text: str
    score: float
    source: str  # fund ticker or company ticker
    node_id: str  # unique key for overlap calculation (first 80 chars of text)


@dataclass
class SemanticResult:
    """Result for one question."""
    question_id: int
    question: str
    category: str
    index_used: str            # "chunks" or "objectives"
    section_label: str

    # Oracle results
    oracle_records: List[SemanticRecord] = field(default_factory=list)
    oracle_time_ms: float = 0.0
    oracle_error: Optional[str] = None

    # Pipeline results
    pipeline_cypher: str = ""
    pipeline_records: List[SemanticRecord] = field(default_factory=list)
    pipeline_time_ms: float = 0.0
    pipeline_used_vector: bool = False
    pipeline_error: Optional[str] = None

    # Comparison
    chunk_overlap: float = 0.0   # Jaccard on node_id sets
    pipeline_success: bool = False


# ── Oracle: direct vector search ──────────────────────────────────────────────

class SemanticSearchOracle:
    """
    Embeds a question and calls the Neo4j vector index directly.
    No LLM involved — this is the ground-truth retriever.
    """

    CHUNK_INDEX = "chunkEmbeddingIndex"
    OBJECTIVE_INDEX = "profileObjectiveIndex"

    # Cypher: retrieve chunks from company 10-K filings
    COMPANY_CHUNK_CYPHER = """
        CALL db.index.vector.queryNodes($index, $k, $queryVector)
        YIELD node AS chunk, score
        MATCH (chunk)<-[:HAS_CHUNK]-(s:Section)<-[:HAS_SECTION]-(filing:Filing10K)<-[:REPORTS_IN]-(c:Company)
        WHERE $section_label IN labels(s)
        RETURN chunk.text AS text, score, c.ticker AS source
        ORDER BY score DESC
        LIMIT $k
    """

    def __init__(self, neo4j_driver, embedder_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.driver = neo4j_driver
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print(f"Loading oracle embedder: {embedder_model}")
        self._embedder = HuggingFaceEmbedding(
            model_name=embedder_model,
            trust_remote_code=True,
            cache_folder="./cache",
        )
        print("✓ Oracle embedder ready")

    def _embed(self, question: str) -> List[float]:
        return self._embedder.get_query_embedding(question)

    def search_chunks(
        self,
        question: str,
        section_label: str,
        k: int = 10,
    ) -> List[SemanticRecord]:
        """Query chunkEmbeddingIndex and traverse back to the Company source."""
        vector = self._embed(question)
        with self.driver.session() as session:
            rows = list(session.run(
                self.COMPANY_CHUNK_CYPHER,
                index=self.CHUNK_INDEX,
                k=k,
                queryVector=vector,
                section_label=section_label,
            ))
        return [
            SemanticRecord(
                text=r["text"] or "",
                score=float(r["score"]),
                source=r["source"] or "",
                node_id=(r["text"] or ""),
            )
            for r in rows
        ]


# ── Benchmark ─────────────────────────────────────────────────────────────────

class SemanticBenchmark:
    """
    Runs the 10 semantic test questions through:
      1. Oracle  — direct vector search (ground truth, k=10)
      2. Pipeline — text2cypher generates Cypher, executed with the query embedding
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder",
        backend: str = "openai",
        openai_compatible_host: str = "localhost",
        openai_compatible_port: int = 1234,
        k: int = 10,
        embedder_model: str = "nomic-ai/nomic-embed-text-v1.5",
    ):
        self.k = k
        self.neo = Neo4jDatabase()

        print("\n=== Initializing SemanticBenchmark ===")

        # Shared embedder — passed to both oracle and handler to avoid loading twice
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print(f"Loading shared embedder: {embedder_model}")
        shared_embedder = HuggingFaceEmbedding(
            model_name=embedder_model,
            trust_remote_code=True,
            cache_folder="./cache",
        )
        print("✓ Shared embedder ready")

        self.oracle = SemanticSearchOracle.__new__(SemanticSearchOracle)
        self.oracle.driver = self.neo.driver
        self.oracle._embedder = shared_embedder

        extra_kwargs: Dict[str, Any] = {}
        if backend == "openai":
            extra_kwargs["openai_compatible_host"] = openai_compatible_host
            extra_kwargs["openai_compatible_port"] = openai_compatible_port

        self.handler = QueryHandler(
            neo4j_driver=self.neo.driver,
            cypher_model=model_name,
            cypher_backend=backend,
            use_entity_resolver=True,
            embedder=shared_embedder,
            embedder_model=embedder_model,
            enable_query_embedding=True,
            default_k=k,
            **extra_kwargs,
        )

        self.model_name = self.handler.translator.model_name
        self.results: List[SemanticResult] = []

    # ── Overlap metric ────────────────────────────────────────────────────────

    @staticmethod
    def _jaccard(a: List[SemanticRecord], b: List[SemanticRecord]) -> float:
        set_a = {r.node_id for r in a if r.node_id}
        set_b = {r.node_id for r in b if r.node_id}
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    # ── Evaluate one question ─────────────────────────────────────────────────

    def evaluate_question(self, item: Dict[str, Any]) -> SemanticResult:
        question = item["question"]
        category = item["category"]
        index_used = item["index"]
        section_label = item["section_label"]

        result = SemanticResult(
            question_id=item["id"],
            question=question,
            category=category,
            index_used=index_used,
            section_label=section_label,
        )

        print(f"\n{'='*80}")
        print(f"Q{item['id']}: {question}")
        print(f"  Category: {category} | Index: {index_used} | Section: {section_label}")
        print(f"{'='*80}")

        # ── 1. Oracle ────────────────────────────────────────────────────────
        try:
            t0 = time.time()
            result.oracle_records = self.oracle.search_chunks(
                question, section_label, k=self.k
            )
            result.oracle_time_ms = (time.time() - t0) * 1000

            print(f"\n[ORACLE] {len(result.oracle_records)} records in {result.oracle_time_ms:.0f}ms")
            for i, r in enumerate(result.oracle_records[:3], 1):
                print(f"  [{i}] score={r.score:.4f} source={r.source} | {r.text}...")

        except Exception as e:
            result.oracle_error = str(e)
            print(f"[ORACLE] ERROR: {e}")

        # ── 2. Pipeline ──────────────────────────────────────────────────────
        try:
            t0 = time.time()
            handle_result = self.handler.handle(
                question,
                execute=True,
                use_schema_injection=True,
            )
            result.pipeline_time_ms = (time.time() - t0) * 1000
            result.pipeline_cypher = handle_result.cypher or ""

            # Check if vector search was actually used
            result.pipeline_used_vector = (
                "db.index.vector.queryNodes" in result.pipeline_cypher
            )

            print(f"\n[PIPELINE] Generated Cypher:")
            print(f"  {result.pipeline_cypher}")
            print(f"  Used vector search: {result.pipeline_used_vector}")
            print(f"  Time: {result.pipeline_time_ms:.0f}ms")

            if handle_result.error:
                result.pipeline_error = handle_result.error
                print(f"  ERROR: {handle_result.error}")
            elif handle_result.data:
                # Convert raw Neo4j records into SemanticRecord objects
                for row in handle_result.data:
                    text = (
                        row.get("text") or row.get("chunk.text") or
                        row.get("obj.text") or row.get("objective") or ""
                    )
                    score = float(row.get("score", 0.0))
                    source = (
                        row.get("source") or row.get("ticker") or
                        row.get("f.ticker") or row.get("c.ticker") or ""
                    )
                    result.pipeline_records.append(SemanticRecord(
                        text=text,
                        score=score,
                        source=source,
                        node_id=text,
                    ))

                print(f"  Records: {len(result.pipeline_records)}")
                for i, r in enumerate(result.pipeline_records[:3], 1):
                    print(f"  [{i}] score={r.score:.4f} source={r.source} | {r.text}...")
            else:
                print(f"  No data returned")

        except Exception as e:
            result.pipeline_error = str(e)
            print(f"[PIPELINE] ERROR: {e}")

        # ── 3. Compute overlap ───────────────────────────────────────────────
        if result.oracle_records and result.pipeline_records:
            result.chunk_overlap = self._jaccard(result.oracle_records, result.pipeline_records)

        result.pipeline_success = (
            result.pipeline_used_vector
            and len(result.pipeline_records) > 0
            and result.pipeline_error is None
        )

        print(f"\n  Chunk overlap (Jaccard): {result.chunk_overlap:.2%}")
        print(f"  Pipeline success: {'✅' if result.pipeline_success else '❌'}")

        return result

    # ── Run all ───────────────────────────────────────────────────────────────

    def run(self):
        import io

        # Report file path
        reports_dir = SCRIPT_DIR / "reports" / "semantic"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = self.model_name.replace("/", "_").replace(":", "-")
        report_path = reports_dir / f"semantic_{timestamp}_{model_slug}.txt"

        # Buffer captures everything printed during the run so we can save it
        # to a file WITHOUT touching sys.stdout — this keeps Jupyter output working.
        buf = io.StringIO()
        _orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else print

        def _buffered_print(*args, **kwargs):
            # Print normally (shows in notebook cell)
            _orig_print(*args, **kwargs)
            # Also write to buffer (saved to file later)
            kwargs.pop("file", None)
            _orig_print(*args, file=buf, **kwargs)

        import builtins
        builtins.print = _buffered_print

        try:
            print(f"\n{'='*80}")
            print(f"SEMANTIC SEARCH BENCHMARK")
            print(f"{'='*80}")
            print(f"Model   : {self.model_name}")
            print(f"k       : {self.k} (chunks retrieved per query)")
            print(f"Report  : {report_path}")
            print(f"{'='*80}")

            for item in TEST_QUESTIONS:
                res = self.evaluate_question(item)
                self.results.append(res)

            self._print_report()

        finally:
            builtins.print = _orig_print

        # Write buffer to file
        report_path.write_text(buf.getvalue(), encoding="utf-8")
        print(f"\n✅ Report saved → {report_path}")
        self.neo.close()

    # ── Report ────────────────────────────────────────────────────────────────

    def _print_report(self):
        total = len(self.results)
        vector_used = sum(1 for r in self.results if r.pipeline_used_vector)
        successful = sum(1 for r in self.results if r.pipeline_success)
        overlaps = [r.chunk_overlap for r in self.results if r.pipeline_records and r.oracle_records]
        avg_overlap = statistics.mean(overlaps) if overlaps else 0.0

        oracle_times = [r.oracle_time_ms for r in self.results if r.oracle_time_ms > 0]
        pipeline_times = [r.pipeline_time_ms for r in self.results if r.pipeline_time_ms > 0]
        avg_oracle_ms = statistics.mean(oracle_times) if oracle_times else 0.0
        avg_pipeline_ms = statistics.mean(pipeline_times) if pipeline_times else 0.0

        print(f"\n\n{'='*80}")
        print(f"SEMANTIC SEARCH BENCHMARK — SUMMARY REPORT")
        print(f"{'='*80}")
        print(f"{'Total questions':<35} {total}")
        print(f"{'Pipeline used vector search':<35} {vector_used}/{total}")
        print(f"{'Pipeline fully successful':<35} {successful}/{total}")
        print(f"{'Avg chunk overlap (Jaccard)':<35} {avg_overlap:.2%}")
        print(f"{'Avg oracle time (ms)':<35} {avg_oracle_ms:.0f}")
        print(f"{'Avg pipeline time (ms)':<35} {avg_pipeline_ms:.0f}")
        if avg_oracle_ms > 0:
            print(f"{'Pipeline / Oracle time ratio':<35} {avg_pipeline_ms / avg_oracle_ms:.1f}x")

        print(f"\n{'─'*80}")
        print(f"PER-QUESTION RESULTS")
        print(f"{'─'*80}")
        print(f"{'#':<4} {'OK':<4} {'Vec':<4} {'Overlap':<10} {'OracleMs':<12} {'PipeMs':<12} {'Question'}")
        print(f"{'─'*80}")
        for r in self.results:
            ok = "✅" if r.pipeline_success else "❌"
            vec = "✅" if r.pipeline_used_vector else "❌"
            overlap = f"{r.chunk_overlap:.0%}" if r.oracle_records and r.pipeline_records else "—"
            print(
                f"{r.question_id:<4} {ok:<4} {vec:<4} {overlap:<10} "
                f"{r.oracle_time_ms:<12.0f} {r.pipeline_time_ms:<12.0f} {r.question[:55]}"
            )

        print(f"\n{'─'*80}")
        print(f"FAILURES / ISSUES")
        print(f"{'─'*80}")
        any_issue = False
        for r in self.results:
            issues = []
            if r.oracle_error:
                issues.append(f"Oracle error: {r.oracle_error}")
            if r.pipeline_error:
                issues.append(f"Pipeline error: {r.pipeline_error}")
            if not r.pipeline_used_vector:
                issues.append(f"Pipeline did NOT use vector search — generated: {r.pipeline_cypher[:120]}")
            elif r.chunk_overlap < 0.1 and r.pipeline_records:
                issues.append(f"Low overlap ({r.chunk_overlap:.0%}) — pipeline may have used wrong index/section")
            if issues:
                any_issue = True
                print(f"\nQ{r.question_id}: {r.question}")
                for issue in issues:
                    print(f"  ⚠️  {issue}")
        if not any_issue:
            print("  No issues detected.")

        print(f"\n{'='*80}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search Benchmark")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["ollama", "groq", "openai"],
                        help="LLM backend for the pipeline (default: openai / LM Studio)")
    parser.add_argument("--model", type=str, default="qwen2.5-coder",
                        help="Model name/ID")
    parser.add_argument("--openai-host", type=str, default="localhost")
    parser.add_argument("--openai-port", type=int, default=1234)
    parser.add_argument("--k", type=int, default=10,
                        help="Number of chunks to retrieve per query (default: 10)")
    parser.add_argument("--embedder", type=str, default="nomic-ai/nomic-embed-text-v1.5",
                        help="Embedding model for both oracle and pipeline")
    args = parser.parse_args()

    bench = SemanticBenchmark(
        model_name=args.model,
        backend=args.provider,
        openai_compatible_host=args.openai_host,
        openai_compatible_port=args.openai_port,
        k=args.k,
        embedder_model=args.embedder,
    )
    try:
        bench.run()
    except KeyboardInterrupt:
        print("\n⏸️  Benchmark interrupted.")
        bench.neo.close()


if __name__ == "__main__":
    main()
