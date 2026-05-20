#!/usr/bin/env python3
"""
Query Classifier Benchmark: SetFit vs LLM-based Classification

Compares two approaches for query classification:
1. SetFit (existing, lightweight, fast)
2. LLM-based (using OpenRouter or local LM Studio)

Measures accuracy, precision, recall, F1, and response time.
"""

import sys
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import Counter


SCRIPT_DIR = Path(__file__).parent
SRC_ROOT = SCRIPT_DIR.parent.parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from sklearn.metrics import hamming_loss, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass
class ClassificationResult:
    """Stores metrics for a single query classification."""
    query_id: int
    query: str
    expected_labels: List[str]
    predicted_labels: List[str]
    confidence_score: float
    time_ms: float
    error: Optional[str] = None


@dataclass
class BenchmarkStats:
    """Aggregated statistics for a classifier."""
    classifier_name: str
    total_queries: int = 0
    correct_exact_match: int = 0
    hamming_loss: float = 0.0
    avg_time_ms: float = 0.0
    median_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    per_label_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    failed_queries: List[int] = field(default_factory=list)
    error_count: int = 0


class SetFitClassifier:
    """Direct SetFit model classifier."""

    LABELS = ["not_related", "fund_basic", "fund_portfolio", "fund_profile", "company_filing", "company_people"]
    LABEL_THRESHOLDS = {
        "not_related": 0.5,
        "fund_basic": 0.5,
        "fund_portfolio": 0.3,
        "fund_profile": 0.5,
        "company_filing": 0.5,
        "company_people": 0.5,
    }

    def __init__(self, model_path: Optional[str] = None):
        from setfit import SetFitModel

        # Default to project's trained model if not specified
        if model_path is None:
            model_path = PROJECT_ROOT / "src" / "simple_rag" / "rag" / "query" / "models" / "query_classifier"
        else:
            model_path = Path(model_path)

        print(f"Loading SetFit model from: {model_path}")
        self.model = SetFitModel.from_pretrained(str(model_path), device="cpu", local_files_only=True)
        self.name = "SetFit"

    def classify(self, query: str) -> Tuple[List[str], float]:
        """
        Classify a query.

        Returns:
            (labels, confidence_score)
        """
        probs = self.model.predict_proba([query])[0]
        if hasattr(probs, "numpy"):
            probs = probs.numpy()

        label_probs = {label: float(probs[i]) for i, label in enumerate(self.LABELS)}
        active_labels = [
            label for label, p in label_probs.items()
            if p >= self.LABEL_THRESHOLDS.get(label, 0.5)
        ]

        # Fallback: if nothing crosses threshold, take highest
        if not active_labels:
            top_label = max(label_probs, key=label_probs.get)
            active_labels = [top_label]

        active_labels.sort(key=lambda l: label_probs[l], reverse=True)
        confidence = label_probs[active_labels[0]]

        return active_labels, confidence


class LLMQueryClassifier:
    """LLM-based query classifier using OpenRouter or local endpoint."""

    CLASSIFICATION_PROMPT = """Classify the query for a financial SEC filings RAG system into one or more categories.

Categories:
- fund_basic: Fund identifiers (ticker, name, exchange, share class), provider/trust structure, charts/tables/images stored for a fund, source documents/accession numbers of fund data, and quantitative metrics (expense ratio, returns, NAV, net assets, turnover, financial highlights). Any fund property NOT about holdings composition.
- fund_portfolio: Holdings, number of holdings, sector/regional allocations, holding weights/market values, which funds hold a company. What is INSIDE a fund's portfolio.
- fund_profile: Fund strategy, risks, objectives, performance commentary, ESG/thematic characteristics from prospectus documents. Covers both semantic queries ("which funds minimize volatility") and direct document retrieval ("what does the risk section say").
- company_filing: Company 10-K content — business overview, risk factors, MD&A, legal proceedings, properties, financial statements (income, balance sheet, cash flow, revenue segments).
- company_people: Any person — fund managers (who manages a fund, how many funds per manager), company CEOs/executives (compensation, actually paid, shareholder return), insider transactions (buy/sell/grant, shares, price).
- not_related: Unrelated to SEC filings or fund data (weather, sports, general market concepts with no specific filing/fund).

Rules: queries can have MULTIPLE categories. Respond ONLY with JSON: {{"categories": [...], "confidence": 0.0-1.0}}

Key disambiguations:
- "how many holdings" → fund_portfolio | "holdings TABLE" (document artifact) → fund_basic
- Charts/tables/images of a fund → fund_basic | CIK for company ticker → company_filing | CIK for fund ticker → fund_basic
- Financial metrics on fund ticker (return %, NAV, net income ratio) → fund_basic, NOT company_filing
- Fund manager → company_people | Fund risk/strategy → fund_profile | Company risk → company_filing
- Open-ended "general info about fund X" → fund_basic + fund_portfolio + fund_profile (all three)

Examples:
- "Expense ratio of VTI?" → {{"categories": ["fund_basic"], "confidence": 0.95}}
- "Charts available for VTI?" → {{"categories": ["fund_basic"], "confidence": 0.93}}
- "Net income ratio for VTI in 2022?" → {{"categories": ["fund_basic"], "confidence": 0.95}}
- "CIK number for Apple?" → {{"categories": ["company_filing"], "confidence": 0.94}}
- "Top 10 holdings of VGT?" → {{"categories": ["fund_portfolio"], "confidence": 0.95}}
- "Which funds use a passive indexing approach?" → {{"categories": ["fund_profile"], "confidence": 0.91}}
- "Risk section for EDV?" → {{"categories": ["fund_profile"], "confidence": 0.93}}
- "Microsoft's 10-K business overview?" → {{"categories": ["company_filing"], "confidence": 0.95}}
- "Who manages Vanguard Small Cap Growth?" → {{"categories": ["company_people"], "confidence": 0.94}}
- "Insider buys for AAPL?" → {{"categories": ["company_people"], "confidence": 0.95}}
- "What is the weather today?" → {{"categories": ["not_related"], "confidence": 0.98}}
- "VBR expense ratio, top 5 holdings, portfolio manager?" → {{"categories": ["fund_basic", "fund_portfolio", "company_people"], "confidence": 0.90}}
- "General information about Vanguard Healthcare fund?" → {{"categories": ["fund_basic", "fund_portfolio", "fund_profile"], "confidence": 0.88}}
- "ESG funds from Vanguard with >10% in Technology?" → {{"categories": ["fund_profile", "fund_portfolio"], "confidence": 0.88}}

Query: {query}

Respond with ONLY the JSON object:"""

    def __init__(
        self,
        provider_type: str = "openrouter",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        local_host: str = "localhost",
        local_port: int = 1234,
    ):
        """
        Initialize LLM classifier.

        Args:
            provider_type: 'openrouter' or 'local'
            model_id: Model ID for OpenRouter (e.g., 'meta-llama/llama-3.3-70b-instruct:free')
            api_key: OpenRouter API key (if None, loads from env)
            local_host: Host for local LM Studio (default: localhost)
            local_port: Port for local LM Studio (default: 1234)
        """
        self.provider_type = provider_type
        self.name = f"LLM ({provider_type})"

        if provider_type == "openrouter":
            from simple_rag.rag.llm_providers.openrouter_provider import OpenRouterProvider
            self.provider = OpenRouterProvider(api_key=api_key, model_id=model_id or "meta-llama/llama-3.3-70b-instruct:free")
            self.name = f"LLM (OpenRouter - {self.provider.model_id})"
        elif provider_type == "local":
            from simple_rag.rag.text2cypher import CypherTranslator
            # Use local LM Studio via OpenAI-compatible endpoint
            import os
            # Create a minimal provider for local LM Studio
            from openai import OpenAI
            self.client = OpenAI(
                base_url=f"http://{local_host}:{local_port}/v1",
                api_key="not-needed"
            )
            self.provider = None
            self.name = f"LLM (Local LM Studio)"
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def classify(self, query: str) -> Tuple[List[str], float]:
        """
        Classify a query using LLM.

        Returns:
            (labels, confidence_score) or raises exception on error
        """
        prompt = self.CLASSIFICATION_PROMPT.format(query=query)

        try:
            if self.provider_type == "openrouter":
                response = self.provider.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                )
                content = response.content
            else:
                try:
                    response = self.client.chat.completions.create(
                        model="local-model",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=500,
                    )
                    content = response.choices[0].message.content
                except Exception as connection_error:
                    raise ConnectionError(
                        f"Failed to connect to local LM Studio at {self.client.base_url}\n"
                        f"Error: {connection_error}\n"
                        f"Make sure LM Studio is running on localhost:1234"
                    )

            # Parse JSON response
            import json
            # Try to extract JSON from potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            labels = result.get("categories", [])
            confidence = result.get("confidence", 0.5)

            return labels, float(confidence)
        except ConnectionError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")


class QueryClassifierBenchmark:
    """Benchmark suite for comparing query classifiers."""

    LABELS = ["not_related", "fund_basic", "fund_portfolio", "fund_profile", "company_filing", "company_people"]

    def __init__(self, test_set_path: str):
        self.test_set_path = Path(test_set_path)
        self.test_data: List[Dict[str, Any]] = []
        self.setfit_results: List[ClassificationResult] = []
        self.llm_results: List[ClassificationResult] = []
        self.mlb = MultiLabelBinarizer(classes=self.LABELS)
        self.mlb.fit([self.LABELS])

    @staticmethod
    def _get_labels(item: Dict[str, Any]) -> List[str]:
        """
        Read expected labels from a data item regardless of key name.
        Handles 'labels' (list), 'label' (string or list), and missing key.
        After the training_data.json normalisation pass all items should
        have 'labels' (list), but this stays defensive.
        """
        raw = item.get("labels") or item.get("label", [])
        if isinstance(raw, str):
            return [raw]
        return list(raw)

    @staticmethod
    def _infer_labels_from_cypher(cypher: str) -> List[str]:
        """
        Infer classification labels from expected Cypher query patterns.
        Patterns are derived directly from schema_slices.py — one block per slice.
        """
        labels: set = set()

        # ── company_people ──────────────────────────────────────────────────────
        # Schema nodes/rels: MANAGED_BY, HAS_CEO, InsiderTransaction,
        #   CompensationPackage, RECEIVED_COMPENSATION, AWARDED_BY, MADE_BY
        # Properties unique to this slice: ceoCompensation, ceoActuallyPaid,
        #   shareholderReturn, remainingShares, transactionType
        # Index: personNameIndex
        if any(p in cypher for p in [
            "MANAGED_BY", "HAS_CEO",
            "InsiderTransaction", "CompensationPackage",
            "RECEIVED_COMPENSATION", "AWARDED_BY", "MADE_BY",
            "ceoCompensation", "ceoActuallyPaid",
            "shareholderReturn", "remainingShares", "transactionType",
            "personNameIndex",
        ]):
            labels.add("company_people")

        # ── company_filing ──────────────────────────────────────────────────────
        # Schema nodes/rels: Filing10K, REPORTS_IN, HAS_FINANCIALS,
        #   FinancialMetric, Segment
        # Section labels (only in 10-K): BusinessInformation, LegalProceeding,
        #   ManagementDiscussion, Properties, Financials
        # Vector index: chunkEmbeddingIndex
        # Properties unique to this slice: incomeStatement, balanceSheet,
        #   cashFlow, fiscalYear
        # chunkEmbeddingIndex is SHARED between company_filing (10-K chunks) and
        # fund_profile (Strategy/RiskFactor chunks inside :Profile).
        # Only count it as company_filing when the traversal goes through Filing10K
        # or a 10-K-specific section — NOT when it goes through :Profile / DEFINED_BY.
        _is_profile_chunk = any(p in cypher for p in [":Profile", "DEFINED_BY", ":Strategy", ":Objective", "profileObjectiveIndex"])
        _chunk_index_present = "chunkEmbeddingIndex" in cypher
        if any(p in cypher for p in [
            "Filing10K", "REPORTS_IN", "HAS_FINANCIALS", "HAS_FINACIALS",
            "FinancialMetric", ":Segment",
            "BusinessInformation", "LegalProceeding", "ManagementDiscussion",
            ":Properties", ":Financials",
            "incomeStatement", "balanceSheet", "cashFlow", "fiscalYear",
        ]) or (_chunk_index_present and not _is_profile_chunk):
            labels.add("company_filing")

        # ── fund_portfolio ──────────────────────────────────────────────────────
        # Schema nodes/rels: HAS_PORTFOLIO, HAS_HOLDING, HAS_SECTOR_ALLOCATION,
        #   HAS_REGION_ALLOCATION, OF_ASSET_TYPE, REPRESENTS
        # Node labels: Portfolio, Holding, Sector, Region, AssetCategory
        if any(p in cypher for p in [
            "HAS_HOLDING", "HAS_PORTFOLIO",
            "HAS_SECTOR_ALLOCATION", "HAS_REGION_ALLOCATION",
            "OF_ASSET_TYPE", "REPRESENTS",
            ":Portfolio", ":Holding",
            ":Sector", ":Region", ":AssetCategory",
        ]):
            labels.add("fund_portfolio")

        # ── fund_profile ────────────────────────────────────────────────────────
        # Schema nodes/rels: DEFINED_BY, Profile, summaryProspectus
        # Section labels (only in Profile): Objective, PerformanceCommentary,
        #   Strategy (Strategy also has chunks)
        # Vector indexes: chunkEmbeddingIndex (shared with 10K), profileObjectiveIndex
        # Note: RiskFactor is shared with company_filing — handled separately below
        if any(p in cypher for p in [
            "DEFINED_BY", ":Profile", "summaryProspectus",
            ":Objective", ":PerformanceCommentary", ":Strategy",
            "profileObjectiveIndex",
        ]) or ("Profile" in cypher and "HAS_SECTION" in cypher):
            labels.add("fund_profile")

        # RiskFactor appears in both slices.
        # If company_filing already matched → 10-K context.
        # Otherwise → fund profile context.
        if ":RiskFactor" in cypher and "company_filing" not in labels:
            labels.add("fund_profile")

        # ── fund_basic ──────────────────────────────────────────────────────────
        # Schema nodes/rels: HAS_FINANCIAL_HIGHLIGHT, FinancialHighlight,
        #   HAS_AVERAGE_RETURNS, AverageReturns, HAS_CHART, HAS_TABLE,
        #   HAS_SHARE_CLASS, ShareClass
        # Fund properties unique to this slice: expenseRatio, advisoryFees,
        #   costsPer10k, netIncomeRatio, netAssetsValueBeginning, netAssetsValueEnd,
        #   return1y, return5y, return10y, returnInception, totalReturn, turnover
        # Indexes: fundNameIndex, providerNameIndex, trustNameIndex
        if any(p in cypher for p in [
            "HAS_FINANCIAL_HIGHLIGHT", "FinancialHighlight",
            "HAS_AVERAGE_RETURNS", "AverageReturns",
            "HAS_CHART", "HAS_TABLE",
            "HAS_SHARE_CLASS", ":ShareClass",
            "fundNameIndex", "providerNameIndex", "trustNameIndex",
            "expenseRatio", "advisoryFees", "costsPer10k",
            "netIncomeRatio", "netAssetsValueBeginning", "netAssetsValueEnd",
            "return1y", "return5y", "return10y", "returnInception",
            "totalReturn", "turnover",
        ]):
            labels.add("fund_basic")

        # ── Fallbacks ────────────────────────────────────────────────────────────
        # Fund / Provider / Trust node with no more specific match → fund_basic
        if not labels and any(p in cypher for p in [":Fund", "Fund {", ":Provider", ":Trust"]):
            labels.add("fund_basic")

        # Company node alone (no filing/people context yet) → company_filing
        if not labels and ":Company" in cypher:
            labels.add("company_filing")

        # Last resort
        if not labels:
            labels.add("fund_basic")

        return sorted(labels)

    def load_test_set(self):
        """
        Load test set from JSON file.

        Supports two formats:
        - Old format: [{"text": "...", "labels": ["fund_basic", ...]}, ...]
        - test_set.json format: [{"question": "...", "expected_cypher": "...", "complexity": "..."}, ...]
          Labels are automatically inferred from the expected Cypher query.
        """
        with open(self.test_set_path, "r") as f:
            raw_data = json.load(f)

        # training_data.json wraps splits under "train"/"test" keys — unwrap test split
        if isinstance(raw_data, dict):
            raw_data = raw_data.get("test", raw_data.get("train", []))

        if not raw_data:
            self.test_data = []
            print("✓ Loaded 0 test queries")
            return

        first = raw_data[0]
        if "text" in first or "labels" in first:
            # Already in canonical format
            self.test_data = raw_data
        else:
            # test_set.json format — normalize and infer labels
            self.test_data = []
            label_dist: Counter = Counter()
            for item in raw_data:
                cypher = item.get("expected_cypher", "")
                labels = self._infer_labels_from_cypher(cypher)
                for l in labels:
                    label_dist[l] += 1
                self.test_data.append({
                    "text": item["question"],
                    "labels": labels,
                    # Keep originals for reference
                    "_expected_cypher": cypher,
                    "_complexity": item.get("complexity", ""),
                })

            print(f"  Format: test_set.json (labels inferred from expected Cypher)")
            print(f"  Label distribution: {dict(label_dist.most_common())}")

        print(f"✓ Loaded {len(self.test_data)} test queries")

    def run_setfit_benchmark(self) -> BenchmarkStats:
        """Run SetFit classifier on all test queries."""
        print("\n" + "="*80)
        print("🚀 Running SetFit Benchmark")
        print("="*80)

        setfit = SetFitClassifier()
        self.setfit_results = []

        for i, item in enumerate(self.test_data, 1):
            query = item["text"]
            expected_labels = self._get_labels(item)

            try:
                start = time.time()
                predicted_labels, confidence = setfit.classify(query)
                elapsed_ms = (time.time() - start) * 1000

                result = ClassificationResult(
                    query_id=i,
                    query=query,
                    expected_labels=expected_labels,
                    predicted_labels=predicted_labels,
                    confidence_score=confidence,
                    time_ms=elapsed_ms,
                )
                self.setfit_results.append(result)

                if i % 20 == 0:
                    print(f"  [{i}/{len(self.test_data)}] {query[:60]}... → {predicted_labels} ({elapsed_ms:.1f}ms)")
            except Exception as e:
                result = ClassificationResult(
                    query_id=i,
                    query=query,
                    expected_labels=expected_labels,
                    predicted_labels=[],
                    confidence_score=0.0,
                    time_ms=0,
                    error=str(e),
                )
                self.setfit_results.append(result)
                print(f"  ❌ Q{i}: {e}")

        print(f"\n✓ SetFit benchmark complete ({len(self.setfit_results)} queries)")
        return self._compute_stats(self.setfit_results, "SetFit")

    def run_llm_benchmark(
        self,
        provider_type: str = "openrouter",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> BenchmarkStats:
        """Run LLM classifier on all test queries."""
        print("\n" + "="*80)
        print(f"🚀 Running LLM Benchmark ({provider_type})")
        print("="*80)

        llm = LLMQueryClassifier(provider_type=provider_type, model_id=model_id, api_key=api_key)
        self.llm_results = []

        for i, item in enumerate(self.test_data, 1):
            query = item["text"]
            expected_labels = self._get_labels(item)

            try:
                start = time.time()
                predicted_labels, confidence = llm.classify(query)
                elapsed_ms = (time.time() - start) * 1000

                result = ClassificationResult(
                    query_id=i,
                    query=query,
                    expected_labels=expected_labels,
                    predicted_labels=predicted_labels,
                    confidence_score=confidence,
                    time_ms=elapsed_ms,
                )
                self.llm_results.append(result)

                if i % 5 == 0:
                    print(f"  [{i}/{len(self.test_data)}] {query[:60]}... → {predicted_labels} ({elapsed_ms:.1f}ms)")
            except Exception as e:
                result = ClassificationResult(
                    query_id=i,
                    query=query,
                    expected_labels=expected_labels,
                    predicted_labels=[],
                    confidence_score=0.0,
                    time_ms=0,
                    error=str(e),
                )
                self.llm_results.append(result)
                print(f"  ❌ Q{i}: {str(e)[:100]}")

        print(f"\n✓ LLM benchmark complete ({len(self.llm_results)} queries)")
        return self._compute_stats(self.llm_results, f"LLM ({provider_type})")

    def _compute_stats(self, results: List[ClassificationResult], name: str) -> BenchmarkStats:
        """Compute metrics from classification results."""
        stats = BenchmarkStats(classifier_name=name, total_queries=len(results))

        if not results:
            return stats

        # Separate errors from successes
        successful = [r for r in results if r.error is None]
        stats.error_count = len(results) - len(successful)

        if not successful:
            return stats

        # Binary matrices for multi-label metrics
        y_true = self.mlb.transform([r.expected_labels for r in successful])
        y_pred = self.mlb.transform([r.predicted_labels for r in successful])

        # Exact match accuracy
        stats.correct_exact_match = sum(
            1 for r in successful if set(r.expected_labels) == set(r.predicted_labels)
        )
        # Superset match: predicted contains ALL expected labels (may have extras).
        # Treated as correct because retrieving more categories still satisfies the query.
        correct_superset = sum(
            1 for r in successful
            if set(r.expected_labels).issubset(set(r.predicted_labels))
        )
        stats.accuracy = correct_superset / len(successful)

        # Hamming loss
        stats.hamming_loss = hamming_loss(y_true, y_pred)

        # Timing stats
        times = [r.time_ms for r in successful]
        stats.avg_time_ms = statistics.mean(times)
        stats.median_time_ms = statistics.median(times)
        stats.min_time_ms = min(times)
        stats.max_time_ms = max(times)

        # Per-label metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(self.LABELS)), zero_division=0
        )

        for i, label in enumerate(self.LABELS):
            stats.per_label_metrics[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        # Macro averages
        stats.precision_macro = float(np.mean(precision))
        stats.recall_macro = float(np.mean(recall))
        stats.f1_macro = float(np.mean(f1))

        # Weighted averages
        total_support = np.sum(support)
        stats.precision_weighted = float(np.average(precision, weights=support)) if total_support > 0 else 0.0
        stats.recall_weighted = float(np.average(recall, weights=support)) if total_support > 0 else 0.0
        stats.f1_weighted = float(np.average(f1, weights=support)) if total_support > 0 else 0.0

        # Failed queries
        stats.failed_queries = [r.query_id for r in results if r.error is not None]

        return stats

    def generate_report(
        self,
        setfit_stats: BenchmarkStats,
        llm_stats: Optional[BenchmarkStats] = None,
        output_file: Optional[str] = None,
    ):
        """
        Generate comprehensive report.

        When llm_stats is None or has total_queries==0, a single-column
        SetFit-only report is produced.
        """
        setfit_only = llm_stats is None or llm_stats.total_queries == 0
        report_lines = []

        report_lines.append("=" * 80)
        title = "SETFIT QUERY CLASSIFIER REPORT" if setfit_only else "QUERY CLASSIFIER BENCHMARK REPORT (SetFit vs LLM)"
        report_lines.append(title)
        report_lines.append("=" * 80)
        report_lines.append("")

        # ── Summary ──────────────────────────────────────────────────────────
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 80)
        if setfit_only:
            report_lines.append(f"  Total Queries   : {setfit_stats.total_queries}")
            report_lines.append(f"  Successful      : {setfit_stats.total_queries - setfit_stats.error_count}")
            report_lines.append(f"  Errors          : {setfit_stats.error_count}")
        else:
            report_lines.append(f"{'Metric':<30} {'SetFit':<25} {'LLM':<25}")
            report_lines.append("-" * 80)
            report_lines.append(f"{'Total Queries':<30} {setfit_stats.total_queries:<25} {llm_stats.total_queries:<25}")
            report_lines.append(f"{'Successful':<30} {setfit_stats.total_queries - setfit_stats.error_count:<25} {llm_stats.total_queries - llm_stats.error_count:<25}")
            report_lines.append(f"{'Errors':<30} {setfit_stats.error_count:<25} {llm_stats.error_count:<25}")
        report_lines.append("")

        # ── Accuracy ─────────────────────────────────────────────────────────
        report_lines.append("🎯 ACCURACY METRICS")
        report_lines.append("-" * 80)
        n_sf = setfit_stats.total_queries - setfit_stats.error_count
        if setfit_only:
            report_lines.append(f"  Superset Match Accuracy : {setfit_stats.accuracy*100:.2f}%  (predicted ⊇ expected)")
            report_lines.append(f"  Exact  Match Accuracy   : {setfit_stats.correct_exact_match / n_sf * 100 if n_sf else 0:.2f}%  (predicted == expected)")
            report_lines.append(f"  Correct (superset)      : {round(setfit_stats.accuracy * n_sf)} / {n_sf}")
            report_lines.append(f"  Correct (exact)         : {setfit_stats.correct_exact_match} / {n_sf}")
            report_lines.append(f"  Hamming Loss            : {setfit_stats.hamming_loss:.4f}")
        else:
            n_lm = llm_stats.total_queries - llm_stats.error_count
            report_lines.append(f"{'Superset Match Acc.':<30} {setfit_stats.accuracy*100:>23.2f}% {llm_stats.accuracy*100:>23.2f}%")
            sf_exact = setfit_stats.correct_exact_match / n_sf * 100 if n_sf else 0
            lm_exact = llm_stats.correct_exact_match / n_lm * 100 if n_lm else 0
            report_lines.append(f"{'Exact Match Acc.':<30} {sf_exact:>23.2f}% {lm_exact:>23.2f}%")
            report_lines.append(f"{'Correct (exact)':<30} {setfit_stats.correct_exact_match:>25} {llm_stats.correct_exact_match:>25}")
            report_lines.append(f"{'Hamming Loss':<30} {setfit_stats.hamming_loss:>25.4f} {llm_stats.hamming_loss:>25.4f}")
        report_lines.append("")

        # ── F1 Scores ────────────────────────────────────────────────────────
        report_lines.append("📈 F1 SCORES (Multi-Label)")
        report_lines.append("-" * 80)
        if setfit_only:
            report_lines.append(f"  F1 (Macro)         : {setfit_stats.f1_macro:.4f}")
            report_lines.append(f"  F1 (Weighted)      : {setfit_stats.f1_weighted:.4f}")
            report_lines.append(f"  Precision (Macro)  : {setfit_stats.precision_macro:.4f}")
            report_lines.append(f"  Recall (Macro)     : {setfit_stats.recall_macro:.4f}")
        else:
            report_lines.append(f"{'F1 (Macro)':<30} {setfit_stats.f1_macro:>25.4f} {llm_stats.f1_macro:>25.4f}")
            report_lines.append(f"{'F1 (Weighted)':<30} {setfit_stats.f1_weighted:>25.4f} {llm_stats.f1_weighted:>25.4f}")
            report_lines.append(f"{'Precision (Macro)':<30} {setfit_stats.precision_macro:>25.4f} {llm_stats.precision_macro:>25.4f}")
            report_lines.append(f"{'Recall (Macro)':<30} {setfit_stats.recall_macro:>25.4f} {llm_stats.recall_macro:>25.4f}")
        report_lines.append("")

        # ── Latency ──────────────────────────────────────────────────────────
        report_lines.append("⚡ PERFORMANCE (Response Time)")
        report_lines.append("-" * 80)
        if setfit_only:
            report_lines.append(f"  Avg Latency    : {setfit_stats.avg_time_ms:.2f} ms")
            report_lines.append(f"  Median Latency : {setfit_stats.median_time_ms:.2f} ms")
            report_lines.append(f"  Min / Max      : {setfit_stats.min_time_ms:.2f} ms / {setfit_stats.max_time_ms:.2f} ms")
        else:
            report_lines.append(f"{'Average Latency':<30} {setfit_stats.avg_time_ms:>23.2f} ms {llm_stats.avg_time_ms:>23.2f} ms")
            report_lines.append(f"{'Median Latency':<30} {setfit_stats.median_time_ms:>23.2f} ms {llm_stats.median_time_ms:>23.2f} ms")
            report_lines.append(f"{'Min Latency':<30} {setfit_stats.min_time_ms:>23.2f} ms {llm_stats.min_time_ms:>23.2f} ms")
            report_lines.append(f"{'Max Latency':<30} {setfit_stats.max_time_ms:>23.2f} ms {llm_stats.max_time_ms:>23.2f} ms")
            if llm_stats.avg_time_ms > 0:
                speedup = llm_stats.avg_time_ms / setfit_stats.avg_time_ms
                report_lines.append(f"  → SetFit is {speedup:.1f}x faster than LLM")
        report_lines.append("")

        # ── Per-label breakdown ───────────────────────────────────────────────
        report_lines.append("🏷️  PER-LABEL METRICS")
        report_lines.append("-" * 80)

        has_llm_labels = not setfit_only and llm_stats.per_label_metrics
        if has_llm_labels:
            report_lines.append(f"{'Label':<22} {'Metric':<12} {'SetFit':>10} {'LLM':>10} {'Support':>8}")
        else:
            report_lines.append(f"{'Label':<22} {'Metric':<12} {'SetFit':>10} {'Support':>8}")
        report_lines.append("-" * 80)

        for label in self.LABELS:
            if label not in setfit_stats.per_label_metrics:
                continue
            support = setfit_stats.per_label_metrics[label]["support"]
            if support == 0:
                continue

            sf_f1 = setfit_stats.per_label_metrics[label]["f1"]
            sf_p  = setfit_stats.per_label_metrics[label]["precision"]
            sf_r  = setfit_stats.per_label_metrics[label]["recall"]

            if has_llm_labels and label in llm_stats.per_label_metrics:
                lm_f1 = llm_stats.per_label_metrics[label]["f1"]
                lm_p  = llm_stats.per_label_metrics[label]["precision"]
                lm_r  = llm_stats.per_label_metrics[label]["recall"]
                report_lines.append(f"{label:<22} {'F1':<12} {sf_f1:>10.4f} {lm_f1:>10.4f} {support:>8}")
                report_lines.append(f"{'':<22} {'Precision':<12} {sf_p:>10.4f} {lm_p:>10.4f}")
                report_lines.append(f"{'':<22} {'Recall':<12} {sf_r:>10.4f} {lm_r:>10.4f}")
            else:
                report_lines.append(f"{label:<22} {'F1':<12} {sf_f1:>10.4f} {support:>8}")
                report_lines.append(f"{'':<22} {'Precision':<12} {sf_p:>10.4f}")
                report_lines.append(f"{'':<22} {'Recall':<12} {sf_r:>10.4f}")
            report_lines.append("")

        # ── Misclassified examples — per classifier ───────────────────────────
        def _render_failures(results: List[ClassificationResult], classifier_name: str) -> None:
            truly_wrong = [
                r for r in results
                if r.error is None and not set(r.expected_labels).issubset(set(r.predicted_labels))
            ]
            superset_only = [
                r for r in results
                if r.error is None
                and set(r.expected_labels).issubset(set(r.predicted_labels))
                and set(r.expected_labels) != set(r.predicted_labels)
            ]

            report_lines.append("=" * 80)
            report_lines.append(f"🔍 MISCLASSIFICATION DETAILS — {classifier_name}")
            report_lines.append("=" * 80)

            if not truly_wrong and not superset_only:
                report_lines.append(f"  ✅ No failures or superset predictions for {classifier_name}.")
                report_lines.append("")
                return

            if truly_wrong:
                report_lines.append(
                    f"  ❌ TRUE FAILURES: {len(truly_wrong)} queries where expected labels are missing from prediction"
                )
                report_lines.append(
                    f"     (These count against accuracy — the predicted set does NOT cover all expected labels)"
                )
                report_lines.append("-" * 80)

                # Group by misclassification pattern: expected → predicted
                from collections import defaultdict
                pattern_groups: dict = defaultdict(list)
                for r in truly_wrong:
                    key = (tuple(sorted(r.expected_labels)), tuple(sorted(r.predicted_labels)))
                    pattern_groups[key].append(r)

                for (expected_tuple, predicted_tuple), group in sorted(
                    pattern_groups.items(), key=lambda x: -len(x[1])
                ):
                    expected_str  = ", ".join(expected_tuple)
                    predicted_str = ", ".join(predicted_tuple)
                    report_lines.append(
                        f"  [{len(group):>2}x]  Expected [{expected_str}]  →  Predicted [{predicted_str}]"
                    )
                    report_lines.append("-" * 80)
                    for r in group:
                        report_lines.append(f"    Q{r.query_id:<5} conf={r.confidence_score:.2f}  {r.query[:72]}")
                    report_lines.append("")

                report_lines.append(
                    f"  Total true failures: {len(truly_wrong)} / "
                    f"{len([r for r in results if r.error is None])} evaluated queries"
                )
                report_lines.append("")

            if superset_only:
                report_lines.append(
                    f"  ⚠️  SUPERSET PREDICTIONS: {len(superset_only)} queries with extra labels beyond expected"
                )
                report_lines.append(
                    f"     (Counted as CORRECT — predicted set covers all expected labels but adds more)"
                )
                report_lines.append("-" * 80)

                from collections import defaultdict
                extra_groups: dict = defaultdict(list)
                for r in superset_only:
                    extra = tuple(sorted(set(r.predicted_labels) - set(r.expected_labels)))
                    key = (tuple(sorted(r.expected_labels)), extra)
                    extra_groups[key].append(r)

                for (expected_tuple, extra_tuple), group in sorted(
                    extra_groups.items(), key=lambda x: -len(x[1])
                ):
                    expected_str = ", ".join(expected_tuple)
                    extra_str    = ", ".join(extra_tuple)
                    report_lines.append(
                        f"  [{len(group):>2}x]  Expected [{expected_str}]  +extra: [{extra_str}]"
                    )
                    for r in group:
                        report_lines.append(f"    Q{r.query_id:<5} conf={r.confidence_score:.2f}  {r.query[:72]}")
                    report_lines.append("")

                report_lines.append(
                    f"  Total superset predictions: {len(superset_only)} / "
                    f"{len([r for r in results if r.error is None])} evaluated queries"
                )
                report_lines.append("")

        _render_failures(self.setfit_results, "SetFit")
        if not setfit_only and self.llm_results:
            _render_failures(self.llm_results, f"LLM ({llm_stats.classifier_name})")
        report_lines.append("")

        # ── Overall assessment ────────────────────────────────────────────────
        report_lines.append("=" * 80)
        report_lines.append("📋 ASSESSMENT")
        report_lines.append("=" * 80)

        if setfit_only:
            acc = setfit_stats.accuracy * 100  # superset match
            n_sf = setfit_stats.total_queries - setfit_stats.error_count
            exact_acc = setfit_stats.correct_exact_match / n_sf * 100 if n_sf else 0
            if acc >= 85:
                report_lines.append(f"✅ SetFit superset accuracy: {acc:.1f}% (exact: {exact_acc:.1f}%) — suitable for production use")
            elif acc >= 70:
                report_lines.append(f"⚠️  SetFit superset accuracy: {acc:.1f}% (exact: {exact_acc:.1f}%) — consider more training data")
            else:
                report_lines.append(f"❌ SetFit superset accuracy: {acc:.1f}% (exact: {exact_acc:.1f}%) — model needs improvement")
            report_lines.append(f"⚡ Avg latency: {setfit_stats.avg_time_ms:.1f} ms/query")
        else:
            if setfit_stats.accuracy > llm_stats.accuracy:
                report_lines.append(f"✅ SetFit has better accuracy: {setfit_stats.accuracy*100:.1f}% vs {llm_stats.accuracy*100:.1f}%")
            else:
                report_lines.append(f"✅ LLM has better accuracy: {llm_stats.accuracy*100:.1f}% vs {setfit_stats.accuracy*100:.1f}%")

            if setfit_stats.avg_time_ms < llm_stats.avg_time_ms:
                speedup = llm_stats.avg_time_ms / setfit_stats.avg_time_ms
                report_lines.append(f"⚡ SetFit is {speedup:.1f}x faster ({setfit_stats.avg_time_ms:.1f} ms vs {llm_stats.avg_time_ms:.1f} ms)")
            else:
                speedup = setfit_stats.avg_time_ms / llm_stats.avg_time_ms
                report_lines.append(f"⚡ LLM is {speedup:.1f}x faster ({llm_stats.avg_time_ms:.1f} ms vs {setfit_stats.avg_time_ms:.1f} ms)")

            report_lines.append("")
            report_lines.append("RECOMMENDATION:")
            if setfit_stats.accuracy > 0.85 and setfit_stats.avg_time_ms < 150:
                report_lines.append("  → SetFit is the superior choice: high accuracy with minimal latency")
            elif llm_stats.accuracy > setfit_stats.accuracy and llm_stats.avg_time_ms < 5000:
                report_lines.append("  → LLM provides better accuracy (acceptable for batch processing)")
            else:
                report_lines.append("  → Hybrid: Use SetFit for real-time, LLM for accuracy-critical tasks")

        report_lines.append("=" * 80)

        # Print to console
        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        # Save to file
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"\n💾 Report saved to: {output_path}")

        return report_text


def check_local_lm_studio(host: str = "localhost", port: int = 1234) -> bool:
    """Check if local LM Studio is available."""
    try:
        import httpx
        response = httpx.get(
            f"http://{host}:{port}/v1/models",
            timeout=2.0,
        )
        return response.status_code == 200
    except Exception:
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query Classifier Benchmark")
    parser.add_argument(
        "--test-set",
        type=str,
        default=str(SCRIPT_DIR.parent / "test_set.json"),
        help="Path to test set JSON (supports both classification_test_set.json and test_set.json formats)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "local"],
        default="local",
        help="LLM provider type (default: local LM Studio)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.3-70b-instruct:free",
        help="Model ID for OpenRouter",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or env var OPEN_ROUTER_API_KEY)",
    )
    parser.add_argument(
        "--local-host",
        type=str,
        default="localhost",
        help="Host for local LM Studio",
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=1234,
        help="Port for local LM Studio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/simple_rag/evaluation/query/benchmark_report.txt",
        help="Output report file",
    )
    parser.add_argument(
        "--skip-llm",
        "--setfit-only",
        dest="skip_llm",
        action="store_true",
        help="Skip LLM benchmark — run SetFit only and produce a single-column report",
    )

    args = parser.parse_args()

    # Load test set
    benchmark = QueryClassifierBenchmark(args.test_set)
    benchmark.load_test_set()

    # Run SetFit benchmark
    setfit_stats = benchmark.run_setfit_benchmark()

    # Run LLM benchmark
    if args.skip_llm:
        print("\n⏭️  Skipping LLM benchmark (--setfit-only / --skip-llm set)")
        llm_stats = None
    else:
        # Check if local LM Studio is available
        if args.provider == "local":
            print(f"\n🔍 Checking for local LM Studio at {args.local_host}:{args.local_port}...")
            if check_local_lm_studio(args.local_host, args.local_port):
                print(f"✅ Local LM Studio found!")
            else:
                print(f"⚠️  WARNING: Local LM Studio not responding at {args.local_host}:{args.local_port}")
                print(f"   Make sure LM Studio is running in the background")
                print(f"   If not available, the benchmark will fail with a connection error")

        try:
            llm_stats = benchmark.run_llm_benchmark(
                provider_type=args.provider,
                model_id=args.model,
                api_key=args.api_key,
            )
        except ConnectionError as e:
            print(f"\n❌ LLM Connection Error: {e}")
            print("\n   Options:")
            if args.provider == "local":
                print("   1. Start LM Studio and run the benchmark again")
                print("   2. Or use OpenRouter instead: --provider openrouter")
            print("   3. Or skip LLM benchmark: --skip-llm")
            raise
        except Exception as e:
            print(f"\n❌ LLM benchmark failed: {e}")
            print("   Run with --skip-llm to benchmark SetFit only")
            raise

    # Generate report
    benchmark.generate_report(setfit_stats, llm_stats=llm_stats, output_file=args.output)


if __name__ == "__main__":
    main()
