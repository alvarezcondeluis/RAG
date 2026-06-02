#!/usr/bin/env python3
"""
Entity Resolver Benchmark

Tests the EntityResolver against queries that:
  - Contain typos / misspellings in fund or company names
  - Use partial names or colloquial references
  - Have no entity at all (false-positive risk)
  - Mix multiple entities in the same sentence
  - Use ticker symbols embedded in natural language

For each query we record:
  - What was resolved (name, type, score)
  - Whether a resolution was expected
  - Outcome: TP / FP / FN / TN

Usage:
    uv run python -m simple_rag.evaluation.query.entity_benchmark
    uv run python -m simple_rag.evaluation.query.entity_benchmark --output report.txt
    uv run python -m simple_rag.evaluation.query.entity_benchmark --debug
"""

from __future__ import annotations

import sys
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
SRC_ROOT = SCRIPT_DIR.parent.parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ── Test case definitions ─────────────────────────────────────────────────────

@dataclass
class EntityTestCase:
    """A single test case for the entity resolver."""
    id: int
    category: str          # typo | partial | colloquial | no_entity | ticker | multi
    query: str
    expect_resolution: bool  # True = at least one entity should be found
    # If expect_resolution=True, the best match should contain one of these strings
    # (case-insensitive, checked against resolved entity name OR ticker)
    expected_match_hints: list[str] = field(default_factory=list)
    notes: str = ""


TEST_CASES: list[EntityTestCase] = [
    # ── Exact ticker in sentence ───────────────────────────────────────────
    EntityTestCase(1,  "ticker",     "What is the expense ratio of VTI?",
                   True,  ["VTI"],        "clean ticker, should always resolve"),
    EntityTestCase(2,  "ticker",     "Show me the top 10 holdings of VOO.",
                   True,  ["VOO"],        "clean ticker"),
    EntityTestCase(3,  "ticker",     "What is AAPL's revenue?",
                   True,  ["AAPL"],       "company ticker"),
    EntityTestCase(4,  "ticker",     "Compare VGT and VTI expense ratios.",
                   True,  ["VGT", "VTI"], "two tickers in one query"),
    EntityTestCase(5,  "ticker",     "I want info about VTI, please.",
                   True,  ["VTI"],        "ticker with punctuation"),

    # ── Typos / misspellings ───────────────────────────────────────────────
    EntityTestCase(6,  "typo",       "What is the expanse ratio of Vangaurd Total Stock Market?",
                   True,  ["vanguard total stock", "VTI"],  "typo in 'Vanguard'"),
    EntityTestCase(7,  "typo",       "Show me the investement strategies for Vangard 500 index fund.",
                   True,  ["vanguard 500", "VOO", "VFFSX"], "double typo"),
    EntityTestCase(8,  "typo",       "What are the riks factors for the Vangaurd S&P 500 ETF?",
                   True,  ["vanguard", "VOO"],               "typo + partial"),
    EntityTestCase(9,  "typo",       "Microsft 10-K risk section.",
                   True,  ["MSFT", "Microsoft"],             "company name typo"),
    EntityTestCase(10, "typo",       "Show me Appl's financial highlights.",
                   True,  ["AAPL", "Apple"],                 "common Apple typo"),
    EntityTestCase(11, "typo",       "What does Amazoon's annual report say about AWS?",
                   True,  ["AMZN", "Amazon"],                "Amazon typo"),
    EntityTestCase(12, "typo",       "Vanguard Totaal Bond Market fund performance.",
                   True,  ["vanguard total bond", "BND"],    "typo in 'Total'"),

    # ── Partial names ──────────────────────────────────────────────────────
    EntityTestCase(13, "partial",    "Tell me about the Total Stock Market ETF.",
                   True,  ["vanguard total stock", "VTI"],   "partial fund name"),
    EntityTestCase(14, "partial",    "What is the strategy of the S&P 500 index fund?",
                   True,  ["vanguard 500", "VOO"],           "common S&P 500 reference"),
    EntityTestCase(15, "partial",    "Show me the Information Technology ETF holdings.",
                   True,  ["information technology", "VGT"], "partial sector ETF name"),
    EntityTestCase(16, "partial",    "What sectors does the Healthcare fund invest in?",
                   True,  ["healthcare", "VHT"],             "partial healthcare fund"),
    EntityTestCase(17, "partial",    "How has the Real Estate ETF performed?",
                   True,  ["real estate", "VNQ"],            "partial REIT ETF"),

    # ── Colloquial / indirect references ──────────────────────────────────
    EntityTestCase(18, "colloquial", "What does the big Vanguard stock market ETF invest in?",
                   True,  ["vanguard", "VTI"],               "colloquial reference"),
    EntityTestCase(19, "colloquial", "Vanguard's healthcare offering strategy.",
                   True,  ["healthcare", "vanguard"],        "colloquial fund reference"),
    EntityTestCase(20, "colloquial", "Apple's latest annual filing risk section.",
                   True,  ["AAPL", "Apple"],                 "colloquial company reference"),
    EntityTestCase(21, "colloquial", "the tech ETF from Vanguard",
                   True,  ["VGT", "technology", "vanguard"], "informal tech ETF reference"),

    # ── No entity — should NOT resolve (false positive risk) ──────────────
    EntityTestCase(22, "no_entity",  "Which funds have the lowest expense ratio?",
                   False, [],                                "general query, no entity"),
    EntityTestCase(23, "no_entity",  "Find funds that focus on small-cap stocks.",
                   False, [],                                "general, no entity"),
    EntityTestCase(24, "no_entity",  "What are the risk factors for a typical bond fund?",
                   False, [],                                "generic bond fund, no entity"),
    EntityTestCase(25, "no_entity",  "Which ETFs have the best 5-year returns?",
                   False, [],                                "comparison query, no entity"),
    EntityTestCase(26, "no_entity",  "Show me funds with high turnover.",
                   False, [],                                "screening query, no entity"),
    EntityTestCase(27, "no_entity",  "What CEO received the highest compensation last year?",
                   False, [],                                "general CEO query"),
    EntityTestCase(28, "no_entity",  "Which companies filed a 10-K in 2023?",
                   False, [],                                "general company query"),
    EntityTestCase(29, "no_entity",  "How does Nvidia's supply chain exposure affect fund risk?",
                   False, [],                                "indirect mention, should NOT lock on Nvidia as provider"),
    EntityTestCase(30, "no_entity",  "What are typical advisory fees for index funds?",
                   False, [],                                "generic financial question"),

    # ── Syntax errors / garbled queries ───────────────────────────────────
    EntityTestCase(31, "syntax",     "vti expense???",
                   True,  ["VTI"],                           "minimal garbled query with ticker"),
    EntityTestCase(32, "syntax",     "VOO holding top ten what are",
                   True,  ["VOO"],                           "garbled word order"),
    EntityTestCase(33, "syntax",     "aapl msft who has more revenue",
                   True,  ["AAPL", "MSFT"],                  "two garbled company tickers"),
    EntityTestCase(34, "syntax",     "VTI VGT both compare strategies",
                   True,  ["VTI", "VGT"],                    "two tickers, garbled syntax"),

    # ── Ambiguous / tricky ─────────────────────────────────────────────────
    EntityTestCase(35, "ambiguous",  "What are the risks of investing in bond markets?",
                   False, [],                                "generic market mention, not a specific fund"),
    EntityTestCase(36, "ambiguous",  "How do index funds track the market?",
                   False, [],                                "general education question"),
    EntityTestCase(37, "ambiguous",  "Which provider manages the most funds?",
                   False, [],                                "general provider question — should NOT false-match a provider"),
    EntityTestCase(38, "ambiguous",  "What trust has the most funds issued?",
                   False, [],                                "general trust question — false-positive risk"),

    # ── Vanguard fund names — typos ────────────────────────────────────────
    EntityTestCase(39, "typo",       "What are the holdings of the Vangaurd Value ETF?",
                   True,  ["vanguard value", "VTV"],         "typo in Vanguard + Value fund"),
    EntityTestCase(40, "typo",       "Show me the Vangard Growth Index performance.",
                   True,  ["vanguard growth", "VUG"],        "typo: Vangard Growth"),
    EntityTestCase(41, "typo",       "Vanguard Large-Cap Indeks Fund strategy.",
                   True,  ["vanguard large-cap", "VV"],      "typo: Indeks instead of Index"),
    EntityTestCase(42, "typo",       "What is the expense ratio of Vangurd Extended Market?",
                   True,  ["vanguard extended market", "VXF"], "typo: Vangurd Extended"),
    EntityTestCase(43, "typo",       "Vanguard Mid-Cap Groth Index holdings.",
                   True,  ["vanguard mid-cap growth", "VOT"], "typo: Groth instead of Growth"),
    EntityTestCase(44, "typo",       "Vanguard Smal-Cap Value Fund top holdings.",
                   True,  ["vanguard small-cap value", "VBR"], "typo: Smal-Cap"),
    EntityTestCase(45, "typo",       "Show Vanguard Comunication Services ETF sector allocation.",
                   True,  ["vanguard communication services", "VOX"], "typo: Comunication"),
    EntityTestCase(46, "typo",       "Vanguard Utillities sector fund expense ratio.",
                   True,  ["vanguard utilities", "VPU"],     "typo: Utillities"),
    EntityTestCase(47, "typo",       "Vanguard Consummer Discretionary ETF holdings.",
                   True,  ["vanguard consumer discretionary", "VCR"], "typo: Consummer"),
    EntityTestCase(48, "typo",       "Vanguard ESG US Stok ETF strategy.",
                   True,  ["vanguard esg", "ESGV"],          "typo: Stok instead of Stock"),
    EntityTestCase(49, "typo",       "Vanguard Materialls Index Fund performance.",
                   True,  ["vanguard materials", "VAW"],     "typo: Materialls"),

    # ── Vanguard fund names — partial / colloquial ─────────────────────────
    EntityTestCase(50, "partial",    "What does the Vanguard Value fund invest in?",
                   True,  ["vanguard value", "VTV"],         "partial: just Value fund"),
    EntityTestCase(51, "partial",    "Show holdings for the Vanguard Mid-Cap ETF.",
                   True,  ["vanguard mid-cap", "VO"],        "partial mid-cap reference"),
    EntityTestCase(52, "partial",    "How has the Vanguard Small-Cap Growth fund performed?",
                   True,  ["vanguard small-cap growth", "VBK"], "partial small-cap growth"),
    EntityTestCase(53, "partial",    "Vanguard Extended Duration Treasury fund returns.",
                   True,  ["vanguard extended duration", "EDV"], "partial duration treasury"),
    EntityTestCase(54, "partial",    "Vanguard Global Wellington strategy section.",
                   True,  ["vanguard global wellington", "VGWLX"], "partial Wellington fund"),
    EntityTestCase(55, "colloquial", "the ESG stock ETF from Vanguard",
                   True,  ["vanguard esg", "ESGV"],          "colloquial ESG stock reference"),
    EntityTestCase(56, "colloquial", "Vanguard's materials offering",
                   True,  ["vanguard materials", "VAW"],     "colloquial materials fund"),
    EntityTestCase(57, "colloquial", "the small cap ETF from Vanguard",
                   True,  ["vanguard small-cap", "VB"],      "colloquial small-cap reference"),
    EntityTestCase(58, "colloquial", "Vanguard's income fund with Wellington in the name",
                   True,  ["wellesley", "VWINX"],            "colloquial Wellesley Income"),
    EntityTestCase(59, "colloquial", "the Vanguard FTSE Social fund",
                   True,  ["vanguard ftse social", "VFTAX"], "colloquial FTSE Social reference"),

    # ── Company names — typos ──────────────────────────────────────────────
    EntityTestCase(60, "typo",       "Show me Aple Inc annual report risk factors.",
                   True,  ["AAPL", "Apple"],                 "typo: Aple"),
    EntityTestCase(61, "typo",       "Microsofft Corp 10-K financial highlights.",
                   True,  ["MSFT", "Microsoft"],             "typo: Microsofft"),
    EntityTestCase(62, "typo",       "What does Alphabt's 10-K say about AI risks?",
                   True,  ["GOOGL", "Alphabet"],             "typo: Alphabt"),
    EntityTestCase(63, "typo",       "Amzon quarterly revenue from the annual filing.",
                   True,  ["AMZN", "Amazon"],                "typo: Amzon"),
    EntityTestCase(64, "typo",       "Teslla Inc risk factors from latest 10-K.",
                   True,  ["TSLA", "Tesla"],                 "typo: Teslla"),
    EntityTestCase(65, "typo",       "Nvdia Corp earnings and revenue metrics.",
                   True,  ["NVDA", "Nvidia"],                "typo: Nvdia"),
    EntityTestCase(66, "typo",       "Meta Platfroms financials from 10-K.",
                   True,  ["META", "Meta"],                  "typo: Platfroms"),
    EntityTestCase(67, "typo",       "Netflx Inc revenue and income statement.",
                   True,  ["NFLX", "Netflix"],               "typo: Netflx"),
    EntityTestCase(68, "typo",       "Wlt Disney 10-K business overview.",
                   True,  ["DIS", "Disney"],                 "typo: Wlt Disney"),
    EntityTestCase(69, "typo",       "Coca Col Co dividend history.",
                   True,  ["KO", "Coca Cola"],               "typo: Coca Col"),
    EntityTestCase(70, "typo",       "Pepscio earnings and financial metrics.",
                   True,  ["PEP", "Pepsico"],                "typo: Pepscio"),
    EntityTestCase(71, "typo",       "Costco Whoesale Corp 10-K filing.",
                   True,  ["COST", "Costco"],                "typo: Whoesale"),
    EntityTestCase(72, "typo",       "Wallmart 10-K risk section.",
                   True,  ["WMT", "Walmart"],                "typo: Wallmart"),
    EntityTestCase(73, "typo",       "Nik Inc annual report strategy.",
                   True,  ["NKE", "Nike"],                   "typo: Nik"),
    EntityTestCase(74, "typo",       "Starbcks Corp financial highlights.",
                   True,  ["SBUX", "Starbucks"],             "typo: Starbcks"),
    EntityTestCase(75, "typo",       "Pfzer Inc drug pipeline risk factors.",
                   True,  ["PFE", "Pfizer"],                 "typo: Pfzer"),
    EntityTestCase(76, "typo",       "Jonson & Johnson 10-K legal proceedings.",
                   True,  ["JNJ", "Johnson"],                "typo: Jonson"),
    EntityTestCase(77, "typo",       "Exxon Mobil Crp annual earnings.",
                   True,  ["XOM", "Exxon"],                  "typo: Crp"),
    EntityTestCase(78, "typo",       "Vsa Inc financial metrics from annual report.",
                   True,  ["V", "Visa"],                     "typo: Vsa"),
    EntityTestCase(79, "typo",       "PayPal Holdngs financial highlights.",
                   True,  ["PYPL", "PayPal"],                "typo: Holdngs"),

    # ── Company names — partial / colloquial ───────────────────────────────
    EntityTestCase(80, "partial",    "Show Apple's revenue breakdown.",
                   True,  ["AAPL", "Apple"],                 "partial: just 'Apple'"),
    EntityTestCase(81, "partial",    "What did Microsoft report about cloud growth?",
                   True,  ["MSFT", "Microsoft"],             "partial company name"),
    EntityTestCase(82, "partial",    "Alphabet's AI risk section.",
                   True,  ["GOOGL", "Alphabet"],             "partial Alphabet reference"),
    EntityTestCase(83, "partial",    "Amazon's logistics strategy from 10-K.",
                   True,  ["AMZN", "Amazon"],                "partial Amazon reference"),
    EntityTestCase(84, "colloquial", "the streaming company Netflix 10-K",
                   True,  ["NFLX", "Netflix"],               "colloquial streaming company"),
    EntityTestCase(85, "colloquial", "Tesla's electric vehicle risk factors",
                   True,  ["TSLA", "Tesla"],                 "colloquial EV reference"),
    EntityTestCase(86, "colloquial", "big Nvidia chip revenue from annual filing",
                   True,  ["NVDA", "Nvidia"],                "colloquial Nvidia reference"),
    EntityTestCase(87, "colloquial", "the social media company Meta financials",
                   True,  ["META", "Meta"],                  "colloquial Meta reference"),

    # ── No-entity guards for new company/fund domain words ─────────────────
    EntityTestCase(88, "no_entity",  "Which tech companies have 10-K filings?",
                   False, [],                                "generic tech company query"),
    EntityTestCase(89, "no_entity",  "Find consumer goods companies with high revenue.",
                   False, [],                                "generic consumer goods query"),
    EntityTestCase(90, "no_entity",  "What ESG funds are available in the market?",
                   False, [],                                "generic ESG question — no specific fund"),
    EntityTestCase(91, "no_entity",  "Which small-cap ETFs have the lowest expense ratio?",
                   False, [],                                "generic small-cap ETF comparison"),
    EntityTestCase(92, "no_entity",  "Show me all communication services stocks.",
                   False, [],                                "generic sector query"),
    EntityTestCase(93, "no_entity",  "What are the risks in the materials sector?",
                   False, [],                                "generic materials sector question"),
    EntityTestCase(94, "no_entity",  "How does consumer discretionary spending affect ETF performance?",
                   False, [],                                "generic macro question"),
    EntityTestCase(95, "no_entity",  "What utilities companies pay the best dividends?",
                   False, [],                                "generic utilities dividend query"),
]


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class EntityResult:
    case: EntityTestCase
    resolved: dict         # raw output from EntityResolver.extract_entities()
    elapsed_ms: float
    outcome: str           # TP | FP | FN | TN
    error: Optional[str] = None

    @property
    def top_entity(self) -> Optional[tuple[str, dict]]:
        """Return the highest-scoring resolved entity."""
        if not self.resolved:
            return None
        return max(self.resolved.items(), key=lambda x: x[1].get("score", 0))


# ── Outcome logic ─────────────────────────────────────────────────────────────

def _compute_outcome(case: EntityTestCase, resolved: dict) -> str:
    """
    TP — expected resolution AND at least one hint matched
    FP — expected NO resolution but something was resolved
    FN — expected resolution but nothing was resolved (or hint not matched)
    TN — expected NO resolution AND nothing was resolved
    """
    has_resolution = bool(resolved)

    if not case.expect_resolution:
        return "FP" if has_resolution else "TN"

    # Expected a resolution
    if not has_resolution:
        return "FN"

    # Something was resolved — check if any hint appears in resolved keys/tickers
    if not case.expected_match_hints:
        return "TP"   # no specific hint required, just any resolution

    parts: list[str] = []
    for k, v in resolved.items():
        parts.append(k.lower())
        parts.append(str(v.get("ticker", "")).lower())
        parts.append(str(v.get("fund", "")).lower())
        parts.append(str(v.get("company", "")).lower())
    resolved_text = " ".join(parts)
    for hint in case.expected_match_hints:
        if hint.lower() in resolved_text:
            return "TP"

    # Something resolved but didn't match any hint — count as wrong resolution
    return "FP"


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(debug: bool = False) -> list[EntityResult]:
    import neo4j as neo4j_lib
    from simple_rag.database.neo4j.config import settings
    from simple_rag.rag.entity_resolver import EntityResolver

    print("\n╔══════════════════════════════════════════════╗")
    print("║    Entity Resolver Benchmark                 ║")
    print("╚══════════════════════════════════════════════╝\n")

    driver = neo4j_lib.GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print("✓ Neo4j connected\n")

    resolver = EntityResolver(driver, debug=debug)
    print()

    results: list[EntityResult] = []

    for case in TEST_CASES:
        try:
            start = time.time()
            resolved = resolver.extract_entities(case.query)
            elapsed_ms = (time.time() - start) * 1000
            outcome = _compute_outcome(case, resolved)
        except Exception as exc:
            results.append(EntityResult(
                case=case, resolved={}, elapsed_ms=0,
                outcome="FN" if case.expect_resolution else "TN",
                error=str(exc),
            ))
            print(f"  ❌ Q{case.id:02d}: ERROR — {exc}")
            continue

        result = EntityResult(case=case, resolved=resolved, elapsed_ms=elapsed_ms, outcome=outcome)
        results.append(result)

        icon = {"TP": "✓", "TN": "✓", "FP": "✗", "FN": "✗"}[outcome]
        color = {"TP": "\033[92m", "TN": "\033[92m", "FP": "\033[91m", "FN": "\033[91m"}[outcome]
        reset = "\033[0m"

        top = result.top_entity
        top_str = f"→ {top[0]!r} (score={top[1].get('score', 0):.0f}, type={top[1].get('type','?')})" if top else "→ (nothing resolved)"
        print(f"  {color}{icon} Q{case.id:02d} [{outcome}]{reset} [{case.category:<12}] {case.query[:55]:<55}  {top_str}")

    driver.close()
    return results


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(results: list[EntityResult]) -> str:
    lines: list[str] = []

    tp = [r for r in results if r.outcome == "TP"]
    tn = [r for r in results if r.outcome == "TN"]
    fp = [r for r in results if r.outcome == "FP"]
    fn = [r for r in results if r.outcome == "FN"]
    errors = [r for r in results if r.error]

    total = len(results)
    correct = len(tp) + len(tn)
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall    = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_rate   = len(fp) / (len(fp) + len(tn)) if (len(fp) + len(tn)) > 0 else 0.0
    avg_ms    = sum(r.elapsed_ms for r in results if not r.error) / max(1, total - len(errors))

    lines += [
        "=" * 80,
        "ENTITY RESOLVER BENCHMARK REPORT",
        "=" * 80,
        "",
        "📊 SUMMARY",
        "-" * 80,
        f"  Total test cases  : {total}",
        f"  Correct (TP + TN) : {correct} / {total}  ({correct/total*100:.1f}%)",
        f"  Errors            : {len(errors)}",
        "",
        f"  True Positives    : {len(tp):>3}  (resolved correctly when entity expected)",
        f"  True Negatives    : {len(tn):>3}  (correctly resolved nothing when no entity)",
        f"  False Positives   : {len(fp):>3}  (resolved something when should not / wrong entity)",
        f"  False Negatives   : {len(fn):>3}  (failed to resolve when entity expected)",
        "",
        f"  Precision         : {precision:.3f}",
        f"  Recall            : {recall:.3f}",
        f"  F1 Score          : {f1:.3f}",
        f"  FP Rate           : {fp_rate:.3f}  (fraction of no-entity queries that got a false hit)",
        f"  Avg latency       : {avg_ms:.1f} ms/query",
        "",
    ]

    # ── Per-category breakdown ────────────────────────────────────────────────
    lines += ["📂 PER-CATEGORY BREAKDOWN", "-" * 80]
    categories = sorted({r.case.category for r in results})
    for cat in categories:
        cat_results = [r for r in results if r.case.category == cat]
        cat_correct = sum(1 for r in cat_results if r.outcome in ("TP", "TN"))
        lines.append(f"  {cat:<14}  {cat_correct}/{len(cat_results)} correct  "
                     f"  TP={sum(1 for r in cat_results if r.outcome=='TP')} "
                     f"TN={sum(1 for r in cat_results if r.outcome=='TN')} "
                     f"FP={sum(1 for r in cat_results if r.outcome=='FP')} "
                     f"FN={sum(1 for r in cat_results if r.outcome=='FN')}")
    lines.append("")

    # ── Full per-query table ──────────────────────────────────────────────────
    lines += ["📋 FULL RESULTS", "-" * 80,
              f"  {'Q':>3}  {'Out':>3}  {'Cat':<12}  {'Query':<50}  {'Resolved (top)':<40}  Score"]
    lines.append("  " + "-" * 118)

    for r in results:
        top = r.top_entity
        if top:
            res_str = f"{top[0][:35]!r} [{top[1].get('type','?')}]"
            score_str = f"{top[1].get('score', 0):.0f}"
        else:
            res_str = "(nothing)"
            score_str = "—"

        err_tag = " !" if r.error else ""
        lines.append(
            f"  Q{r.case.id:02d}  {r.outcome:<3}  {r.case.category:<12}  "
            f"{r.case.query[:48]:<50}  {res_str:<40}  {score_str}{err_tag}"
        )
    lines.append("")

    # ── False Positives detail ────────────────────────────────────────────────
    if fp:
        lines += ["=" * 80, "⚠️  FALSE POSITIVES — Resolved when no entity expected (or wrong entity)", "=" * 80]
        for r in fp:
            lines.append(f"  Q{r.case.id:02d} [{r.case.category}] {r.case.query}")
            lines.append(f"       Notes  : {r.case.notes}")
            for name, info in r.resolved.items():
                lines.append(f"       Resolved: {name!r:<40} type={info.get('type','?'):<12} score={info.get('score',0):.1f}")
            lines.append("")

    # ── False Negatives detail ────────────────────────────────────────────────
    if fn:
        lines += ["=" * 80, "❌  FALSE NEGATIVES — Expected resolution but got nothing (or wrong)", "=" * 80]
        for r in fn:
            lines.append(f"  Q{r.case.id:02d} [{r.case.category}] {r.case.query}")
            lines.append(f"       Expected hints : {r.case.expected_match_hints}")
            lines.append(f"       Notes          : {r.case.notes}")
            if r.resolved:
                for name, info in r.resolved.items():
                    lines.append(f"       Got instead    : {name!r:<40} type={info.get('type','?'):<12} score={info.get('score',0):.1f}")
            else:
                lines.append(f"       Got instead    : (nothing)")
            if r.error:
                lines.append(f"       Error          : {r.error}")
            lines.append("")

    # ── Assessment ────────────────────────────────────────────────────────────
    lines += ["=" * 80, "📋 ASSESSMENT", "=" * 80]
    if f1 >= 0.85:
        lines.append(f"  ✅ F1={f1:.3f} — Entity resolver is performing well overall.")
    elif f1 >= 0.70:
        lines.append(f"  ⚠️  F1={f1:.3f} — Acceptable but has room for improvement.")
    else:
        lines.append(f"  ❌ F1={f1:.3f} — Entity resolver needs tuning.")

    if fp_rate > 0.2:
        lines.append(f"  ⚠️  High false-positive rate ({fp_rate:.1%}) on no-entity queries — consider raising score thresholds.")
    else:
        lines.append(f"  ✓  False-positive rate on no-entity queries: {fp_rate:.1%}")

    if recall < 0.7:
        lines.append(f"  ⚠️  Low recall ({recall:.1%}) — resolver is missing entities in typo/partial queries.")
    else:
        lines.append(f"  ✓  Recall on entity queries: {recall:.1%}")

    lines.append("=" * 80)

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entity Resolver Benchmark")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to this file (default: reports/entity/entity_benchmark_<timestamp>.txt)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable EntityResolver debug output (shows scores per candidate)")
    args = parser.parse_args()

    results = run_benchmark(debug=args.debug)
    report = generate_report(results)

    print("\n" + report)

    # Determine output path: use --output if provided, otherwise default to reports/entity/
    if args.output:
        out = Path(args.output)
    else:
        # Default: save to reports/entity/ with timestamp
        from datetime import datetime
        reports_dir = PROJECT_ROOT / "src" / "simple_rag" / "evaluation" / "reports" / "entity"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = reports_dir / f"entity_benchmark_{timestamp}.txt"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\n💾 Report saved to: {out}")


if __name__ == "__main__":
    main()
