"""
LLM-as-Judge evaluator for Text2Cypher benchmark results.

Loads a JSON results file produced by text2cypher_benchmark.py, evaluates each
question/result pair using an LLM, and writes YES/NO verdicts back to the file.

If --results is omitted, the most recently modified benchmark JSON in
src/simple_rag/evaluation/reports/ is used automatically.

Usage:
    uv run python src/simple_rag/evaluation/llm_judge.py
    uv run python src/simple_rag/evaluation/llm_judge.py --failed-only
    uv run python src/simple_rag/evaluation/llm_judge.py --results results.json
    uv run python src/simple_rag/evaluation/llm_judge.py --provider groq --model llama-3.3-70b-versatile
    uv run python src/simple_rag/evaluation/llm_judge.py --provider openai --host localhost --port 1234
    uv run python src/simple_rag/evaluation/llm_judge.py --provider openrouter
    uv run python src/simple_rag/evaluation/llm_judge.py --provider openrouter --model meta-llama/llama-3.3-70b-instruct

Providers:
    openai      — LM Studio or any OpenAI-compatible endpoint (default)
    groq        — Groq cloud API (requires GROQ_API_KEY in .env)
    openrouter  — OpenRouter cloud API (requires OPEN_ROUTER_API_KEY in .env)
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_REPORTS_DIR = SCRIPT_DIR / "reports"
_JUDGE_REPORTS_DIR = _REPORTS_DIR / "llm_judge_reports"


def _find_latest_results() -> Path:
    """Return the most recently modified benchmark JSON in the reports directory.

    Excludes files inside llm_judge_reports/ (those are outputs).
    """
    if not _REPORTS_DIR.exists():
        raise FileNotFoundError(
            f"Reports directory not found: {_REPORTS_DIR}\n"
            "Run the benchmark first or pass --results explicitly."
        )
    candidates = [
        p for p in _REPORTS_DIR.glob("*.json")
        if p.parent == _REPORTS_DIR  # skip subdirectories
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No benchmark JSON files found in {_REPORTS_DIR}\n"
            "Run the benchmark first or pass --results explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _judge_output_paths(results_path: Path) -> tuple[Path, Path]:
    """Return (json_path, txt_path) inside llm_judge_reports/."""
    _JUDGE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = results_path.stem
    return (
        _JUDGE_REPORTS_DIR / f"{stem}_llm_judge.json",
        _JUDGE_REPORTS_DIR / f"{stem}_llm_judge.txt",
    )

# ── Judge prompt ──────────────────────────────────────────────────────────────
_JUDGE_PROMPT = """\
You are evaluating whether a database query result can answer a financial data question.
The database contains SEC filings, mutual fund profiles, ETF holdings, and company 10-K data.

Question asked:
{question}

Result returned by the generated query ({gen_count} record(s)):
{generated_result}

Reference result for comparison ({exp_count} record(s)):
{expected_result}

Your primary task: decide whether the generated result contains enough information to answer the question.
The reference result is a guide, not a strict template — the generated result may be formatted differently, return more records, or include additional entities and still be correct.

Rules:
- CORRECT if the generated result contains the information needed to answer the question, even with extra fields or a different format
- CORRECT if numeric values match within reasonable precision (e.g. 0.03 vs 0.030)
- CORRECT if the generated result returns more records than the reference (e.g. extra chunks, extra companies) as long as the relevant answer is present within those records
- CORRECT if the generated result is a better or more complete answer than the reference
- INCORRECT if the generated result is empty and the question has a meaningful answer
- INCORRECT if the generated result contains only wrong values (wrong fund, wrong company, wrong metric) and the correct entity's data is absent
- INCORRECT if the generated result is completely off-topic and cannot answer the question at all

Answer on the first line with exactly YES (correct) or NO (incorrect).
Then write one sentence explaining your reasoning.
"""


_CYPHER_CAP = 500   # characters before truncating a Cypher string
_RESULT_CAP = 5     # max records shown per result block
_TEXT_CAP   = 300   # characters before truncating a text/blob value in a record


def _cap(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"… [{len(text)-limit} more chars]"


def _format_records(records: List[Dict], limit: int = _RESULT_CAP) -> str:
    if not records:
        return "(empty — no records returned)"
    shown = records[:limit]
    lines = []
    for r in shown:
        # Cap any long string values inside a record before serialising
        capped = {
            k: _cap(v, _TEXT_CAP) if isinstance(v, str) and len(v) > _TEXT_CAP else v
            for k, v in r.items()
        }
        lines.append(json.dumps(capped, ensure_ascii=False))
    if len(records) > limit:
        lines.append(f"... and {len(records) - limit} more record(s)")
    return "\n".join(lines)


def _build_prompt(entry: Dict) -> str:
    return _JUDGE_PROMPT.format(
        question=entry["question"],
        gen_count=entry.get("gen_record_count", len(entry.get("generated_results", []))),
        generated_result=_format_records(entry.get("generated_results", [])),
        exp_count=entry.get("exp_record_count", len(entry.get("expected_results", []))),
        expected_result=_format_records(entry.get("expected_results", [])),
    )


def _parse_response(content: str) -> tuple[str, str]:
    """Parse LLM response into (verdict, reasoning)."""
    content = content.strip()
    lines = content.split("\n", 1)
    first = lines[0].strip().upper()
    verdict = "YES" if "YES" in first else "NO"
    reasoning = lines[1].strip() if len(lines) > 1 else ""
    return verdict, reasoning


# ── Provider clients ──────────────────────────────────────────────────────────

def _make_openai_client(host: str, port: int):
    from openai import OpenAI
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="lm-studio")


def _make_groq_client():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _make_openrouter_client():
    from simple_rag.rag.llm_providers.openrouter_provider import OpenRouterProvider
    provider = OpenRouterProvider()
    return provider.client


def _call_judge(client, model: str, prompt: str, provider: str) -> tuple[str, str]:
    """Call the LLM judge and return (verdict, reasoning)."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    content = response.choices[0].message.content or ""
    return _parse_response(content)


# ── Human-readable report ─────────────────────────────────────────────────────

def _build_txt_report(
    results: List[Dict],
    to_judge: List[Dict],
    results_by_id: Dict[int, Dict],
    yes_count: int,
    no_count: int,
    errors: int,
    false_negatives: int,
    false_positives: int,
    rule_accuracy: float,
    judge_accuracy: float,
    provider: str,
    model: str,
    source_file: str,
    auto_yes_count: int = 0,
    null_expected_count: int = 0,
) -> str:
    SEP  = "=" * 80
    SEP2 = "-" * 80
    lines: List[str] = []

    lines += [
        SEP,
        "LLM-AS-JUDGE REPORT",
        SEP,
        f"Source file       : {source_file}",
        f"Provider          : {provider.upper()} / {model}",
        f"Entries judged    : {yes_count + no_count}  (errors: {errors})",
        f"Auto-passed       : {auto_yes_count}  (identical results — no LLM call)",
        f"Skipped           : {null_expected_count}  (null expected results)",
        "",
    ]

    # ── Per-question entries ──────────────────────────────────────────────────
    for entry in sorted(results_by_id.values(), key=lambda r: r["question_id"]):
        qid     = entry["question_id"]
        verdict = entry.get("judge_verdict")
        if verdict is None:
            continue  # not judged (routing / pipeline_error / skipped)

        rule_ok  = entry.get("success", False)
        outcome_tag = entry.get("outcome", "")

        flip = ""
        if verdict == "YES" and not rule_ok:
            flip = "  [FALSE NEGATIVE — rule failed, judge passed]"
        elif verdict == "NO" and rule_ok:
            flip = "  [FALSE POSITIVE — rule passed, judge failed]"

        lines += [
            SEP,
            f"Q{qid}  |  Judge: {verdict}{flip}  |  Rule-based: {'PASS' if rule_ok else 'FAIL (' + outcome_tag + ')'}",
            SEP2,
        ]

        # Question
        lines += ["QUESTION:", f"  {entry.get('question', '')}", ""]

        # Cypher
        exp_cypher = _cap(entry.get("expected_cypher") or "(none)", _CYPHER_CAP)
        gen_cypher = _cap(entry.get("generated_cypher") or "(none)", _CYPHER_CAP)
        lines += [
            "EXPECTED CYPHER:",
            f"  {exp_cypher}",
            "",
            "GENERATED CYPHER:",
            f"  {gen_cypher}",
            "",
        ]

        # Results
        exp_records = entry.get("expected_results", [])
        gen_records = entry.get("generated_results", [])
        lines += [
            f"EXPECTED RESULTS  ({len(exp_records)} record(s)):",
            _format_records(exp_records),
            "",
            f"GENERATED RESULTS  ({len(gen_records)} record(s)):",
            _format_records(gen_records),
            "",
        ]

        # Judge verdict + reasoning
        reasoning = entry.get("judge_reasoning", "")
        lines += [
            f"JUDGE VERDICT: {verdict}",
            f"REASONING: {reasoning}",
            "",
        ]

    # ── Summary ───────────────────────────────────────────────────────────────
    judged = yes_count + no_count
    delta  = judge_accuracy - rule_accuracy

    lines += [
        SEP,
        "JUDGE SUMMARY",
        SEP,
        f"  Entries judged          : {judged}",
        f"    Judge YES             : {yes_count}  ({yes_count/judged*100:.1f}%)" if judged else "    Judge YES             : 0",
        f"    Judge NO              : {no_count}  ({no_count/judged*100:.1f}%)"  if judged else "    Judge NO              : 0",
        f"    Errors                : {errors}",
        SEP2,
        f"  False negatives (rule FAIL, judge YES) : {false_negatives}",
        f"  False positives (rule PASS, judge NO)  : {false_positives}",
        SEP2,
        f"  Rule-based accuracy     : {rule_accuracy:.2f}%",
        f"  Judge-corrected accuracy: {judge_accuracy:.2f}%",
        f"  Delta                   : {delta:+.2f}%  ({'rule undercounts' if delta > 0 else 'rule overcounts' if delta < 0 else 'identical'})",
        SEP,
    ]

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _results_equal(gen: List[Dict], exp: List[Dict]) -> bool:
    """True if both record lists contain the same values regardless of key names or order."""
    if len(gen) != len(exp):
        return False
    def _values(record: Dict):
        return tuple(sorted(str(v) for v in record.values()))
    return sorted(_values(r) for r in gen) == sorted(_values(r) for r in exp)


# ── Main judge loop ───────────────────────────────────────────────────────────

def run_judge(
    results_path: Path,
    provider: str,
    model: str,
    host: str,
    port: int,
    failed_only: bool,
    delay_ms: int,
    json_path: Path,
    txt_path: Path,
) -> None:
    # Load results
    with open(results_path, encoding="utf-8") as f:
        results: List[Dict] = json.load(f)

    # Filter entries to judge
    skippable_outcomes = {"routing", "pipeline_error"}

    auto_yes_count = 0
    null_expected_count = 0

    for r in results:
        if r.get("outcome") in skippable_outcomes or not r.get("generated_cypher"):
            continue
        gen = r.get("generated_results") or []
        exp = r.get("expected_results") or []
        # Auto-YES: results are identical (rule-based false negative due to column name mismatch etc.)
        if gen and exp and _results_equal(gen, exp):
            r["judge_verdict"] = "YES"
            r["judge_reasoning"] = "Auto-pass: generated results are identical to expected results."
            auto_yes_count += 1

    to_judge = [
        r for r in results
        if r.get("outcome") not in skippable_outcomes
        and r.get("generated_cypher")
        and r.get("judge_verdict") is None       # skip already auto-passed
        and (r.get("expected_results") or [])    # skip null expected results
        and (not failed_only or not r.get("success", False))
    ]

    null_expected_count = sum(
        1 for r in results
        if r.get("outcome") not in skippable_outcomes
        and r.get("generated_cypher")
        and r.get("judge_verdict") is None
        and not (r.get("expected_results") or [])
    )

    print(f"\n{'='*70}")
    print(f"🔍 LLM-as-Judge — {provider.upper()} / {model}")
    print(f"{'='*70}")
    print(f"Total results loaded : {len(results)}")
    print(f"  Auto-passed (identical results) : {auto_yes_count}")
    print(f"  Skipped (null expected results) : {null_expected_count}")
    print(f"Entries to judge     : {len(to_judge)}")
    if failed_only:
        print("Mode                 : failed queries only")
    print(f"{'='*70}\n")

    if not to_judge:
        print("Nothing to judge.")
        return

    # Build client
    if provider == "openai":
        client = _make_openai_client(host, port)
    elif provider == "groq":
        client = _make_groq_client()
    elif provider == "openrouter":
        client = _make_openrouter_client()
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Build a lookup by question_id for fast update
    results_by_id: Dict[int, Dict] = {r["question_id"]: r for r in results}

    yes_count = 0
    no_count = 0
    errors = 0

    for i, entry in enumerate(to_judge, 1):
        qid = entry["question_id"]
        question = entry["question"]
        current_success = entry.get("success", False)
        outcome = entry.get("outcome", "")

        print(f"[{i}/{len(to_judge)}] Q{qid}: {question[:70]}")
        print(f"         Rule-based: {'✅ pass' if current_success else f'❌ {outcome}'}")

        prompt = _build_prompt(entry)
        try:
            verdict, reasoning = _call_judge(client, model, prompt, provider)
        except Exception as e:
            print(f"         ⚠️  Judge call failed: {e}")
            results_by_id[qid]["judge_verdict"] = "ERROR"
            results_by_id[qid]["judge_reasoning"] = str(e)
            errors += 1
            continue

        results_by_id[qid]["judge_verdict"] = verdict
        results_by_id[qid]["judge_reasoning"] = reasoning

        marker = "✅" if verdict == "YES" else "❌"
        flip = ""
        if verdict == "YES" and not current_success:
            flip = "  ← FALSE NEGATIVE (rule said fail, judge says pass)"
        elif verdict == "NO" and current_success:
            flip = "  ← FALSE POSITIVE (rule said pass, judge says fail)"

        print(f"         Judge:      {marker} {verdict} — {reasoning}{flip}")

        if verdict == "YES":
            yes_count += 1
        else:
            no_count += 1

        if delay_ms > 0 and i < len(to_judge):
            time.sleep(delay_ms / 1000)

    # ── Summary stats ─────────────────────────────────────────────────────────
    judged = yes_count + no_count
    false_negatives = sum(
        1 for r in to_judge
        if results_by_id[r["question_id"]].get("judge_verdict") == "YES"
        and not r.get("success", False)
    ) + auto_yes_count  # auto-passed identical results are also rule false-negatives

    false_positives = sum(
        1 for r in to_judge
        if results_by_id[r["question_id"]].get("judge_verdict") == "NO"
        and r.get("success", False)
    )

    # judge_pass: LLM YES + auto YES + rule pass not overridden by judge NO
    judge_pass = sum(
        1 for r in results
        if r.get("judge_verdict") == "YES"
        or (r.get("judge_verdict") is None and r.get("success", False))
    )
    judge_accuracy = judge_pass / len(results) * 100 if results else 0.0
    rule_accuracy  = sum(1 for r in results if r.get("success", False)) / len(results) * 100 if results else 0.0
    delta = judge_accuracy - rule_accuracy

    print(f"\n{'='*70}")
    print("📊 JUDGE SUMMARY")
    print(f"{'='*70}")
    print(f"Judged entries (LLM)    : {judged}")
    print(f"  Auto-passed (identical): {auto_yes_count}")
    print(f"  Skipped (null expected): {null_expected_count}")
    if judged:
        print(f"  Judge YES             : {yes_count} ({yes_count/judged*100:.1f}%)")
        print(f"  Judge NO              : {no_count} ({no_count/judged*100:.1f}%)")
    print(f"  Errors                : {errors}")
    print(f"─{'─'*69}")
    print(f"False negatives (rule ❌, judge ✅) : {false_negatives}")
    print(f"False positives (rule ✅, judge ❌) : {false_positives}")
    print(f"─{'─'*69}")
    print(f"Rule-based accuracy     : {rule_accuracy:.2f}%")
    print(f"Judge-corrected accuracy: {judge_accuracy:.2f}%")
    print(f"Delta                   : {delta:+.2f}% ({'rule undercounts' if delta > 0 else 'rule overcounts' if delta < 0 else 'identical'})")
    print(f"{'='*70}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(results_by_id.values()), f, indent=2, ensure_ascii=False)
    print(f"\n💾 Judged JSON  → {json_path}")

    # ── Save TXT report ───────────────────────────────────────────────────────
    txt_report = _build_txt_report(
        results=results,
        to_judge=to_judge,
        results_by_id=results_by_id,
        yes_count=yes_count,
        no_count=no_count,
        errors=errors,
        false_negatives=false_negatives,
        false_positives=false_positives,
        rule_accuracy=rule_accuracy,
        judge_accuracy=judge_accuracy,
        provider=provider,
        model=model,
        source_file=results_path.name,
        auto_yes_count=auto_yes_count,
        null_expected_count=null_expected_count,
    )
    txt_path.write_text(txt_report, encoding="utf-8")
    print(f"📄 Human report → {txt_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluator for Text2Cypher benchmark results"
    )
    parser.add_argument(
        "--results", default=None,
        help="Path to JSON results file (default: latest in evaluation/reports/)",
    )
    parser.add_argument(
        "--provider", choices=["openai", "groq", "openrouter"], default="openai",
        help="LLM provider (default: openai for LM Studio)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name/ID to use as judge (default: qwen2.5-coder for openai/groq, meta-llama/llama-3.3-70b-instruct:free for openrouter)",
    )
    parser.add_argument("--host", default="localhost", help="OpenAI-compatible host")
    parser.add_argument("--port", type=int, default=1234, help="OpenAI-compatible port")
    parser.add_argument(
        "--failed-only", action="store_true",
        help="Only judge entries that failed the rule-based check",
    )
    parser.add_argument(
        "--delay-ms", type=int, default=500,
        help="Delay between API calls in ms to avoid rate limits (default: 500)",
    )

    args = parser.parse_args()

    if args.model is None:
        args.model = (
            "meta-llama/llama-3.3-70b-instruct"
            if args.provider == "openrouter"
            else "qwen2.5-coder"
        )

    if args.results is None:
        try:
            results_path = _find_latest_results()
            print(f"📂 Auto-selected results: {results_path}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)
    else:
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"❌ Results file not found: {results_path}")
            sys.exit(1)

    json_path, txt_path = _judge_output_paths(results_path)

    run_judge(
        results_path=results_path,
        provider=args.provider,
        model=args.model,
        host=args.host,
        port=args.port,
        failed_only=args.failed_only,
        delay_ms=args.delay_ms,
        json_path=json_path,
        txt_path=txt_path,
    )


if __name__ == "__main__":
    main()
