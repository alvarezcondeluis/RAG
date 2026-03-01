#!/usr/bin/env python3
"""
Training Data Augmenter
=======================
Uses Ollama (local LLM) to paraphrase existing training examples and bring
every class up to TARGET_PER_CLASS examples in training_data.json.

Usage:
    cd src/simple_rag/rag/query
    python augment_training_data.py [--target 100] [--model llama3.2:3b] [--dry-run]
"""

import json
import random
import argparse
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import requests

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATA_PATH    = SCRIPT_DIR / "training_data.json"
OLLAMA_URL   = "http://localhost:11434/api/generate"
TARGET       = 100          # examples per class in the train split
BATCH_SIZE   = 5            # how many paraphrases to request per prompt

# ── Label descriptions (injected into the prompt so the LLM understands scope)
LABEL_DESCRIPTIONS = {
    "not_related": (
        "completely off-topic questions with no relation to investment funds, ETFs, "
        "stocks, or financial filings – e.g. cooking, sport, travel, general science"
    ),
    "fund_basic": (
        "questions about static fund properties such as ticker, name, expense ratio, "
        "net assets, advisory fees, number of holdings, exchange, cost per 10k, "
        "provider, trust, or share class – for specific tickers like VTI, VOO, VO, VB, "
        "VUG, VTV, IVV, IWM, AGG, LQD, QUAL, USMV, MTUM, MCHI, ESGV, DGRO, HDV etc."
    ),
    "fund_performance": (
        "questions about fund returns over time: trailing 1-year / 5-year / 10-year / "
        "inception performance, NAV at beginning or end of year, total return for a "
        "specific year, net income ratio, or turnover HISTORY from financial highlights"
    ),
    "fund_portfolio": (
        "questions about fund holdings (top N holdings, weight of a security), "
        "sector allocation (Technology %, Healthcare % …), geographic allocation "
        "(US %, Europe %, Asia % …), portfolio date, or number of holdings"
    ),
    "fund_profile": (
        "questions answered via vector / semantic search on fund strategy text, "
        "risk description, investment objective, or performance commentary – "
        "e.g. 'find conservative funds', 'funds focused on ESG', 'funds that mention "
        "currency hedging'. These require reading free-text fund descriptions."
    ),
    "company_filing": (
        "questions about 10-K filing sections for individual companies: risk factors, "
        "business description, legal proceedings, management discussion & analysis, "
        "income statement, balance sheet, cash flow, revenue segments – "
        "for companies like Apple (AAPL), Microsoft (MSFT), Tesla (TSLA), "
        "Google (Alphabet), Amazon (AMZN), NVIDIA (NVDA), Meta, Netflix etc."
    ),
    "company_people": (
        "questions about people connected to funds or companies: fund portfolio managers, "
        "company CEOs / CFOs / executives, insider transactions (buys/sells, share counts, "
        "values), executive compensation packages"
    ),
    "hybrid_graph_vector": (
        "questions that need BOTH a graph traversal on fund properties (expense ratio, "
        "returns, holdings, provider) AND a vector search on strategy / risk text – "
        "e.g. 'find ESG funds from Vanguard with low expense ratio', "
        "'conservative strategy funds with good 5-year returns'"
    ),
    "cross_entity": (
        "questions that span multiple entity types together: Fund + Company + Person – "
        "e.g. 'which funds hold Apple and who is Apple's CEO?', "
        "'fund managers whose funds invest in companies with insider activity'"
    ),
}

TICKERS = [
    "VTI","VOO","VGT","VNQ","VIG","BND","VDE","VHT","VCR","VFH","VPU","VAW","VIS","VOX",
    "VTV","VUG","VV","VO","VB","VYM","ESGV","VYMI","VIGI","VWOB","VSMAX","VIMAX",
    "IVV","IWM","AGG","LQD","HYG","TIP","GOVT","IEFA","EFA","MCHI","INDA","ACWI",
    "IBB","SOXX","IYR","DVY","DGRO","HDV","QUAL","USMV","MTUM","VLUE","SIZE",
    "IJS","IJH","IJR","IWO","IWN","SCZ","EFV","EFG","ICF","USRT","MGK","MGC","MGV",
]
COMPANIES = ["Apple","Microsoft","Tesla","Google","Alphabet","Amazon","NVIDIA","Meta","Netflix"]
TICKERS_STR = ", ".join(random.sample(TICKERS, 12))


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def ollama_generate(prompt: str, model: str) -> str:
    """Call Ollama and return the raw response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.85, "num_predict": 600},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.RequestException as e:
        print(f"  ⚠️  Ollama error: {e}")
        return ""


def build_prompt(label: str, existing_examples: list[str], n: int) -> str:
    sample = random.sample(existing_examples, min(6, len(existing_examples)))
    sample_str = "\n".join(f"  - {e}" for e in sample)
    tickers = ", ".join(random.sample(TICKERS, 8))
    companies = ", ".join(random.sample(COMPANIES, 4))
    return f"""You are generating training data for a query classifier that routes financial RAG queries.

CLASS: {label}
DESCRIPTION: {LABEL_DESCRIPTIONS[label]}

EXISTING EXAMPLES (do NOT copy these, only use as style reference):
{sample_str}

TASK: Generate exactly {n} new, diverse query examples for the class "{label}".
- Vary phrasing: use both analyst-style ("Get the expense ratio of …") and conversational ("Can you tell me about …")
- Use varied tickers from this pool (where relevant): {tickers}
- Use varied companies (where relevant): {companies}
- Do NOT repeat any existing example
- Do NOT include explanations or numbering, just one query per line
- Each query must clearly belong to the class "{label}" and not another class

OUTPUT (one query per line, no numbering, no quotes):"""


def parse_response(raw: str, n: int) -> list[str]:
    """Extract clean query lines from LLM response."""
    lines = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•*0123456789.) ").strip('"').strip("'").strip()
        if len(line) > 10 and line not in lines:
            lines.append(line)
        if len(lines) >= n:
            break
    return lines


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",   type=int, default=TARGET,          help="Target examples per class")
    parser.add_argument("--model",    type=str, default="llama3.2:3b",   help="Ollama model name")
    parser.add_argument("--dry-run",  action="store_true",               help="Show plan without calling Ollama")
    parser.add_argument("--class",    dest="only_class", default=None,   help="Only augment one class")
    args = parser.parse_args()

    # ── Load existing data ──────────────────────────────────────────────
    with open(DATA_PATH) as f:
        data = json.load(f)

    train = data["train"]
    counts = Counter(item["label"] for item in train)

    print(f"\n{'='*60}")
    print(f"  Training Data Augmenter  (target = {args.target}/class)")
    print(f"{'='*60}\n")
    print(f"  {'Class':25} {'Current':>8} {'Need':>6}")
    print(f"  {'-'*40}")

    plan: dict[str, int] = {}
    for label in LABEL_DESCRIPTIONS:
        cur  = counts.get(label, 0)
        need = max(0, args.target - cur)
        tag  = "✅" if need == 0 else f"+{need}"
        print(f"  {label:25} {cur:>8}   {tag}")
        if need > 0:
            plan[label] = need

    if args.only_class:
        plan = {k: v for k, v in plan.items() if k == args.only_class}

    if not plan:
        print("\n  ✅ All classes already at target. Nothing to do.\n")
        return

    if args.dry_run:
        print("\n  🔍 Dry-run mode – no Ollama calls made.\n")
        return

    # ── Augment ─────────────────────────────────────────────────────────
    # Index existing examples by label
    by_label: dict[str, list[str]] = defaultdict(list)
    for item in train:
        by_label[item["label"]].append(item["text"])

    new_examples: list[dict] = []
    for label, need in plan.items():
        print(f"\n  ➕ Generating {need} examples for '{label}' …")
        generated: list[str] = []
        attempts = 0
        while len(generated) < need and attempts < 8:
            attempts += 1
            batch = min(BATCH_SIZE, need - len(generated))
            prompt = build_prompt(label, by_label[label] + generated, batch)
            raw = ollama_generate(prompt, args.model)
            lines = parse_response(raw, batch)
            # deduplicate against existing + already generated
            existing_texts = set(by_label[label]) | set(generated)
            for line in lines:
                if line not in existing_texts:
                    generated.append(line)
                    existing_texts.add(line)
            print(f"    attempt {attempts}: got {len(lines)} → total {len(generated)}/{need}")
            time.sleep(0.3)  # be kind to Ollama

        generated = generated[:need]
        print(f"  ✅ Added {len(generated)} examples for '{label}'")

        for text in generated:
            new_examples.append({"text": text, "label": label})
            by_label[label].append(text)  # update index so next batches don't repeat

    # ── Write back ───────────────────────────────────────────────────────
    data["train"] = train + new_examples

    # Final counts
    final_counts = Counter(item["label"] for item in data["train"])
    print(f"\n{'='*60}")
    print("  Final class counts (train split)")
    print(f"{'='*60}")
    for label in LABEL_DESCRIPTIONS:
        print(f"  {label:25} {final_counts.get(label, 0):>4}")

    # Backup original
    backup = DATA_PATH.with_suffix(".json.bak")
    import shutil
    shutil.copy(DATA_PATH, backup)
    print(f"\n  💾 Backed up original → {backup.name}")

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    total_new = len(new_examples)
    print(f"  ✅ Wrote {total_new} new examples to {DATA_PATH.name}")
    print(f"  📊 Total train examples: {len(data['train'])}\n")


if __name__ == "__main__":
    main()
