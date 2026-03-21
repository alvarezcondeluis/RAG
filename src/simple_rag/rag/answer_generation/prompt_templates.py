"""
Prompt templates for answer generation.

These are separate from the Text2Cypher prompts (in rag/prompt_templates.py).
They transform raw Neo4j results into natural language answers for investors.
"""

from typing import Any, Dict, List

from simple_rag.rag.answer_generation.result_classifier import ResultType

# ── System prompt ────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """You are a senior financial analyst assistant specializing in SEC filings, mutual funds, and ETFs. You help investors understand fund and company data from official SEC EDGAR filings.

GUIDELINES:
1. Always cite specific numbers from the data provided. Never fabricate figures.
2. Present financial metrics with proper formatting ($X.XB for billions, $X.XM for millions, X.XX% for percentages).
3. When comparing values, note the reporting period or date when available.
4. Flag any data limitations or caveats (e.g., "as of the most recent filing").
5. If the data does not contain enough information to fully answer the question, say so explicitly rather than guessing.
6. Do NOT reference the database, Cypher queries, or knowledge graph — speak as if you naturally know this from the SEC filings.
7. Use professional but accessible language suitable for retail investors.
8. Structure your response clearly: lead with the direct answer, then provide context.
9. When data includes multiple entities, present them in a clear comparison format.
10. For numerical data, round appropriately (2 decimal places for percentages, whole numbers for large dollar amounts)."""


# ── Schema context descriptions ──────────────────────────────────────────────

_SCHEMA_CONTEXT = {
    "fund_basic": "Fund properties and financial metrics from N-CSR and prospectus filings, including expense ratios, net assets, turnover rates, and annual returns.",
    "fund_portfolio": "Portfolio holdings data from N-PORT filings, including individual security positions with share counts, market values, and portfolio weights.",
    "fund_profile": "Fund strategy, risk factors, and investment objective descriptions extracted from prospectus filings.",
    "company_filing": "10-K annual report sections including risk factors, business descriptions, management discussion, and financial statements.",
    "company_people": "Executive compensation data from DEF 14A proxy statements and insider transaction records from Form 4 filings.",
}


# ── Per-type user prompt templates ───────────────────────────────────────────

_TYPE_INSTRUCTIONS = {
    ResultType.FINANCIAL_METRICS: """Provide a clear answer that:
- Highlights the key metrics the investor asked about
- Provides brief context for what these numbers mean for an investor
- Notes the reporting period if dates are present in the data
- Compares values if multiple data points or funds are present""",

    ResultType.HOLDINGS_TABLE: """Provide a clear answer that:
- Lists the most relevant holdings from the data
- Notes portfolio concentration if weight data is available
- Mentions the total number of holdings if the data includes it
- Highlights any notable positions (large weights, well-known companies)""",

    ResultType.TEXT_CHUNKS: """Provide a clear answer that:
- Summarizes the key points from the filing text relevant to the question
- Preserves important details and specific language from the filings
- Organizes information thematically if multiple sections are returned
- Notes which type of filing section the information comes from""",

    ResultType.CHART_SVG: """Chart data is available and will be displayed visually alongside your answer.
Briefly describe what the chart shows and highlight any notable trends or observations from the accompanying data.""",

    ResultType.ALLOCATION_DATA: """Provide a clear answer that:
- Presents the allocation breakdown clearly (sector, region, or asset class)
- Highlights the largest and smallest allocations
- Notes the reporting date if available
- Mentions any notable concentration or diversification patterns""",

    ResultType.PEOPLE_COMPENSATION: """Provide a clear answer that:
- Presents compensation figures clearly with proper formatting
- Provides context for pay-for-performance if shareholder return data is available
- Notes the reporting period
- For insider transactions, note the transaction type and significance""",

    ResultType.COMPANY_FINANCIALS: """Provide a clear answer that:
- Presents financial figures with proper formatting
- Highlights key metrics and trends
- Notes the fiscal year or reporting period
- Provides brief context for what the numbers indicate about the company's financial health""",

    ResultType.GENERIC: """Provide a clear, structured answer based on the data. Lead with the direct answer to the question, then provide supporting details.""",

    ResultType.EMPTY: """The query returned no results from the database. Let the investor know that no matching data was found and suggest they rephrase their question or check entity names (fund tickers, company names).""",
}


# ── Build the full user prompt ───────────────────────────────────────────────

def build_answer_prompt(
    user_query: str,
    neo4j_results: List[Dict[str, Any]],
    result_type: ResultType,
    query_category: str = "",
    enrichment_context: str = "",
) -> str:
    """Build the user prompt for answer generation.

    Args:
        user_query: The original investor question.
        neo4j_results: Raw results from Neo4j (list of dicts).
        result_type: Classified result type.
        query_category: SetFit category (e.g. "fund_basic").
        enrichment_context: Pre-formatted supplementary context from enrichment
                            queries (fund overview, managers, returns, etc.).

    Returns:
        Formatted user prompt string.
    """
    # Schema context
    schema_context = _SCHEMA_CONTEXT.get(
        query_category,
        "Financial data from SEC filings stored in the knowledge graph."
    )

    # Format results — strip embeddings and SVGs from the text representation
    skip_keys = {"embedding", "svg"}
    formatted_rows = []
    for row in neo4j_results[:50]:  # Cap at 50 rows to avoid huge prompts
        cleaned = {k: v for k, v in row.items() if k not in skip_keys}
        formatted_rows.append(str(cleaned))
    results_text = "\n".join(formatted_rows) if formatted_rows else "(no data)"

    # Type-specific instructions
    instructions = _TYPE_INSTRUCTIONS.get(result_type, _TYPE_INSTRUCTIONS[ResultType.GENERIC])

    # Build enrichment block if present
    enrichment_block = ""
    if enrichment_context:
        enrichment_block = f"""

Additional context about the entities mentioned (use this to provide a richer, more complete answer):
{enrichment_context}
"""

    prompt = f"""The investor asked: "{user_query}"

Data source context: {schema_context}

Retrieved data ({len(neo4j_results)} records):
{results_text}
{enrichment_block}
{instructions}"""

    return prompt
