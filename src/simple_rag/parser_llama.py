
import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from llama_parse import LlamaParse

# ---- load environment variables from .env
load_dotenv()
api_key = os.getenv("LLAMAPARSE_API_KEY")
if not api_key:
    raise RuntimeError("LLAMAPARSE_API_KEY not set in .env")

# ---- inputs/outputs
ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "data" / "raw" / "example.pdf"
SLICE_PATH = ROOT / "data" / "processed" / "example_p468_470.pdf"
OUT_PATH = ROOT / "data" / "processed" / "example_p468_470.json"

START_PAGE, END_PAGE = 508, 510

# ---- slice PDF with PyPDF
reader = PdfReader(str(PDF_PATH))
writer = PdfWriter()
for p in range(START_PAGE - 1, END_PAGE):
    writer.add_page(reader.pages[p])
SLICE_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SLICE_PATH, "wb") as f:
    writer.write(f)

# ---- LlamaParse client
instruction = ("Preserve the document structure and content maintaining, mathematical equations \n"
               "Pay special attention also to the table structure and content so that it matches the original one (headers included)")

parser = LlamaParse(
    api_key=api_key,
    result_type="markdown",
    language="en",
    num_workers=4,
    parsing_instruction=instruction
)

try:
    print("[llamaparse] sending slice…")
    docs = parser.load_data(str(SLICE_PATH))
    print("[llamaparse] returned docs:", len(docs))
except Exception as e:
    print("[llamaparse] exception:", repr(e))
    raise

# If nothing came back, print more breadcrumbs
if len(docs) == 0:
    print("[warn] LlamaParse returned 0 docs. Quick checklist:")
    print("  - Is the file really text/scan? Try a different page range.")
    print("  - Try result_type='text' or 'structured'.")
    print("  - Try without slicing: pass the full PDF once.")
    print("  - Check network/proxy; Llama Cloud needs outbound https.")
    print("  - Double-check key validity (a bad key can yield empty results).")

# --------------------
# SAVE JSONL (skip empties)


# --- put this after you have `docs` ---
from pathlib import Path
import re, html

SINGLE_MD = OUT_PATH.with_suffix(".md")

HEADING_RE = re.compile(r"(?m)^(#{1,6})(\S)")                     # ensure '#Title' -> '# Title'
TABLE_ROW_RE = re.compile(r"^\s*\|")                               # lines that look like pipe-table rows
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

def normalize_markdown(md: str) -> str:
    if not md:
        return ""
    # 1) decode HTML entities (&#x3C; -> <)
    md = html.unescape(md)

    # 2) fix headings missing a space: '#Title' -> '# Title'
    md = HEADING_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", md)

    # 3) collapse >2 blank lines to max 2
    md = re.sub(r"\n{3,}", "\n\n", md)

    # 4) ensure a blank line BEFORE headings
    md = re.sub(r"(?m)([^\n])\n(#{1,6}\s)", r"\1\n\n\2", md)

    # 5) ensure a blank line AFTER headings (if next line is not blank)
    md = re.sub(r"(?m)^(#{1,6}\s.+)\n(?!\n|\||-{3,})", r"\1\n\n", md)

    # 6) ensure blank lines around tables (pipe-style)
    lines = md.splitlines()
    out = []
    for i, line in enumerate(lines):
        # insert blank line before a table row if previous non-empty line isn't blank or header
        if TABLE_ROW_RE.match(line):
            if out and out[-1].strip() and not TABLE_ROW_RE.match(out[-1]) and not TABLE_SEP_RE.match(out[-1]):
                out.append("")  # blank line before table
        out.append(line)
        # if this is a table separator row, ensure there's at least one row above & below (renderer-friendly)
        # (we won’t synthesize; just spacing)
        if TABLE_SEP_RE.match(line):
            # ensure there is a blank line after the table end (will be added when next non-table appears)
            pass
    md = "\n".join(out)

    # 7) demote standalone page banners like "# 468" to smaller heading or italic line
    md = re.sub(r"(?m)^#\s+(\d{1,5})\s*$", r"### Page \1", md)

    # 8) strip stray trailing spaces
    md = re.sub(r"[ \t]+(\n)", r"\1", md)

    return md.strip() + "\n"

def safe_anchor(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"\s+", "-", s)
    return s[:80] or "chunk"

toc = ["# Parsed Output", "## Table of Contents"]
parts = []

for i, d in enumerate(docs):
    raw = (getattr(d, "text", "") or "").strip()
    if not raw:
        continue
    md = normalize_markdown(raw)

    # build a visible section header so chunk boundaries are obvious
    meta = dict(getattr(d, "metadata", {}) or {})
    page = meta.get("page_number") or meta.get("page") or "n/a"
    # use first heading in the chunk as title, else "Chunk i"
    m = re.search(r"(?m)^\s*#{1,6}\s+(.+)$", md)
    title = m.group(1).strip() if m else f"Chunk {i}"
    anchor = f"chunk-{i}-{safe_anchor(title)}"

    toc.append(f"- [Chunk {i} — p. {page}: {title}](#{anchor})")
    parts.append(
        f"\n\n---\n\n<a id='{anchor}'></a>\n\n## Chunk {i} — Page {page}\n\n{md}"
    )

final_md = "\n".join(toc) + "\n" + "".join(parts)
SINGLE_MD.parent.mkdir(parents=True, exist_ok=True)
SINGLE_MD.write_text(final_md, encoding="utf-8")
print(f"Saved single markdown → {SINGLE_MD}")
