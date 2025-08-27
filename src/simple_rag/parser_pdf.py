from pathlib import Path
from ocrmypdf.api import ocr  # <- use the API module import
from ocrmypdf import ExitCode
from pypdf import PdfReader
import pikepdf

SLICE = Path("data/processed/example_p468_470.pdf").resolve()
OCRD  = Path("data/processed/example_page_parsed.pdf").resolve()
OCRD.parent.mkdir(parents=True, exist_ok=True)

# Valid, commonly useful kwargs:
# - language: list of ISO codes for Tesseract
# - jobs: parallel workers
# - use_threads: True for threads (easier debug), False/None for processes (faster)
# - skip_text: only OCR pages without a text layer
# - force_ocr: OCR even if a text layer exists
# - redo_ocr: replace poor/previous OCR
# - optimize: 0 (fast), up to 3 (smaller/better)
# - rotate_pages, deskew: page fixes
# - clean, clean_final: remove junk images
# - progress_bar: show a progress bar in the console
# - output_type: e.g., "pdfa", "pdfa-2" if you need PDF/A

code = ocr(
    input_file=str(SLICE),
    output_file=str(OCRD),
    language=["eng"],   # or just ["eng"]
    jobs=4,
    use_threads=True,          # set to False/None for process-based parallelism
    skip_text=True,            # only OCR image-only pages
    force_ocr=False,
    redo_ocr=False,
    optimize=0,
    rotate_pages=True,
    deskew=True,
    clean=False,
    clean_final=False,
    progress_bar=True,         # show progress; supported by the API
    output_type="pdf",      # uncomment if you need PDF/A output
)

n_pages = 0

if code == ExitCode.ok:
    with pikepdf.open(str(OCRD)) as pdf:
        n_pages = len(pdf.pages)

print(f"[OK] OCR’d + fixed PDF has {n_pages} pages → {OCRD}")


