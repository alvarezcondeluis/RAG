from unstructured.partition.pdf import partition_pdf
import os
import json

PDF_PATH = "data/processed/example_page_parsed2.pdf"
OUT_PATH = "data/processed/page_508.json"
# parse only page 508

chunks = partition_pdf(
    filename=PDF_PATH,
    infer_table_structure=True,
    strategy="hi_res",
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True,
    chunking_strategy="by_title",
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
)

print(f"Extracted {len(chunks)} elements")

# ensure folder exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# save to jsonl
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i, el in enumerate(chunks):
        record = {
            "id": f"chunk-{i}",
            "text": getattr(el, "text", "").strip(),
            "category": getattr(el, "category", ""),
            "page": getattr(el.metadata, "page_number", None),
        }
        if record["text"]:  # skip empty
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved parsed elements → {OUT_PATH}")