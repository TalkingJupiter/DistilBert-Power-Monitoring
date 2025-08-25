#!/usr/bin/env python3
"""Extract raw text from the Amazon Polarity dataset for LM distillation.

- Uses Hugging Face `datasets` to load `amazon_polarity`.
- Writes one *review text* per line to `data/dump.txt` (no labels).
- Keeps only reasonably sized lines to avoid extremely short/long samples.

Usage:
  python distillation/generate_dataset_amazon.py  # writes data/dump.txt

If you want a labeled dump for classification experiments, see:
  distillation/data/make_dump.py
"""
import os
from datasets import load_dataset

MIN_CHARS = int(os.environ.get("MIN_CHARS", 30))
MAX_CHARS = int(os.environ.get("MAX_CHARS", 600))
OUT_DIR = os.environ.get("OUT_DIR", "data")
OUT_FILE = os.path.join(OUT_DIR, "dump.txt")

def main():
    print("[INFO] Loading amazon_polarity (train + test)...")
    ds = load_dataset("amazon_polarity")

    os.makedirs(OUT_DIR, exist_ok=True)
    kept = 0
    with open(OUT_FILE, "w", encoding="utf-8") as w:
        for split in ("train", "test"):
            dsplit = ds[split]
            for i, row in enumerate(dsplit):
                # Fields: {label: 0|1, title: str, content: str}
                text = (row.get("content") or "").strip().replace("\n", " ")
                if MIN_CHARS <= len(text) <= MAX_CHARS:
                    w.write(text + "\n")
                    kept += 1
                if i and i % 200_000 == 0:
                    print(f"[PROGRESS] {split}: processed {i:,} examples, kept {kept:,}")

    print(f"[DONE] Wrote {kept} lines to {OUT_FILE}")
    print("[NEXT] Run binarization, e.g.:")
    print(f" python scripts/binarized_data.py \n--file_path data/dump.txt \n--tokenizer_type bert \n--tokenizer_name bert-base-uncased \n--dump_file data/binarized_text ")

if __name__ == "__main__":
    main()
