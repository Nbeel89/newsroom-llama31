"""
03_prepare_aj_style_dataset.py

Builds a simple AJ-style SFT dataset for LoRA v3.

INPUT:
    data/raw/aljazeera_articles_full.jsonl

    Each line looks like:
    {
      "url": "...",
      "title": "...",
      "published": "... or null",
      "content": "...full article text...",
      "word_count": 593,
      "scraped_at": "2025-11-24T09:37:00Z"
    }

OUTPUT:
    data/cleaned/aj_style_sft_v3.jsonl

    Each line:
    {
      "text": "<title>\\n\\n<content>"
    }

We do:
  - Skip lines with no content
  - Optionally skip very short articles (min_word_count)
"""

import json
import pathlib


def main():
    # Project root = parent of scripts/
    root = pathlib.Path(__file__).resolve().parent.parent

    # Input crawled file (adjust if yours is differently named)
    in_path = root / "data" / "raw" / "aljazeera_articles_full.jsonl"

    # Output cleaned SFT dataset
    out_dir = root / "data" / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aj_style_sft_v3.jsonl"

    # Filter threshold: ignore very small briefs
    min_word_count = 150  # change if you want more/less aggressive filtering

    print(f"[prep-aj-style] input : {in_path}")
    print(f"[prep-aj-style] output: {out_path}")

    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    n_in = 0
    n_out = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue

            n_in += 1

            # Parse JSONL line
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[prep-aj-style] Skipping bad JSON line #{n_in}")
                continue

            title = (obj.get("title") or "").strip()
            content = (obj.get("content") or "").strip()
            wc = obj.get("word_count")  # may be int or None

            # Skip if no content
            if not content:
                continue

            # Optional filter: skip very short pieces
            if isinstance(wc, int) and wc < min_word_count:
                continue

            # Build training text: title + blank line + content
            if title:
                text = f"{title}\n\n{content}"
            else:
                text = content

            text = text.strip()
            if not text:
                continue

            out_obj = {"text": text}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[prep-aj-style] Done. Read {n_in} lines, wrote {n_out} examples to {out_path}")


if __name__ == "__main__":
    main()
