#!/usr/bin/env python
"""
02_clean_articles.py

Clean raw Al Jazeera crawl JSONL into a normalized dataset for
downstream dataset preparation and LLaMA fine-tuning.

Usage (from repo root):

  python scripts/02_clean_articles.py \
      --in data/raw/aljazeera_articles_full.jsonl \
      --out data/cleaned/aljazeera_articles_cleaned.jsonl \
      --min-words 80 \
      --max-words 2500
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime


def normalise_whitespace(text: str) -> str:
    """Collapse whitespace and strip."""
    # replace weird non-breaking spaces etc.
    text = text.replace("\u00a0", " ")
    # collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_record(raw: dict, min_words: int, max_words: int) -> dict | None:
    """
    Take one raw crawl record and return a cleaned version or None if unusable.
    Expected raw keys: url, title, content, published, word_count, scraped_at
    """
    url = (raw.get("url") or "").strip()
    title = (raw.get("title") or "").strip()
    content = (raw.get("content") or "").strip()
    published = (raw.get("published") or "").strip()
    scraped_at = (raw.get("scraped_at") or "").strip()

    if not url or not content:
        return None

    # Normalise whitespace
    title_clean = normalise_whitespace(title)
    content_clean = normalise_whitespace(content)

    # Compute word count robustly
    words = content_clean.split()
    word_count = len(words)

    if word_count < min_words or word_count > max_words:
        return None

    # Try to parse published datetime, but don't crash if it fails
    pub_iso = None
    if published:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(published, fmt)
                pub_iso = dt.isoformat()
                break
            except ValueError:
                continue
    if not pub_iso:
        pub_iso = None  # keep as None if unknown

    cleaned = {
        "url": url,
        "title": title_clean,
        "published": pub_iso,        # may be None
        "text": content_clean,       # unified field name for article body
        "word_count": word_count,
        "scraped_at": scraped_at or None,
    }
    return cleaned


def dedupe_on_url(records):
    """Simple deduplication by URL (first occurrence wins)."""
    seen = set()
    for rec in records:
        url = rec["url"]
        if url in seen:
            continue
        seen.add(url)
        yield rec


def main():
    parser = argparse.ArgumentParser(
        description="Clean raw crawl JSONL into normalized article records."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Path to raw JSONL from crawler (data/raw/...).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Path to write cleaned JSONL (data/cleaned/...).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=80,
        help="Minimum word count for an article to keep (default: 80).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=2500,
        help="Maximum word count (to avoid absurdly long articles).",
    )

    args = parser.parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[clean] Reading from: {in_path}")
    print(f"[clean] Writing to:   {out_path}")
    print(f"[clean] Min words: {args.min_words}, Max words: {args.max_words}")

    total = 0
    kept = 0
    cleaned_records = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                print(f"[clean] Skipping bad JSON line (#{total})")
                continue

            rec = clean_record(raw, args.min_words, args.max_words)
            if rec is None:
                continue
            cleaned_records.append(rec)

    # Deduplicate by URL
    deduped = list(dedupe_on_url(cleaned_records))
    kept = len(deduped)

    with out_path.open("w", encoding="utf-8") as f_out:
        for rec in deduped:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[clean] Done. Raw records: {total}, after cleaning+dedupe: {kept}")


if __name__ == "__main__":
    main()
