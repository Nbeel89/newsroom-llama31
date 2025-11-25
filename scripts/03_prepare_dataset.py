#!/usr/bin/env python
"""
03_prepare_dataset.py

Build a supervised fine-tuning dataset for LLaMA from cleaned
Al Jazeera articles.

Input:  data/cleaned/aljazeera_articles_cleaned.jsonl
Output: data/cleaned/newsroom_finetune.jsonl

Each output record:

{
  "instruction": "...",
  "input": "ARTICLE:\\n<full article text>",
  "output": "HEADLINE: ...\\nLEDE: ...\\n- bullet 1\\n- bullet 2\\n- bullet 3"
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple


def sentence_split(text: str) -> List[str]:
    """
    Very simple sentence splitter based on punctuation.
    Not perfect, but good enough for training labels.
    """
    # Normalise weird chars a bit
    text = text.replace("Â", "")
    # Split on ., ?, ! followed by space/cap/newline
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    # Clean up and drop empty
    return [s.strip() for s in parts if s.strip()]


def build_lede_and_bullets(
    text: str, max_lede_words: int = 40, max_bullet_words: int = 35
) -> Tuple[str, List[str]]:
    """
    Create a lede (<= max_lede_words) and up to 3 bullets from article text.
    Lede is built from the first sentence(s) until the word budget is reached.
    Bullets are the next 3 sentences (truncated).
    """
    sentences = sentence_split(text)
    if not sentences:
        return "", []

    # Build LEDE from first sentence(s) until <= max_lede_words
    lede_tokens: List[str] = []
    lede_idx = 0
    while lede_idx < len(sentences) and len(lede_tokens) < max_lede_words:
        toks = sentences[lede_idx].split()
        for t in toks:
            if len(lede_tokens) >= max_lede_words:
                break
            lede_tokens.append(t)
        lede_idx += 1

    lede = " ".join(lede_tokens).strip()

    # Build up to 3 bullets from the next sentences
    bullets: List[str] = []
    for sent in sentences[lede_idx:lede_idx + 6]:  # look ahead a bit
        if len(bullets) >= 3:
            break
        toks = sent.split()
        if not toks:
            continue
        # truncate bullet length
        if len(toks) > max_bullet_words:
            toks = toks[:max_bullet_words]
        bullet = " ".join(toks).strip()
        if bullet:
            bullets.append(bullet)

    # If we still have < 3 bullets, backfill from later sentences
    if len(bullets) < 3:
        for sent in sentences[lede_idx + 6:]:
            if len(bullets) >= 3:
                break
            toks = sent.split()
            if len(toks) > max_bullet_words:
                toks = toks[:max_bullet_words]
            bullet = " ".join(toks).strip()
            if bullet:
                bullets.append(bullet)

    # At most 3 bullets
    bullets = bullets[:3]

    return lede, bullets


def build_example(record: dict) -> dict | None:
    """
    Turn one cleaned article into an instruction/input/output triple.
    """
    title = (record.get("title") or "").strip()
    text = (record.get("text") or "").strip()

    if not text:
        return None

    # Fallback headline if missing
    headline = title if title else "News update"

    lede, bullets = build_lede_and_bullets(text)
    if not lede:
        return None

    # Ensure exactly 3 bullets by padding with shorter ones if needed
    if len(bullets) == 0:
        # fallback: slice later part of article
        bullets = [lede]
    while len(bullets) < 3:
        bullets.append(bullets[-1])

    bullets = bullets[:3]

    instruction = (
        "You are an Al Jazeera–style news editor. "
        "Read the ARTICLE and produce a factual, neutral news summary "
        "with this structure:\n"
        "- A concise HEADLINE.\n"
        "- A LEDE of at most 40 words.\n"
        "- Exactly three short factual bullet points highlighting key details."
    )

    input_text = f"ARTICLE:\n{text}"

    output_lines = [
        f"HEADLINE: {headline}",
        f"LEDE: {lede}",
        f"- {bullets[0]}",
        f"- {bullets[1]}",
        f"- {bullets[2]}",
    ]
    output_text = "\n".join(output_lines)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare fine-tune dataset from cleaned AJ articles."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Path to cleaned articles JSONL (data/cleaned/...)." ,
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Path to write newsroom_finetune.jsonl.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of examples (for quick experiments).",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prep] Reading cleaned articles from: {in_path}")
    print(f"[prep] Writing fine-tune dataset to: {out_path}")

    total = 0
    kept = 0

    with in_path.open("r", encoding="utf-8") as f_in, \
            out_path.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[prep] Skipping bad JSON line #{total}")
                continue

            ex = build_example(rec)
            if ex is None:
                continue

            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

            if args.max_examples is not None and kept >= args.max_examples:
                break

    print(f"[prep] Done. Input articles: {total}, output examples: {kept}")


if __name__ == "__main__":
    main()
