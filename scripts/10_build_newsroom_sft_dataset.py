import json
import re
from pathlib import Path
from typing import List


def clean_content(raw: str) -> List[str]:
    """
    Clean raw AJ article content:
    - Remove 'list X of Y ...' navigation lines
    - Strip whitespace
    - Return list of non-empty paragraphs
    """
    paragraphs = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove 'list 1 of 3 ...' style lines
        if re.match(r"^list \d+ of \d+ ", line):
            continue
        paragraphs.append(line)
    return paragraphs


def build_lede(paragraphs: List[str], max_words: int = 60) -> str:
    """
    Use the first paragraph as the lede, optionally truncating to max_words
    to keep it reasonably short.
    """
    if not paragraphs:
        return ""
    first_para = paragraphs[0].strip()
    words = first_para.split()
    if len(words) <= max_words:
        return first_para
    # Truncate softly
    return " ".join(words[:max_words]) + "..."


def build_id_from_url(url: str) -> str:
    """
    Create a stable ID from the URL slug.
    Example:
      https://www.aljazeera.com/news/2025/11/22/waves-of-sudanese-families-flee-expanding-war-arrive-in-impoverished-chad
      -> waves-of-sudanese-families-flee-expanding-war-arrive-in-impoverished-chad
    """
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = slug or "article"
    # Replace any weird characters with hyphen
    slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", slug)
    return slug.lower()


def main():
    # Adjust these paths to match your repo
    input_path = Path("data/raw/aljazeera_articles_full.jsonl")  # your crawled file
    output_path = Path("data/train/newsroom_v2_train.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            url = rec.get("url")
            title = (rec.get("title") or "").strip()
            content = rec.get("content") or ""
            word_count = rec.get("word_count") or 0

            # Basic filters: need title + content with some length
            if not url or not title or not content:
                continue
            if isinstance(word_count, int) and word_count < 150:
                # Skip very short items (e.g. briefs, stubs)
                continue

            paragraphs = clean_content(content)
            if len(paragraphs) < 2:
                # Skip if we don't really have a meaningful article
                continue

            article_text = "\n\n".join(paragraphs)
            lede = build_lede(paragraphs, max_words=60)
            if not lede:
                continue

            sample_id = build_id_from_url(url)

            # Build the SFT-style input and output
            sft_input = (
                "You are an Al Jazeera–style news editor. "
                "Use ONLY the information in ARTICLE. "
                "Do NOT add facts, numbers, places or background that are not explicitly mentioned.\n\n"
                f"ARTICLE:\n{article_text}"
            )

            sft_output = (
                f"HEADLINE: {title}\n\n"
                f"LEDE: {lede}"
            )

            out_rec = {
                "id": sample_id,
                "input": sft_input,
                "output": sft_output,
                "source_url": url,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Processed {n_in} raw records, wrote {n_out} training samples to {output_path}")


if __name__ == "__main__":
    main()
