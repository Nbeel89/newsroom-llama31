# scripts/10_prepare_eval_sheet.py

import json
import csv
from pathlib import Path

EVAL_PATH = Path("data/eval/newsroom_eval_set.jsonl")
RESULTS_PATH = Path("data/eval/results_llama31_baseline.jsonl")
OUT_CSV = Path("data/eval/manual_eval_llama31_baseline.csv")


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    print(f"[sheet] loading eval set from: {EVAL_PATH}")
    eval_items = {item["id"]: item for item in load_jsonl(EVAL_PATH)}

    print(f"[sheet] loading results from: {RESULTS_PATH}")
    results = load_jsonl(RESULTS_PATH)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "mode",
        "model_version",
        "article_snippet",
        "generation",
        # scores you will fill manually (1–5 or 1–10)
        "tone_score",
        "structure_score",
        "faithfulness_score",
        "overall_score",
        "notes",
    ]

    print(f"[sheet] writing CSV to: {OUT_CSV}")
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            _id = r.get("id")
            mode = r.get("mode")
            model_version = r.get("model_version", "llama31_baseline")
            gen = r.get("generation", "")

            eval_item = eval_items.get(_id, {})
            prompt = eval_item.get("prompt", "")
            snippet = prompt[:400]  # to keep sheet readable

            writer.writerow({
                "id": _id,
                "mode": mode,
                "model_version": model_version,
                "article_snippet": snippet,
                "generation": gen,
                "tone_score": "",
                "structure_score": "",
                "faithfulness_score": "",
                "overall_score": "",
                "notes": "",
            })

    print("[sheet] done. Open the CSV in Excel and fill in scores.")


if __name__ == "__main__":
    main()
