import argparse
import json
import os
from datetime import datetime

from utils import generate_text


def iter_eval_items(path: str):
    """Yield JSON objects from a JSONL file, skipping blank/bad lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[eval] skipping bad line: {line[:80]}...")
                continue
            yield obj


def main():
    parser = argparse.ArgumentParser(description="Batch eval for newsroom style.")
    parser.add_argument(
        "--eval-path",
        default="data/eval/newsroom_eval_set.jsonl",
        help="Path to JSONL with eval items.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Where to write JSONL with model outputs (default auto from MODEL_VERSION).",
    )
    args = parser.parse_args()

    eval_path = args.eval_path

    # Decide output path
    if args.out_path:
        out_path = args.out_path
    else:
        model_version = os.environ.get("MODEL_VERSION", "llama31_unknown")
        out_path = f"data/eval/results_{model_version}.jsonl"

    print(f"[eval] loading eval set from: {eval_path}")
    print(f"[eval] writing outputs to: {out_path}")

    # Resolve config path relative to repo root
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "configs", "llama31_inference.yaml")
    )
    print(f"[eval] using config: {config_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    count = 0
    model_version = os.environ.get("MODEL_VERSION", "llama31_unknown")

    with open(out_path, "w", encoding="utf-8") as out_f:
        for item in iter_eval_items(eval_path):
            # Basic fields with fallbacks
            id_ = item.get("id", f"case_{count + 1:03d}")
            mode = item.get("mode", "B")

            # 🔴 IMPORTANT: support both "article" and "prompt"
            article = item.get("article") or item.get("prompt")
            notes = item.get("notes", "")

            if not article:
                print(f"[eval] skipping {id_}: no 'article' or 'prompt' field")
                continue

            print(f"[case {count + 1}] id={id_} mode={mode}")

            # We let utils.generate_text handle the system prompt based on mode
            text = generate_text(
                prompt=article,
                mode=mode,
                config_path=config_path,
            )

            result = {
                "id": id_,
                "mode": mode,
                "article": article,
                "system_prompt_mode": mode,
                "notes": notes,
                "output": text,
                "model_version": model_version,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            count += 1

    print(f"[eval] completed {count} generations.")


if __name__ == "__main__":
    main()
