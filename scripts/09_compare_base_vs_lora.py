import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def get_system_prompt(mode: str) -> str:
    """
    A/B/C prompt templates.
    Adjust wording if you want it to match your original prompts more closely.
    """
    mode = (mode or "A").upper()

    if mode == "A":
        return (
            "You are an Al Jazeera–style news editor. "
            "Write a concise, single-paragraph summary in a neutral, factual tone, "
            "using only the information in the article."
        )
    elif mode == "B":
        return (
            "You are an Al Jazeera–style editor. "
            "Given an article, produce a HEADLINE and a LEDE (no more than 40 words) "
            "in a neutral, factual tone. Use only the information in the article."
        )
    elif mode == "C":
        return (
            "You are an Al Jazeera–style editor. "
            "Given an article, produce a HEADLINE, a LEDE (no more than 40 words), "
            "and exactly three factual bullet points. Use only the information in the article."
        )
    else:
        return (
            "You are a neutral news editor. Summarize the article factually in a neutral tone."
        )


def format_prompt(article: str, mode: str) -> str:
    """
    Build a plain-text prompt for LLaMA 3.1.
    If you later want, you can replace this with tokenizer.apply_chat_template.
    """
    system_prompt = get_system_prompt(mode)
    user_task = (
        "TASK: Follow the instructions and write the requested output.\n\n"
        f"ARTICLE:\n{article.strip()}"
    )

    prompt = (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_task}\n"
        f"<|assistant|>\n"
    )
    return prompt


def load_baseline_results(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model ID (HF hub or local path).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="outputs/llama31_lora_v1",
        help="Path to LoRA adapter directory.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="data/eval/results_llama31_baseline.jsonl",
        help="Path to baseline results JSONL file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/eval/comparison_llama31_base_vs_lora.csv",
        help="Where to write comparison CSV.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device, e.g. 'cuda' or 'cpu'.",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[compare] Loading baseline results from {baseline_path}")
    baseline_records = load_baseline_results(baseline_path)
    print(f"[compare] Found {len(baseline_records)} baseline records")

    print(f"[compare] Loading tokenizer: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"[compare] Loading base model on device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.to(device)

    print(f"[compare] Loading LoRA adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    timestamp = datetime.utcnow().isoformat() + "Z"

    # Define CSV columns (including placeholders for your manual scores)
    fieldnames = [
        "id",
        "mode",
        "article_snippet",
        "baseline_output",
        "lora_output",
        "model_version_base",
        "model_version_lora",
        "tone_base",
        "tone_lora",
        "structure_base",
        "structure_lora",
        "faithfulness_base",
        "faithfulness_lora",
        "overall_base",
        "overall_lora",
        "notes",
        "generated_at",
    ]

    print(f"[compare] Writing comparison CSV to {out_path}")
    with out_path.open("w", encoding="utf-8", newline="") as f_csv, torch.no_grad():
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for rec in tqdm(baseline_records, desc="Generating LoRA outputs"):
            rec_id = rec.get("id")
            mode = rec.get("mode") or rec.get("system_prompt_mode") or "A"
            article = rec.get("article") or rec.get("prompt", "")
            baseline_output = rec.get("output", "")
            model_version_base = rec.get("model_version", "llama31_baseline")

            prompt = format_prompt(article, mode)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            generated = gen_ids[0, inputs["input_ids"].shape[1]:]
            lora_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

            row = {
                "id": rec_id,
                "mode": mode,
                "article_snippet": (article[:180] + "…") if len(article) > 180 else article,
                "baseline_output": baseline_output,
                "lora_output": lora_text,
                "model_version_base": model_version_base,
                "model_version_lora": "llama31_lora_v1",
                # Manual scoring fields left blank for you to fill in Excel:
                "tone_base": "",
                "tone_lora": "",
                "structure_base": "",
                "structure_lora": "",
                "faithfulness_base": "",
                "faithfulness_lora": "",
                "overall_base": "",
                "overall_lora": "",
                "notes": "",
                "generated_at": timestamp,
            }

            writer.writerow(row)

    print("[compare] Done.")


if __name__ == "__main__":
    main()
