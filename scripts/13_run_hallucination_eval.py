import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_eval_items(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # minimal validation
            if "id" not in rec or "prompt" not in rec:
                raise ValueError(f"Record missing id or prompt: {rec}")
            items.append(rec)
    return items


def generate_batch(model, tokenizer, items, device: str, max_new_tokens: int = 160):
    """
    Greedy generation (no sampling) for deterministic comparison.
    Returns list of dicts: {"id", "model", "output"}.
    """
    results = []

    for rec in tqdm(items, desc="Generating", unit="item"):
        prompt = rec["prompt"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for stability
            )

        # Slice off the prompt to get only the completion
        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text_out = tokenizer.decode(gen_ids, skip_special_tokens=True)

        results.append(
            {
                "id": rec["id"],
                "output": text_out.strip(),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/eval/hallucination_eval_v1.jsonl",
        help="Path to JSONL eval set",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/eval/hallucination_eval_results_v1.csv",
        help="Where to write model outputs",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        default="outputs/llama31_lora_v2/checkpoint-200",
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
        help="Max new tokens to generate",
    )

    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    out_path = Path(args.output_csv)

    print(f"[eval] Loading eval items from {eval_path}")
    items = load_eval_items(eval_path)
    print(f"[eval] Loaded {len(items)} records")

    print(f"[eval] Using device: {args.device}")

    # ------------------------------------------------------------------
    # 1) BASE MODEL RUN
    # ------------------------------------------------------------------
    print("\n[eval] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[eval] Loading BASE model in 4-bit (no device_map)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        load_in_4bit=True,  # same style as training
        device_map=None,    # force everything on single GPU
    )
    base_model.eval()

    base_results = generate_batch(
        base_model,
        tokenizer,
        items,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    # Tag model name
    for r in base_results:
        r["model"] = "base"

    # Free GPU before loading LoRA
    print("[eval] Freeing BASE model from GPU memory...")
    del base_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2) LoRA MODEL RUN
    # ------------------------------------------------------------------
    print("\n[eval] Loading BASE model again for LoRA...")
    lora_base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        load_in_4bit=True,
        device_map=None,
    )
    print(f"[eval] Loading LoRA adapter from: {args.lora_checkpoint}")
    lora_model = PeftModel.from_pretrained(
        lora_base,
        args.lora_checkpoint,
    )
    lora_model.eval()

    lora_results = generate_batch(
        lora_model,
        tokenizer,
        items,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    for r in lora_results:
        r["model"] = "lora_v2"

    # ------------------------------------------------------------------
    # 3) WRITE CSV
    # ------------------------------------------------------------------
    print(f"\n[eval] Writing outputs to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "model", "output"])
        for r in base_results + lora_results:
            writer.writerow([r["id"], r["model"], r["output"]])

    print("[eval] Done.")
    print(f"[eval] Rows written: {len(base_results) + len(lora_results)}")
    print("[eval] Next step: open the CSV in Excel and manually mark hallucinations per row.")


if __name__ == "__main__":
    main()
