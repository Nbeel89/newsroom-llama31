import argparse
import pathlib
import textwrap
import yaml
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


DEFAULT_CONFIG_PATH = pathlib.Path("configs/llama31_inference.yaml")


def load_inference_config(path: str | None = None) -> dict:
    cfg_path = pathlib.Path(path or DEFAULT_CONFIG_PATH)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system_and_user_prompt(mode: str, article: str) -> tuple[str, str]:
    """
    Mode here is ONLY for debugging prompts, independent from your Flask console.
    You can see clearly what each one asks the model to do.
    """

    # --- SYSTEM PROMPTS (simple, explicit, not hidden) ---
    if mode == "none":
        system = ""
        user = f"ARTICLE:\n{article}\n\nWrite any short response based only on the article."
    elif mode == "simple":
        system = (
            "You are an Al Jazeera–style news editor. "
            "Use ONLY the facts in the ARTICLE. Do not add outside context."
        )
        user = f"ARTICLE:\n{article}\n\nSummarise this in one or two neutral news sentences."
    elif mode == "lede":
        system = (
            "You are an Al Jazeera–style news editor. "
            "Write a single lede sentence (max 20 words) that summarises the key news."
        )
        user = f"ARTICLE:\n{article}\n\nTASK: Produce ONLY the lede sentence."
    elif mode == "brief":
        system = (
            "You are an Al Jazeera–style news editor. "
            "Write a short news brief: 2–3 sentences, maximum 80 words, neutral tone."
        )
        user = f"ARTICLE:\n{article}\n\nTASK: Produce ONLY the brief."
    elif mode == "headlines3":
        system = (
            "You are an Al Jazeera–style news editor. "
            "Write EXACTLY three bullet-point headlines. "
            "Each headline must be between 5 and 9 words. "
            "Do NOT copy sentences from the ARTICLE. "
            "Do NOT repeat the same wording."
        )
        user = f"ARTICLE:\n{article}\n\nTASK: Produce ONLY the three bullet headlines."
    else:
        raise ValueError(f"Unknown debug mode: {mode}")

    return system, user


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Use LLaMA chat template so we see the real text sent to the model."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def load_model_and_tokenizer(cfg: dict, use_lora: bool) -> tuple:
    model_id = cfg["base_model_id"]
    adapters_path = cfg.get("adapters_path")

    print(f"[load] base_model_id = {model_id}")
    print(f"[load] use_lora      = {use_lora}")
    if use_lora:
        print(f"[load] adapters_path = {adapters_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if use_lora:
        if not adapters_path:
            raise ValueError("No adapters_path set in config but use_lora=True")
        model = PeftModel.from_pretrained(base_model, adapters_path)
    else:
        model = base_model

    model.eval()
    return tokenizer, model


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float,
             top_p: float, repetition_penalty: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )

    # Drop the prompt part
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Debug single LLaMA / LoRA inference.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--mode", default="simple",
                        choices=["none", "simple", "lede", "brief", "headlines3"])
    parser.add_argument("--use-lora", action="store_true",
                        help="If set, load LoRA adapters from config.")
    parser.add_argument("--article", type=str, help="Article text inline.")
    parser.add_argument("--article-file", type=str, help="Path to .txt file with article.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)

    args = parser.parse_args()

    if not args.article and not args.article_file:
        raise SystemExit("Provide --article or --article-file")

    if args.article_file:
        article = pathlib.Path(args.article_file).read_text(encoding="utf-8").strip()
    else:
        article = args.article.strip()

    cfg = load_inference_config(args.config)
    tokenizer, model = load_model_and_tokenizer(cfg, use_lora=args.use_lora)

    system, user = build_system_and_user_prompt(args.mode, article)
    prompt = build_chat_prompt(tokenizer, system, user)

    print("\n=== SYSTEM PROMPT ===")
    print(textwrap.indent(system or "[NONE]", "  "))

    print("\n=== USER PROMPT ===")
    print(textwrap.indent(user, "  "))

    print("\n=== FULL MODEL INPUT (first 400 chars) ===")
    print(textwrap.indent(prompt[:400], "  "))
    if len(prompt) > 400:
        print("  ... [truncated] ...")

    print("\n=== GENERATION PARAMS ===")
    print(f"  max_new_tokens     = {args.max_new_tokens}")
    print(f"  temperature        = {args.temperature}")
    print(f"  top_p              = {args.top_p}")
    print(f"  repetition_penalty = {args.repetition_penalty}")

    print("\n=== MODEL OUTPUT ===")
    out = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(textwrap.indent(out, "  "))


if __name__ == "__main__":
    main()
