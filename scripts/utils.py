import os
import pathlib
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# -------- CONFIG LOADING --------

DEFAULT_CONFIG_PATH = pathlib.Path("configs/llama31_inference.yaml")


def load_inference_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML config for inference. Falls back to sane defaults if file missing.
    """
    cfg_path = pathlib.Path(path or DEFAULT_CONFIG_PATH)
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Defaults
    data.setdefault("model_id", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    data.setdefault("load_4bit", True)
    data.setdefault("max_ctx", 2048)
    data.setdefault("device_map", "auto")
    data.setdefault("default_mode", "B")

    return data


# -------- MODEL PRESETS --------

MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "A": {
        "label": "A — One-sentence lede (≤20 words)",
        "sys": (
            "You are a neutral news copy editor.\n"
            "Output exactly ONE sentence (≤20 words).\n"
            "Only rephrase what the user provides.\n"
            "Do NOT add dates, numbers, or new facts.\n"
            'If uncertain, output: "Insufficient confirmed information."'
        ),
        "params": {
            "max_new_tokens": 28,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.15,
        },
    },
    "B": {
        "label": "B — Short news brief (2–3 sentences, ≤80 words)",
        "sys": (
            "You are a newsroom editor writing in a neutral, factual tone.\n"
            "RULES:\n"
            "- Use ONLY information explicitly present in the ARTICLE (user text).\n"
            "- Do NOT invent countries, cities, dates, numbers, quotes, or 'first time' claims.\n"
            "- Paraphrasing is allowed, but every concrete fact must come from ARTICLE.\n\n"
            "TASK:\n"
            "Write a concise, neutral news brief in 2–3 sentences (≤80 words)."
        ),
        "params": {
            "max_new_tokens": 100,
            "temperature": 0.25,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.1,
        },
    },
    "C": {
        "label": "C — 3 bullet headlines (5–9 words each)",
        "sys": (
            "Produce exactly three bullet headlines (5–9 words each), neutral and factual.\n"
            "Use ONLY information present in the ARTICLE.\n"
            "Do NOT add new locations, dates, numbers, or background context."
        ),
        "params": {
            "max_new_tokens": 48,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.2,
        },
    },
}


def _bitsandbytes_gpu_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


# -------- MODEL LOADING --------

@lru_cache(maxsize=1)
def load_model_and_tokenizer(
    config_path: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load LLaMA 3.1 8B (or whatever is configured) with optional 4-bit + LoRA.
    Cached so it only loads once per process.
    """
    cfg = load_inference_config(config_path)
    model_id = cfg["model_id"]
    load_4bit = bool(cfg.get("load_4bit", True))
    device_map = cfg.get("device_map", "auto")

    adapters_env = os.environ.get("ADAPTERS", "").strip()
    adapters_path = pathlib.Path(adapters_env) if adapters_env else None

    print(f"[boot] loading model={model_id} 4bit={load_4bit} adapters={adapters_path}")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model_kwargs: Dict[str, Any] = dict(
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    final_4bit = load_4bit and _bitsandbytes_gpu_available()
    if final_4bit:
        print("[boot] enabling 4-bit quantization")
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = qconf
    else:
        print("[boot] running without 4-bit (bitsandbytes/CUDA not available or disabled)")

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )

    model = base
    if adapters_path and adapters_path.is_dir():
        print(f"[boot] attaching LoRA adapters from {adapters_path}")
        model = PeftModel.from_pretrained(base, adapters_path)
    else:
        if adapters_env:
            print(f"[boot] ADAPTERS set but path not found: {adapters_path}, using base model only")

    model.eval()
    return model, tok


# -------- GENERATION LOGIC --------

def _build_chat_ids(
    mode: str,
    user_text: str,
    tok: "AutoTokenizer",
    max_ctx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """
    Build chat-style input IDs using the model's chat template and truncate to max_ctx.
    """
    m = MODE_PRESETS.get(mode.upper(), MODE_PRESETS["A"])
    sys_prompt = m["sys"]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text},
    ]

    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Truncate to max_ctx from the left if needed
    if ids.shape[1] > max_ctx:
        ids = ids[:, -max_ctx:]

    ids = ids.to(device)
    return ids, ids.shape[1]


def _postprocess_output(mode: str, text: str, user_text: str) -> str:
    """
    Apply simple, deterministic post-processing per mode for length & shape.
    More advanced anti-hallucination can be added later here.
    """
    text = text.strip()

    if mode == "A":
        # Single sentence, <= 20 words
        first = text.split("\n", 1)[0].strip()
        for sep in [". ", "? ", "! "]:
            if sep in first:
                first = first.split(sep, 1)[0] + "."
                break
        words = first.split()
        if len(words) > 20:
            first = " ".join(words[:20]).rstrip(" .") + "."
        return first

    if mode == "B":
        # 2–3 sentences, <= 80 words
        body = text.replace("\n", " ").strip()
        parts = [p.strip() for p in body.split(".") if p.strip()]
        body = ". ".join(parts[:3]) + ("." if parts else "")
        words = body.split()
        if len(words) > 80:
            body = " ".join(words[:80]).rstrip(" .") + "."
        return body

    if mode == "C":
        # Exactly 3 bullet headlines (5–9 words)
        lines = [l.strip("-• \t") for l in text.splitlines() if l.strip()]
        bullets = []
        for l in lines:
            w = l.split()
            if 5 <= len(w) <= 9:
                bullets.append("• " + l)
            if len(bullets) == 3:
                break

        if len(bullets) < 3:
            seed = " ".join(user_text.split()[:7]) or "Update pending"
            bullets = ["• " + seed] * 3

        return "\n".join(bullets)

    return text


def generate_text(
    prompt: str,
    mode: str = "B",
    config_path: Optional[str] = None,
    override_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    High-level helper to generate newsroom-style text using configured model.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Empty prompt")

    cfg = load_inference_config(config_path)
    max_ctx = int(cfg.get("max_ctx", 2048))

    model, tok = load_model_and_tokenizer(config_path)
    mode = (mode or cfg.get("default_mode", "B")).upper()
    preset = MODE_PRESETS.get(mode, MODE_PRESETS["B"])
    base_params = preset["params"].copy()

    if override_params:
        base_params.update(override_params)

    # Build chat-style input
    ids, input_len = _build_chat_ids(
        mode=mode,
        user_text=prompt,
        tok=tok,
        max_ctx=max_ctx,
        device=model.device,
    )

    gen_kwargs = dict(
        max_new_tokens=int(base_params["max_new_tokens"]),
        do_sample=(base_params["temperature"] > 0),
        temperature=float(base_params["temperature"]),
        top_p=float(base_params["top_p"]),
        top_k=int(base_params["top_k"]),
        repetition_penalty=float(base_params["repetition_penalty"]),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        no_repeat_ngram_size=3,
    )

    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            **gen_kwargs,
        )

    gen_ids = out[0, input_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return _postprocess_output(mode, text, prompt)
