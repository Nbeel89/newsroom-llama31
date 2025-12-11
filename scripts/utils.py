import re
import os
import pathlib
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional
import contextlib

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# CONFIG LOADING
# ============================================================

DEFAULT_CONFIG_PATH = pathlib.Path("configs/llama31_inference.yaml")


def load_inference_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML config for inference. Falls back to sane defaults if file missing.

    YAML can define:
      - base_model_id: huggingface model name
      - adapters_path: LoRA checkpoint directory
      - load_4bit: true/false
      - max_ctx: context window
      - system_prompt_editorial: master system prompt
      - mode_prompts: {A: "...{article}...", B: "...", C: "..."}
    """
    cfg_path = pathlib.Path(path or DEFAULT_CONFIG_PATH)
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Support either `base_model_id` or `model_id` in YAML, but prefer explicit.
    base_model_id = data.get("base_model_id") or data.get(
        "model_id", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    data["model_id"] = base_model_id

    # Defaults (only used if YAML does not define them)
    data.setdefault("load_4bit", True)
    data.setdefault("max_ctx", 2048)
    data.setdefault("device_map", "auto")
    data.setdefault("default_mode", "B")
    # Default LoRA adapter directory (your fine-tuned weights)
    data.setdefault("adapters_path", "outputs/llama31_lora_v2/checkpoint-200")

    return data


# ============================================================
# MODE PRESETS (fallbacks if YAML prompts not present)
# ============================================================

MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "A": {
        "label": "A — One-sentence lede (≤20 words)",
        "sys": (
            "You are a neutral news editor. "
            "Write one concise sentence (maximum 20 words) based strictly on the ARTICLE. "
            "Do not add any facts that are not explicitly stated in the ARTICLE."
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
            "You are a newsroom writer. "
            "Write a neutral 2–3 sentence news brief (maximum 80 words) "
            "based only on what the user writes. "
            "Do not invent extra facts, names, locations, dates or numbers."
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
            "You are a neutral news editor. "
            "From the ARTICLE, write three different concise bullet headlines, "
            "each 5–9 words. Use only facts from the ARTICLE. "
            "Do not repeat the same wording across bullets and do not add new facts."
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


# ============================================================
# MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_model_and_tokenizer(
    config_path: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load LLaMA 3.1 8B with optional 4-bit + LoRA.
    Cached so it only loads once per process.
    """
    cfg = load_inference_config(config_path)
    model_id = cfg["model_id"]
    load_4bit = bool(cfg.get("load_4bit", True))

    # Prefer adapters_path from config; fall back to ADAPTERS env var if set.
    adapters_cfg = cfg.get("adapters_path", "")
    adapters_env = os.environ.get("ADAPTERS", "").strip()

    adapters_path: Optional[pathlib.Path] = None
    if adapters_cfg:
        adapters_path = pathlib.Path(adapters_cfg)
    elif adapters_env:
        adapters_path = pathlib.Path(adapters_env)

    print(
        f"[boot] loading {model_id} "
        f"(4bit={load_4bit}) adapters={adapters_path}"
    )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and load_4bit:
        print("[boot] loading base model in 4-bit on CUDA")
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            load_in_4bit=True,
        )
    else:
        print("[boot] loading base model in full precision on", device)
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        base.to(device)

    model = base
    if adapters_path and adapters_path.is_dir():
        print(f"[boot] attaching LoRA from {adapters_path}")
        model = PeftModel.from_pretrained(base, adapters_path)
    else:
        if adapters_cfg or adapters_env:
            print(
                f"[boot] WARNING: adapters path not found: {adapters_path}, "
                f"using base model only"
            )

    model.eval()
    return model, tok


# ============================================================
# PROMPT BUILDERS
# ============================================================

def _build_chat_ids(
    mode: str,
    user_text: str,
    tok: "AutoTokenizer",
    max_ctx: int,
    device: torch.device,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build chat-style input IDs using the model's chat template.
    Uses YAML editorial prompts if present, otherwise falls back to MODE_PRESETS.
    """
    mode_key = (mode or "A").upper()
    article = (user_text or "").strip()

    sys_prompt: Optional[str] = None
    user_content: str = article

    if cfg is not None:
        editorial_sys = (cfg.get("system_prompt_editorial") or "").strip()
        mode_prompts = cfg.get("mode_prompts") or {}
        tpl = mode_prompts.get(mode_key)

        if editorial_sys and tpl:
            sys_prompt = editorial_sys
            # SAFE replacement: no .format() so '%' and other chars can't break it.
            try:
                user_content = tpl.replace("{article}", article)
            except Exception:
                user_content = tpl + "\n\nARTICLE:\n" + article

    # Fallback if YAML prompts not available
    if not sys_prompt:
        m = MODE_PRESETS.get(mode_key, MODE_PRESETS["A"])
        sys_prompt = m["sys"].strip()
        user_content = article

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    if tok.pad_token_id is not None:
        attention_mask = (ids != tok.pad_token_id).long()
    else:
        attention_mask = torch.ones_like(ids)

    if ids.shape[1] > max_ctx:
        ids = ids[:, -max_ctx:]
        attention_mask = attention_mask[:, -max_ctx:]

    input_len = ids.shape[1]
    return ids, attention_mask, input_len


def _build_raw_ids(
    user_text: str,
    tok: "AutoTokenizer",
    max_ctx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build raw token IDs without chat roles (for natural continuation).
    Used only by generate_natural_text for debugging.
    """
    article = user_text.strip()
    enc = tok(
        article,
        return_tensors="pt",
        add_special_tokens=False,
    )
    ids = enc["input_ids"].to(device)

    if ids.shape[1] > max_ctx:
        ids = ids[:, -max_ctx:]

    if tok.pad_token_id is not None:
        attention_mask = (ids != tok.pad_token_id).long()
    else:
        attention_mask = torch.ones_like(ids)

    input_len = ids.shape[1]
    return ids, attention_mask, input_len


# ============================================================
# CLEANUP HELPERS
# ============================================================

def _strip_preamble(text: str) -> str:
    """
    Remove obvious instruction-style preambles the model sometimes echoes.
    """
    text = text.strip()
    if not text:
        return text

    text = re.sub(
        r"^[^A-Za-z]*today date:[^.]*\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    leak_phrases = [
        "write a neutral 2–3 sentence news brief",
        "maximum 80 words) based only on what the user writes",
        "do not invent extra facts, names, locations, dates or numbers",
        "you are a newsroom writer",
        "you are a neutral news editor",
        "write one concise sentence (maximum 20 words)",
        "write exactly three bullet-point headlines",
    ]
    low = text.lower()
    cut_idx = 0
    for phrase in leak_phrases:
        idx = low.find(phrase.lower())
        if idx != -1:
            end = low.find(".", idx)
            if end != -1:
                cut_idx = max(cut_idx, end + 1)

    if cut_idx:
        text = text[cut_idx:].lstrip()

    text = re.sub(r"assistant\.?\s*$", "", text, flags=re.IGNORECASE).strip()
    return " ".join(text.split()).strip()


def _remove_labels(text: str) -> str:
    """
    Remove technical labels like HEADLINE:/LEDE:/BULLETS: if the model emits them.
    """
    lines = []
    pattern = re.compile(
        r"^\s*(HEADLINE|LEDE|BULLETS?)\s*[:：]\s*",
        flags=re.IGNORECASE,
    )

    for raw in text.splitlines():
        line = raw.rstrip()
        cleaned = pattern.sub("", line)
        if cleaned.strip():
            lines.append(cleaned)

    return "\n".join(lines).strip()


# ============================================================
# NATURAL (RAW) GENERATION – DEBUG ONLY
# ============================================================

def generate_natural_text(
    prompt: str,
    config_path: Optional[str] = None,
    max_new_tokens: int = 160,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.05,
) -> str:
    """
    Natural continuation (no system roles) – mainly for debugging.

    Use this when you want to inspect how the base+LoRA model behaves
    WITHOUT mode prompts or post-processing.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Empty prompt")

    cfg = load_inference_config(config_path)
    max_ctx = int(cfg.get("max_ctx", 2048))

    model, tok = load_model_and_tokenizer(config_path)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ids, attention_mask, input_len = _build_raw_ids(
        user_text=prompt,
        tok=tok,
        max_ctx=max_ctx,
        device=device,
    )

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=(temperature > 0),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        no_repeat_ngram_size=3,
    )

    with torch.inference_mode():
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if use_cuda
            else contextlib.nullcontext()
        )
        with amp_ctx:
            out = model.generate(
                input_ids=ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

    gen_ids = out[0, input_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    if not text:
        full = tok.decode(out[0], skip_special_tokens=True)
        text = full[-400:].strip()

    return text


# ============================================================
# LIGHT POST-PROCESSING (NO REWRITES)
# ============================================================

def _postprocess_output(mode: str, text: str, user_text: str) -> str:
    """
    Final shaping / cleanup of model output for modes A, B, C.

    - Never reconstructs from ARTICLE.
    - Only removes obvious junk / labels and lightly enforces shape.
    """
    mode = (mode or "B").upper()
    raw_text = (text or "").strip()
    article = (user_text or "").strip()

    cleaned = _strip_preamble(raw_text)
    cleaned = _remove_labels(cleaned)

    if not cleaned:
        cleaned = raw_text

    cleaned = cleaned.strip()
    if not cleaned:
        return article or ""

    # Mode A: lede (single sentence, ~≤20 words)
    if mode == "A":
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        first = sentences[0] if sentences else cleaned

        words = first.split()
        if len(words) > 20:
            first = " ".join(words[:20]) + "…"
        if not first.endswith((".", "!", "?")):
            first += "."
        return first

    # Mode B: brief – keep as-is, ensure it ends with punctuation
    if mode == "B":
        brief = cleaned
        if len(brief.split()) < 4:
            return brief
        if not brief.endswith((".", "!", "?")):
            brief += "."
        return brief

    # Mode C: bullet headlines – keep model bullets, normalize format
    if mode == "C":
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]

        if len(lines) == 1 and ("•" in lines[0] or "-" in lines[0]):
            parts = re.split(r"[•\-]\s*", lines[0])
            lines = [p.strip() for p in parts if p.strip()]

        bullets = []
        for ln in lines:
            ln = ln.lstrip("-•*").strip()
            if not ln:
                continue
            bullets.append(ln)

        if not bullets:
            return cleaned

        final = []
        for b in bullets:
            words = b.split()
            if len(words) > 12:
                b = " ".join(words[:12]) + "…"
            final.append(f"• {b}")

        return "\n".join(final)

    return cleaned


# ============================================================
# MAIN GENERATION ENTRYPOINT
# ============================================================

def generate_text(
    prompt: str,
    mode: str = "B",
    config_path: Optional[str] = None,
    override_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    High-level helper to generate newsroom-style text using configured model.

    To tweak without touching code:
      - Edit system_prompt_editorial + mode_prompts in YAML.
      - Or edit MODE_PRESETS for fallback behaviour.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Empty prompt")

    cfg = load_inference_config(config_path)
    max_ctx = int(cfg.get("max_ctx", 2048))

    model, tok = load_model_and_tokenizer(config_path)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mode = (mode or cfg.get("default_mode", "B")).upper()
    preset = MODE_PRESETS.get(mode, MODE_PRESETS["B"])
    base_params = preset["params"].copy()

    if override_params:
        base_params.update(override_params)

    ids, attention_mask, input_len = _build_chat_ids(
        mode=mode,
        user_text=prompt,
        tok=tok,
        max_ctx=max_ctx,
        device=device,
        cfg=cfg,
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
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if use_cuda
            else contextlib.nullcontext()
        )
        with amp_ctx:
            out = model.generate(
                input_ids=ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

    gen_ids = out[0, input_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    if not text:
        full = tok.decode(out[0], skip_special_tokens=True)
        text = full[-400:].strip()

    return _postprocess_output(mode, text, prompt)
