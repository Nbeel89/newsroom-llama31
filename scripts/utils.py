import os
import re
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
    Load YAML config for inference and provide sane defaults.

    Supports both:
      - model_id
      - base_model_id   (as in your current YAML)
    """
    cfg_path = pathlib.Path(path or DEFAULT_CONFIG_PATH)

    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Base model id (prefer explicit key if present).
    model_id = (
        data.get("model_id")
        or data.get("base_model_id")
        or "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    data["model_id"] = model_id

    # Quantization / device defaults.
    data.setdefault("load_4bit", True)
    data.setdefault("device_map", "auto")

    # Context length and default mode.
    data.setdefault("max_ctx", 2048)
    data.setdefault("default_mode", "B")

    # LoRA adapter path (fine-tuned weights).
    data.setdefault("adapters_path", "outputs/llama31_lora_v2/checkpoint-200")

    return data


# ============================================================
# MODE PRESETS (UI + decoding params)
# ============================================================

MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "A": {
        "label": "A — One-sentence lede (≤20 words)",
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
    Load LLaMA 3.1 8B with optional 4-bit quantization and LoRA adapters.

    Cached so we only load once per process.
    """
    cfg = load_inference_config(config_path)
    model_id = cfg["model_id"]
    load_4bit = bool(cfg.get("load_4bit", True))

    adapters_cfg = cfg.get("adapters_path", "")
    adapters_env = os.environ.get("ADAPTERS", "").strip()

    adapters_path: Optional[pathlib.Path] = None
    if adapters_cfg:
        adapters_path = pathlib.Path(adapters_cfg)
    elif adapters_env:
        adapters_path = pathlib.Path(adapters_env)

    print(
        f"[boot] loading {model_id} (4bit={load_4bit}) adapters={adapters_path}"
    )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Base model load
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
# CHAT / RAW INPUT BUILDING
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

    If YAML contains:
      - system_prompt_editorial
      - mode_prompts[mode]

    we use those. Otherwise we just use the ARTICLE as a plain user prompt.
    """
    mode_key = (mode or "B").upper()
    article = (user_text or "").strip()

    sys_prompt: str = ""
    user_content: str = article

    if cfg is None:
        cfg = {}

    editorial_sys = (cfg.get("system_prompt_editorial") or "").strip()
    mode_prompts = cfg.get("mode_prompts") or {}
    tpl = mode_prompts.get(mode_key)

    if editorial_sys and tpl:
        sys_prompt = editorial_sys
        try:
            user_content = tpl.format(article=article)
        except Exception:
            # Fail-safe if formatting breaks.
            user_content = f"{tpl}\n\nARTICLE:\n{article}"
    else:
        # Very light generic system prompt, if nothing is configured.
        if mode_key == "A":
            sys_prompt = (
                "You are a neutral news editor. "
                "Write one concise lede sentence based only on ARTICLE."
            )
        elif mode_key == "B":
            sys_prompt = (
                "You are a neutral news writer. "
                "Write a 2–3 sentence factual brief based only on ARTICLE."
            )
        else:
            sys_prompt = (
                "You are a neutral news editor. "
                "Write three concise factual bullet headlines from ARTICLE."
            )

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

    # Attention mask
    if tok.pad_token_id is not None:
        attention_mask = (ids != tok.pad_token_id).long()
    else:
        attention_mask = torch.ones_like(ids)

    # Context truncation
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
    Raw token IDs without roles. Used only by generate_natural_text.
    """
    article = (user_text or "").strip()
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
# PRE / POST-PROCESSING
# ============================================================

def _strip_preamble(text: str) -> str:
    """
    Remove obvious instruction-style preambles the model sometimes echoes.
    We keep this deliberately conservative.
    """
    t = (text or "").strip()
    if not t:
        return t

    # Drop "Today Date: ..." if it appears.
    t = re.sub(
        r"^[^A-Za-z]*today date:[^.]*\.\s*",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()

    # Remove fragments of our own instructions if echoed.
    leak_phrases = [
        "you are a newsroom writer",
        "you are a neutral news editor",
        "write a neutral 2–3 sentence news brief",
        "do not invent extra facts",
    ]
    low = t.lower()
    cut_idx = 0
    for phrase in leak_phrases:
        idx = low.find(phrase)
        if idx != -1:
            end = low.find(".", idx)
            if end != -1:
                cut_idx = max(cut_idx, end + 1)

    if cut_idx:
        t = t[cut_idx:].lstrip()

    return t.strip()


def _remove_labels(text: str) -> str:
    """
    Remove technical labels like HEADLINE:, LEDE:, BULLETS: that the prompts
    ask the model to include. We keep only the actual content.
    """
    if not text:
        return text

    replacements = [
        (r"\bHEADLINE\s*[:：]\s*", ""),
        (r"\bLEDE\s*[:：]\s*", ""),
        (r"\bBULLETS?\s*[:：]\s*", ""),
    ]
    result = text
    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result


def _quick_brief_from_article(article: str, max_sentences: int = 3) -> str:
    """
    Fallback: build a 2–3 sentence brief directly from ARTICLE text.
    """
    s = (article or "").strip()
    if not s:
        return "Update pending."

    sentences = re.split(r"(?<=[.!?])\s+", s)
    sentences = [x.strip() for x in sentences if x.strip()]
    if not sentences:
        return "Update pending."

    brief = " ".join(sentences[:max_sentences])

    # Ensure final punctuation.
    if not brief.endswith((".", "!", "?")):
        brief += "."

    return brief


def _postprocess_output(mode: str, text: str, user_text: str) -> str:
    """
    Final shaping / cleanup of model output for modes A, B, C.
    """
    mode = (mode or "B").upper()
    raw = (text or "").strip()
    article = (user_text or "").strip()

    # 1) Light generic cleanup.
    t = _strip_preamble(raw)
    t = _remove_labels(t)

    # Remove simple "User:" / "Assistant:" prefixes at start of lines.
    t = re.sub(
        r"(?mi)^(User|Assistant)\s*[:>]\s*",
        "",
        t,
    ).strip()

    if not t:
        # Completely cleaned away -> fall back to ARTICLE.
        t = article

    # Helper for sentence splitting.
    def _sentences(s: str):
        parts = re.split(r"(?<=[.!?])\s+", s)
        return [x.strip() for x in parts if x.strip()]

    # ---------- Mode A: one-sentence lede ----------
    if mode == "A":
        sents = _sentences(t)
        if not sents:
            sents = _sentences(article)

        if not sents:
            return "Update pending."

        first = sents[0]
        words = first.split()
        if len(words) > 20:
            first = " ".join(words[:20]) + "…"
        if not first.endswith((".", "!", "?")):
            first += "."
        return first

    # ---------- Mode B: 2–3 sentence brief ----------
    if mode == "B":
        # If model output is extremely short or degenerate (like "LEDE."),
        # ignore it and build directly from the ARTICLE.
        normalized = t.lower().strip()
        if (
            len(t.split()) < 6
            or normalized in {"lede", "lede.", "headline", "headline."}
            or normalized.startswith("lede ")
        ):
            return _quick_brief_from_article(article, max_sentences=3)

        # Otherwise, try to build a brief from model output.
        sents = _sentences(t)
        if not sents:
            return _quick_brief_from_article(article, max_sentences=3)

        brief = " ".join(sents[:3])
        if not brief.endswith((".", "!", "?")):
            brief += "."
        return brief

    # ---------- Mode C: three bullet headlines ----------
    if mode == "C":
        # Collect candidate lines from model output.
        lines = []
        for line in t.splitlines():
            line = line.strip()
            if not line:
                continue
            # strip bullet symbols
            line = line.lstrip("-•*").strip()
            if len(line.split()) < 3:
                continue
            lines.append(line)

        # If we didn't get good lines from the output, fall back to article.
        if not lines:
            article_sents = _sentences(article)
            for s in article_sents[:3]:
                lines.append(s)

        # Deduplicate while preserving order.
        seen = set()
        uniq = []
        for l in lines:
            key = " ".join(l.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(l)

        def trim_headline(h: str, min_w: int = 5, max_w: int = 9) -> str:
            words = h.split()
            if len(words) > max_w:
                return " ".join(words[:max_w]) + "…"
            if len(words) < min_w:
                return h
            return h

        bullets = []
        for h in uniq:
            bullets.append(trim_headline(h))
            if len(bullets) == 3:
                break

        if not bullets:
            bullets = ["Update pending"] * 3
        while len(bullets) < 3:
            bullets.append(bullets[-1])

        return "\n".join(f"• {b}" for b in bullets)

    # ---------- Fallback ----------
    return t


# ============================================================
# NATURAL CONTINUATION (DEBUG)
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
# TOP-LEVEL GENERATION ENTRY POINT
# ============================================================

def generate_text(
    prompt: str,
    mode: str = "B",
    config_path: Optional[str] = None,
    override_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    High-level helper used by the Flask console.

    - Reads YAML config
    - Uses MODE_PRESETS for decoding params (overridable from UI)
    - Builds chat-style prompt with editorial system + mode_prompts
    - Applies light post-processing and strong fallbacks.
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

    # Allow per-call overrides (UI sliders).
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
    raw_text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    if not raw_text:
        full = tok.decode(out[0], skip_special_tokens=True)
        raw_text = full[-400:].strip()

    final_text = _postprocess_output(mode, raw_text, prompt)

    return final_text
