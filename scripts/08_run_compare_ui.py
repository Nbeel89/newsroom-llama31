"""
Flask UI to compare BASE vs LoRA-finetuned Newsroom LLM outputs.

Run with:
    (venv) python scripts/08_run_compare_ui.py

Then open:
    http://127.0.0.1:7861/compare
"""

import pathlib
from functools import lru_cache
from typing import Dict, Any

import torch
import yaml
from flask import Flask, render_template, request, redirect, url_for

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

CONFIG_BASE = ROOT_DIR / "configs" / "llama31_inference_base.yaml"
CONFIG_LORA = ROOT_DIR / "configs" / "llama31_inference.yaml"


# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------

def load_inference_config(path: pathlib.Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Inference config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_base_config() -> Dict[str, Any]:
    return load_inference_config(CONFIG_BASE)


@lru_cache(maxsize=1)
def get_lora_config() -> Dict[str, Any]:
    return load_inference_config(CONFIG_LORA)


# -------------------------------------------------------------------
# TOKENIZER
# -------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_tokenizer():
    cfg = get_base_config()
    model_id = cfg["base_model_id"]
    print(f"[compare-ui] loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# -------------------------------------------------------------------
# MODEL LOADING (simple 4-bit / full precision)
# -------------------------------------------------------------------

def _load_base_model_from_cfg(cfg: Dict[str, Any]):
    model_id = cfg["base_model_id"]
    load_4bit = cfg.get("load_4bit", False)

    print(f"[compare-ui] loading BASE model: {model_id} (4bit={load_4bit})")

    if load_4bit:
        # 4-bit quantized on GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        # Full / half precision on GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model


@lru_cache(maxsize=1)
def get_base_model():
    cfg = get_base_config()
    return _load_base_model_from_cfg(cfg)


@lru_cache(maxsize=1)
def get_lora_model():
    """
    Load a separate base model + LoRA adapter for comparison.

    IMPORTANT: we do NOT pass device_map to PeftModel.from_pretrained
    to avoid meta-tensor issues. The base model is already on CUDA.
    """
    cfg = get_lora_config()

    adapters_rel = cfg.get("adapters_path")
    if not adapters_rel:
        raise ValueError("No adapters_path found in llama31_inference.yaml")

    # Resolve adapters_path relative to project root
    adapters_path = (ROOT_DIR / adapters_rel).resolve()
    if not adapters_path.exists():
        raise FileNotFoundError(f"LoRA adapters_path does not exist: {adapters_path}")

    base_model_id = cfg["base_model_id"]
    load_4bit = cfg.get("load_4bit", False)

    print(f"[compare-ui] loading LoRA BASE model: {base_model_id} (4bit={load_4bit})")
    if load_4bit:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cuda",
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
    base.eval()

    print("[compare-ui] Loading LoRA adapters from:", adapters_path)
    # NOTE: no device_map here; base is already on CUDA
    model = PeftModel.from_pretrained(
        base,
        str(adapters_path),
    )
    model.eval()
    print("[compare-ui] LoRA model loaded and ready.")
    return model


# -------------------------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------------------------

def get_system_prompt() -> str:
    cfg = get_base_config()
    sys_prompt = cfg.get("system_prompt_editorial")
    if not sys_prompt:
        sys_prompt = get_lora_config().get("system_prompt_editorial", "")
    return sys_prompt or ""


# -------------------------------------------------------------------
# GENERATION HELPERS
# -------------------------------------------------------------------

def _get_gen_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "max_new_tokens": int(cfg.get("max_new_tokens", 220)),
        "temperature": float(cfg.get("temperature", 0.25)),
        "top_p": float(cfg.get("top_p", 0.9)),
        "repetition_penalty": float(cfg.get("repetition_penalty", 1.05)),
    }


def build_user_prompt(instruction: str, text: str) -> str:
    return (
        f"INSTRUCTION:\n{instruction.strip()}\n\n"
        f"TEXT:\n{text.strip()}"
    )


def generate_text(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    gen_params: Dict[str, Any],
) -> str:
    """
    Run generation and return decoded new text.

    If the model returns zero *new* tokens, we decode the tail of the sequence
    so you can at least see something and we log the lengths.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=enc.get("attention_mask", None).to(model.device)
            if "attention_mask" in enc
            else None,
            do_sample=True,
            **gen_params,
        )

    seq = outputs[0]
    total_len = seq.shape[-1]
    prompt_len = input_ids.shape[-1]
    new_len = total_len - prompt_len

    print(
        f"[compare-ui] generate_text lengths: total={total_len}, "
        f"prompt={prompt_len}, new={new_len}"
    )

    if new_len > 0:
        gen_tokens = seq[prompt_len:]
    else:
        tail = min(128, total_len)
        print("[compare-ui] WARNING: no new tokens; decoding last", tail, "tokens instead.")
        gen_tokens = seq[-tail:]

    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return text


# -------------------------------------------------------------------
# FLASK APP
# -------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / "templates"),
    static_folder=str(ROOT_DIR / "static"),
)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    instruction = ""
    text = ""
    base_output = None
    lora_output = None
    error_message = None

    if request.method == "POST":
        instruction = (request.form.get("instruction") or "").strip()
        text = (request.form.get("text") or "").strip()
        run_base = request.form.get("run_base") == "on"
        run_lora = request.form.get("run_lora") == "on"

        print("[compare-ui] POST received. run_base=%s run_lora=%s" % (run_base, run_lora))

        if not instruction or not text:
            error_message = "Please provide both Instruction and Text."
        elif not (run_base or run_lora):
            error_message = "Select at least one model to run."
        else:
            try:
                tokenizer = get_tokenizer()
                system_prompt = get_system_prompt()
                user_prompt = build_user_prompt(instruction, text)

                if run_base:
                    print("[compare-ui] running BASE generation...")
                    cfg = get_base_config()
                    params = _get_gen_params(cfg)
                    model = get_base_model()
                    base_output = generate_text(
                        model, tokenizer, system_prompt, user_prompt, params
                    )
                    print("[compare-ui] raw BASE output repr:", repr(base_output))

                if run_lora:
                    print("[compare-ui] running LoRA generation...")
                    cfg = get_lora_config()
                    params = _get_gen_params(cfg)
                    model = get_lora_model()
                    lora_output = generate_text(
                        model, tokenizer, system_prompt, user_prompt, params
                    )
                    print("[compare-ui] raw LoRA output repr:", repr(lora_output))

                    # If LoRA text is empty, surface that explicitly in the UI
                    if not lora_output:
                        lora_output = "[LoRA generation returned empty text]"

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_message = f"Error during generation: {e}"

    return render_template(
        "compare.html",
        instruction=instruction,
        text=text,
        base_output=base_output,
        lora_output=lora_output,
        error_message=error_message,
    )


@app.route("/")
def home():
    return redirect(url_for("compare"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7861, debug=False)
