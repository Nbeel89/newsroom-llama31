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
from flask import Flask, request, redirect, url_for, render_template_string

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
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
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
    model = PeftModel.from_pretrained(base, str(adapters_path))
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
    return f"INSTRUCTION:\n{instruction.strip()}\n\nTEXT:\n{text.strip()}"


def generate_text(model, tokenizer, system_prompt: str, user_prompt: str, gen_params: Dict[str, Any]) -> str:
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

    print(f"[compare-ui] generate_text lengths: total={total_len}, prompt={prompt_len}, new={new_len}")

    if new_len > 0:
        gen_tokens = seq[prompt_len:]
    else:
        tail = min(128, total_len)
        print("[compare-ui] WARNING: no new tokens; decoding last", tail, "tokens instead.")
        gen_tokens = seq[-tail:]

    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# -------------------------------------------------------------------
# INLINE DARK THEME TEMPLATE (NO compare.html DEPENDENCY)
# -------------------------------------------------------------------

DARK_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Newsroom LLM – Compare</title>
  <style>
    :root{
      --bg0:#070a0f; --bg1:#0b1220;
      --panel:rgba(8,12,22,.55);
      --card:rgba(255,255,255,.06);
      --text:#e5e7eb; --muted:#a7b0c0;
      --line:rgba(255,255,255,.10);
      --blue:#2563eb; --blue2:#1d4ed8;
      --danger:#ef4444;
      --shadow:0 10px 30px rgba(0,0,0,.45);
      --radius:16px;
    }
    body{
      margin:0;
      font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;
      background:
        radial-gradient(1200px 600px at 20% 0%, rgba(37,99,235,.15), transparent 45%),
        radial-gradient(900px 500px at 80% 10%, rgba(34,197,94,.10), transparent 40%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
      color:var(--text);
    }
    .wrap{max-width:1200px;margin:28px auto;padding:0 18px 40px;}
    .title{display:flex;align-items:center;gap:12px;font-size:34px;font-weight:750;letter-spacing:-.02em;margin:6px 0 6px;}
    .subtitle{color:var(--muted);margin:0 0 18px;font-size:14px;}
    .card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden;}
    .card-inner{background:var(--panel);padding:18px;}
    label{display:block;font-size:12px;color:var(--muted);margin:0 0 8px;letter-spacing:.02em;}
    textarea{
      width:100%;min-height:120px;resize:vertical;
      border-radius:12px;border:1px solid var(--line);
      background:rgba(3,6,14,.55);color:var(--text);
      padding:12px;outline:none;line-height:1.4;font-size:13px;
    }
    textarea:focus{border-color:rgba(37,99,235,.65);box-shadow:0 0 0 3px rgba(37,99,235,.20);}
    .row{display:grid;grid-template-columns:1fr;gap:14px;}
    @media(min-width:980px){.row.two{grid-template-columns:1fr 1fr;}}
    .controls{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;margin-top:12px;}
    .checks{display:flex;gap:14px;align-items:center;color:var(--muted);font-size:13px;}
    .checks input{transform:translateY(1px);}
    .btn{
      border:1px solid rgba(255,255,255,.12);
      background:linear-gradient(180deg,var(--blue),var(--blue2));
      color:white;padding:10px 16px;border-radius:12px;cursor:pointer;
      font-weight:650;letter-spacing:.01em;
    }
    .btn:hover{filter:brightness(1.05);}
    .tip{margin-top:10px;color:var(--muted);font-size:12px;}
    .err{
      margin-top:14px;padding:12px;border-radius:12px;
      border:1px solid rgba(239,68,68,.35);background:rgba(239,68,68,.10);
      color:#fecaca;font-size:13px;
    }
    .results{margin-top:18px;}
    .results h2{margin:0 0 10px;font-size:18px;letter-spacing:-.01em;}
    .outbox{
      min-height:220px;white-space:pre-wrap;
      border-radius:12px;border:1px solid var(--line);
      background:rgba(3,6,14,.45);
      padding:12px;color:var(--text);
      font-size:13px;line-height:1.45;
    }
    .pill{
      display:inline-flex;align-items:center;gap:8px;
      border:1px solid rgba(255,255,255,.12);
      background:rgba(255,255,255,.05);
      padding:8px 10px;border-radius:999px;
      color:var(--muted);font-size:12px;margin-bottom:14px;
    }
    code{color:#c7d2fe;}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">📰 Newsroom LLM – Compare</div>
    <div class="subtitle">Compare Base vs LoRA outputs (same prompts, same inputs).</div>

    <div class="pill">LoRA Adapter: <code>{{ adapters_path }}</code></div>

    <div class="card">
      <div class="card-inner">
        <form method="post" action="/compare">
          <div class="row">
            <div>
              <label>Instruction</label>
              <textarea name="instruction">{{ instruction }}</textarea>
            </div>
            <div>
              <label>Text to process</label>
              <textarea name="text">{{ text }}</textarea>
            </div>
          </div>

          <div class="controls">
            <div class="checks">
              <label style="margin:0;">
                <input type="checkbox" name="run_base" {% if run_base_checked %}checked{% endif %}>
                Run Base Model
              </label>
              <label style="margin:0;">
                <input type="checkbox" name="run_lora" {% if run_lora_checked %}checked{% endif %}>
                Run Fine-tuned (LoRA)
              </label>
            </div>
            <button class="btn" type="submit">Compare</button>
          </div>

          <div class="tip">Tip: keep input under ~800–1200 words for faster results on 6GB VRAM.</div>

          {% if error_message %}
            <div class="err">{{ error_message }}</div>
          {% endif %}
        </form>

        {% if show_results %}
          <div class="results">
            <h2>Results</h2>
            <div class="row two">
              <div>
                <label>Base Model Output</label>
                <div class="outbox">{{ base_output }}</div>
              </div>
              <div>
                <label>Fine-tuned (LoRA) Output</label>
                <div class="outbox">{{ lora_output }}</div>
              </div>
            </div>
          </div>
        {% endif %}
      </div>
    </div>
  </div>
</body>
</html>
"""


# -------------------------------------------------------------------
# FLASK APP
# -------------------------------------------------------------------

app = Flask(__name__)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    instruction = ""
    text = ""
    base_output = ""
    lora_output = ""
    error_message = None
    show_results = False

    # show which adapter is configured (nice to confirm you are on correct run)
    adapters_path = ""
    try:
        adapters_path = str(get_lora_config().get("adapters_path", "") or "")
    except Exception:
        adapters_path = ""

    # default: checked on GET (like your old UI)
    run_base_checked = True
    run_lora_checked = True

    if request.method == "POST":
        instruction = (request.form.get("instruction") or "").strip()
        text = (request.form.get("text") or "").strip()
        run_base = request.form.get("run_base") == "on"
        run_lora = request.form.get("run_lora") == "on"

        run_base_checked = run_base
        run_lora_checked = run_lora

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
                    base_output = generate_text(model, tokenizer, system_prompt, user_prompt, params)

                if run_lora:
                    print("[compare-ui] running LoRA generation...")
                    cfg = get_lora_config()
                    params = _get_gen_params(cfg)
                    model = get_lora_model()
                    lora_output = generate_text(model, tokenizer, system_prompt, user_prompt, params)
                    if not lora_output:
                        lora_output = "[LoRA generation returned empty text]"

                show_results = True

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_message = f"Error during generation: {e}"

    return render_template_string(
        DARK_TEMPLATE,
        instruction=instruction,
        text=text,
        base_output=base_output,
        lora_output=lora_output,
        error_message=error_message,
        show_results=show_results,
        run_base_checked=run_base_checked,
        run_lora_checked=run_lora_checked,
        adapters_path=adapters_path,
    )


@app.route("/")
def home():
    return redirect(url_for("compare"))


if __name__ == "__main__":
    print("[compare-ui] ROOT_DIR =", ROOT_DIR)
    print("[compare-ui] CONFIG_BASE =", CONFIG_BASE)
    print("[compare-ui] CONFIG_LORA =", CONFIG_LORA)
    app.run(host="0.0.0.0", port=7861, debug=False)
