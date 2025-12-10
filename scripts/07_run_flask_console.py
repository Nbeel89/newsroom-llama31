import os
import time
import pathlib
from functools import lru_cache

from flask import Flask, request, jsonify, Response
import torch

from utils import (
    generate_text,
    load_inference_config,
    MODE_PRESETS,
)

# -------- PATHS & BUILD TAG --------

THIS_FILE = pathlib.Path(__file__).resolve()
BASE_DIR = THIS_FILE.parents[1]
STATIC_DIR = BASE_DIR / "static"

BUILD_TAG = time.strftime("%Y-%m-%d_%H-%M-%S")


# -------- FLASK APP --------

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
)

# -------- SIMPLE HTML PAGE (UI) --------

HTML_PAGE = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Newsroom LLaMA Console — {BUILD_TAG}</title>
  <link rel="icon" href="/static/img/ajlogo.png?v={BUILD_TAG}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="/static/css/console.css?v={BUILD_TAG}">
  <style>
    :root {{
      --bg:#0b0c10; --card:#111217; --muted:#9aa3b2; --text:#e7eaee;
      --accent:#ffd166; --line:#1f2430; --btn:#1a73e8; --btnText:#fff; --chip:#202532;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:24px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background:var(--bg); color:var(--text);
    }}
    .topbar {{
      display:flex; align-items:center; justify-content:space-between;
      gap:16px; margin-bottom:16px;
    }}
    .brand {{
      display:flex;
      align-items:center;
      gap:14px;
    }}
    .brand-logo {{
      height:56px;
      width:56px;
      border-radius:18px;
      background:rgba(255,255,255,0.06);
      backdrop-filter:blur(4px);
      display:flex;
      align-items:center;
      justify-content:center;
      box-shadow:
        0 0 0 1px rgba(255,255,255,0.08),
        0 8px 22px rgba(0,0,0,0.55);
    }}
    .brand-logo img {{
      max-height:36px;
      max-width:36px;
      display:block;
      filter:drop-shadow(0 0 4px rgba(0,0,0,0.7));
    }}
    .muted {{ color:var(--muted); font-size:12px; }}
    .tiny {{ font-size:12px; }}
    .badges {{ display:flex; gap:8px; flex-wrap:wrap; }}
    .badge {{
      background:var(--chip); color:var(--muted);
      padding:6px 10px; border-radius:999px;
      font-size:12px; border:1px solid var(--line);
    }}
    .split {{
      display:grid; grid-template-columns:2fr 1fr; gap:16px;
    }}
    .card {{
      background:var(--card); border:1px solid var(--line);
      border-radius:14px; padding:16px;
    }}
    label {{ font-weight:600; display:block; margin-top:10px; margin-bottom:6px; }}
    select, textarea, input {{
      width:100%; background:#0f1117; color:var(--text);
      border:1px solid var(--line); border-radius:10px; padding:10px;
    }}
    textarea {{ min-height:140px; }}
    .row {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    button {{
      cursor:pointer; border:0; border-radius:10px;
      padding:10px 14px; background:var(--btn); color:var(--btnText);
      font-size:13px;
    }}
    button.ghost {{
      background:transparent; border:1px solid var(--line); color:var(--muted);
    }}
    pre {{
      white-space:pre-wrap; background:#0f1117;
      border:1px solid var(--line); border-radius:10px; padding:12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size:13px;
    }}
    .modechips {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }}
    .chip {{
      background:var(--chip); border:1px solid var(--line);
      padding:6px 10px; border-radius:999px; cursor:pointer; font-size:12px;
    }}
    .chip.active {{ outline:2px solid var(--btn); }}
    .right {{ display:flex; gap:8px; align-items:center; }}
    @media (max-width:900px) {{
      .split {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
<div class="topbar">
  <div class="brand">
    <div class="brand-logo">
      <img src="/static/img/ajlogo.png?v={BUILD_TAG}" alt="Al Jazeera" onerror="this.style.display='none'">
    </div>
    <div>
      <div style="font-weight:700">Al Jazeera · Newsroom LLaMA Console</div>
      <div class="muted tiny">LLaMA 3.1 8B · Local on-prem · Internal tooling</div>
    </div>
  </div>
  <div class="badges">
    <div class="badge">LLaMA 3.1 8B</div>
    <div class="badge">Flask</div>
    <div class="badge">GPU: <span id="gpu-label">—</span></div>
  </div>
</div>

  <div class="split">
    <div class="card">
      <form id="f">
        <label>Mode</label>
        <select name="mode" id="mode">
          <option value="A">A — One-sentence lede (≤20 words)</option>
          <option value="B" selected>B — Short news brief (2–3 sentences, ≤80 words)</option>
          <option value="C">C — 3 bullet headlines (5–9 words each)</option>
        </select>

        <div class="modechips" id="modechips">
          <div class="chip" data-mode="A">A</div>
          <div class="chip active" data-mode="B">B</div>
          <div class="chip" data-mode="C">C</div>
        </div>

        <label>Article / Facts</label>
        <textarea name="prompt" id="prompt" placeholder="Paste exact facts only. Model may still invent details — keep inputs tight and verify."></textarea>
        <div class="muted tiny" id="counter">0 words</div>

        <div class="row">
          <div>
            <label>Max new tokens</label>
            <input type="number" name="max_new_tokens" id="max_new_tokens" value="100" min="8" max="512">
          </div>
          <div>
            <label>Temperature</label>
            <input type="number" step="0.05" name="temperature" id="temperature" value="0.25" min="0" max="2">
          </div>
        </div>

        <div class="row">
          <div>
            <label>Top-p</label>
            <input type="number" step="0.05" name="top_p" id="top_p" value="0.85" min="0" max="1">
          </div>
          <div>
            <label>Top-k</label>
            <input type="number" name="top_k" id="top_k" value="40" min="0" max="200">
          </div>
        </div>

        <div class="row">
          <div>
            <label>Repetition penalty</label>
            <input type="number" step="0.05" name="repetition_penalty" id="repetition_penalty" value="1.1" min="0.8" max="2">
          </div>
          <div>
            <label>Stop (optional)</label>
            <input type="text" name="stop" id="stop" placeholder="Answer:, </s>, etc.">
          </div>
        </div>

        <div style="display:flex; gap:8px; align-items:center; margin-top:12px">
          <button type="submit">Generate</button>
          <button type="button" class="ghost" id="ping">Health</button>
          <div class="muted tiny">Outputs are length-guarded; stricter anti-hallucination filters are applied per mode.</div>
        </div>
      </form>
    </div>

    <div class="card">
      <div style="display:flex; align-items:center; justify-content:space-between">
        <h3 style="margin:0">Health</h3>
        <button class="ghost tiny" id="refresh">Refresh</button>
      </div>
      <pre id="health">—</pre>
      <div class="muted tiny">Base model, adapters, GPU device, and config flags.</div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px">
      <h3 style="margin:0">Output</h3>
      <div class="right">
        <button class="ghost tiny" id="copy">Copy</button>
      </div>
    </div>
    <pre id="out">—</pre>
  </div>

  <script>
    const PRESETS = {{
      A: {{max_new_tokens:28,  temperature:0.2,  top_p:0.8,  top_k:20, repetition_penalty:1.15}},
      B: {{max_new_tokens:100, temperature:0.25, top_p:0.85, top_k:40, repetition_penalty:1.1}},
      C: {{max_new_tokens:48,  temperature:0.2,  top_p:0.8,  top_k:30, repetition_penalty:1.2}},
    }};

    const $ = id => document.getElementById(id);
    const form = $('f');
    const out = $('out');
    const healthBox = $('health');
    const modeSel = $('mode');
    const chips = $('modechips');
    const promptEl = $('prompt');
    const copyBtn = $('copy');
    const gpuLabel = $('gpu-label');

    function applyPresets(mode) {{
      const p = PRESETS[mode] || PRESETS.B;
      $('max_new_tokens').value = p.max_new_tokens;
      $('temperature').value = p.temperature;
      $('top_p').value = p.top_p;
      $('top_k').value = p.top_k;
      $('repetition_penalty').value = p.repetition_penalty;
      [...chips.children].forEach(c => c.classList.toggle('active', c.dataset.mode === mode));
    }}

    modeSel.addEventListener('change', () => applyPresets(modeSel.value));
    chips.addEventListener('click', e => {{
      if (e.target.classList.contains('chip')) {{
        modeSel.value = e.target.dataset.mode;
        applyPresets(modeSel.value);
      }}
    }});

    applyPresets(modeSel.value);

    promptEl.addEventListener('input', () => {{
      const words = (promptEl.value.trim().match(/\\S+/g) || []).length;
      $('counter').textContent = `${{words}} word${{words !== 1 ? 's' : ''}}`;
    }});

    async function refreshHealth() {{
      try {{
        const r = await fetch('/health');
        const j = await r.json();
        healthBox.textContent = JSON.stringify(j, null, 2);
        if (j.gpu && j.gpu.device_name) {{
          gpuLabel.textContent = j.gpu.device_name;
        }}
      }} catch (e) {{
        healthBox.textContent = String(e);
      }}
    }}

    document.getElementById('ping').addEventListener('click', refreshHealth);
    document.getElementById('refresh').addEventListener('click', refreshHealth);
    refreshHealth();

    form.addEventListener('submit', async (e) => {{
      e.preventDefault();
      out.textContent = 'Generating...';

      const body = Object.fromEntries(new FormData(form).entries());
      body.mode = modeSel.value;
      ['max_new_tokens', 'top_k'].forEach(k => body[k] = parseInt(body[k], 10));
      ['temperature', 'top_p', 'repetition_penalty'].forEach(k => body[k] = parseFloat(body[k]));

      try {{
        const res = await fetch('/api/generate', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(body)
        }});
        const j = await res.json();
        if (res.ok) {{
          out.textContent = j.output || '(no output)';
        }} else {{
          out.textContent = j.error || 'Error from server';
        }}
      }} catch (err) {{
        out.textContent = String(err);
      }}
    }});

    copyBtn.addEventListener('click', async function () {{
      try {{
        await navigator.clipboard.writeText(out.textContent || '');
        this.textContent = 'Copied';
        setTimeout(() => this.textContent = 'Copy', 900);
      }} catch {{
        this.textContent = 'Failed';
        setTimeout(() => this.textContent = 'Copy', 900);
      }}
    }});
  </script>
</body>
</html>
"""


# -------- HEALTH CACHE (NO MODEL LOAD) --------

@lru_cache(maxsize=1)
def _get_health_model_info():
    """
    Return static configuration + GPU info for the Health panel.

    IMPORTANT: we DO NOT load the big model here, to avoid meta-tensor /
    device_map issues and double-loading. The model is only created on
    first /api/generate call via utils.generate_text().
    """
    cfg = load_inference_config()

    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else "cpu"

    adapters_cfg = cfg.get("adapters_path", "") or ""
    adapters_env = os.environ.get("ADAPTERS", "").strip()
    adapters = adapters_cfg or adapters_env or None

    return {
        "status": "ok",
        "build": BUILD_TAG,
        "base_model": cfg.get("model_id"),
        "adapters": adapters,
        "config": {
            "adapters_path": adapters_cfg,
            "base_model_id": cfg.get("model_id"),
            "default_mode": cfg.get("default_mode", "B"),
            "device_map": cfg.get("device_map"),
            "load_4bit": bool(cfg.get("load_4bit", True)),
            "max_ctx": int(cfg.get("max_ctx", 2048)),
            # show B's defaults as representative
            "max_new_tokens": MODE_PRESETS["B"]["params"]["max_new_tokens"],
            "temperature": MODE_PRESETS["B"]["params"]["temperature"],
            "top_p": MODE_PRESETS["B"]["params"]["top_p"],
            "repetition_penalty": MODE_PRESETS["B"]["params"]["repetition_penalty"],
        },
        "gpu": {
            "cuda_available": cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "device_name": gpu_name,
        },
        "modes": {k: v["label"] for k, v in MODE_PRESETS.items()},
    }


# -------- ROUTES --------

@app.get("/")
def index():
    # No-cache headers so HTML/JS always reflect latest build tag
    return Response(
        HTML_PAGE,
        headers={
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
def health():
    return jsonify(_get_health_model_info())


@app.post("/api/generate")
def api_generate():
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "").strip()
    mode = (data.get("mode") or "B").strip().upper()

    # Safety: if something weird comes in, fall back to B
    if mode not in ("A", "B", "C"):
        mode = "B"

    if not prompt:
        return jsonify(error="Empty prompt"), 400

    override = {}
    for key in ["max_new_tokens", "top_k"]:
        if key in data:
            try:
                override[key] = int(data[key])
            except Exception:
                pass
    for key in ["temperature", "top_p", "repetition_penalty"]:
        if key in data:
            try:
                override[key] = float(data[key])
            except Exception:
                pass

    try:
        out = generate_text(
            prompt=prompt,
            mode=mode,
            override_params=override or None,
        )
        return jsonify(output=out)
    except Exception as e:
        print("[error] generate_text failed:", repr(e))
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    print(f"[boot] starting Flask console on http://127.0.0.1:{port}")
    print(f"[boot] base_dir={BASE_DIR}")
    app.run(host="127.0.0.1", port=port, debug=False)
