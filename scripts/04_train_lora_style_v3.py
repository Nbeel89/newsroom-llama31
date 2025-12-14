"""
04_train_lora_style_v3.py

LoRA v3 – AJ editorial STYLE training on full articles.

- Uses data/cleaned/aj_style_sft_v3.jsonl (one {"text": "..."} per line)
- Tries 4-bit (QLoRA); if not available, falls back to fp16 model with
  aggressive memory-saving settings.

Run:
    (venv) python scripts/04_train_lora_style_v3.py
"""

import pathlib
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# PEFT (LoRA)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


@dataclass
class TrainConfig:
    base_model_id: str
    dataset_path: str
    output_dir: str
    max_seq_len: int
    load_4bit: bool

    # LoRA settings
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]

    # training settings
    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int | float
    learning_rate: float | str
    warmup_steps: int | str
    weight_decay: float | str
    save_steps: int | str
    logging_steps: int | str

    # optional safety knobs
    max_steps: int | str | None = None  # allow stopping early if needed


def load_cfg(cfg_path: pathlib.Path) -> TrainConfig:
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainConfig(**raw)


def _try_load_4bit_model(cfg: TrainConfig):
    """
    Try QLoRA loading: 4-bit base weights + LoRA adapters trained in fp16.
    Force model placement on GPU 0 (prevents CPU/disk scattering).
    """
    if BitsAndBytesConfig is None:
        return None

    print("[train-v3] Trying 4-bit quantization (QLoRA) on cuda:0…")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},  # IMPORTANT: keep everything on GPU 0
    )
    return model


def _load_fp16_fallback(cfg: TrainConfig):
    """
    Fallback if 4-bit doesn't work:
    Loads fp16 model with device_map=auto (may offload to CPU/disk).
    This is slower and riskier on a 6GB GPU.
    """
    print("[train-v3] Loading model in fp16 without 4-bit quantization…")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return model


def load_model_and_tokenizer(cfg: TrainConfig):
    print("[train-v3] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[train-v3] Loading base model…")

    model = None

    # --- Try QLoRA first ---
    if bool(cfg.load_4bit):
        try:
            model = _try_load_4bit_model(cfg)
            print("[train-v3] 4-bit model loaded successfully.")
        except Exception as e:
            print("[train-v3][WARN] 4-bit loading failed, falling back to fp16.")
            print("          Reason:", repr(e))

    # --- Fallback: fp16 ---
    if model is None:
        model = _load_fp16_fallback(cfg)

    # Memory/stability settings
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # If in k-bit, prepare for k-bit training (important for QLoRA)
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    print("[train-v3] Applying LoRA adapters…")
    lora_cfg = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(model, lora_cfg)

    # Show trainable params (sanity)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    model.train()
    return model, tokenizer


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    cfg_path = root / "configs" / "llama31_lora_style_v3.yaml"
    cfg = load_cfg(cfg_path)

    print("=== LoRA v3 – AJ Style Training ===")
    print("base_model_id   :", cfg.base_model_id)
    print("dataset_path    :", cfg.dataset_path)
    print("output_dir      :", cfg.output_dir)
    print("max_seq_len     :", cfg.max_seq_len)
    print("batch_size      :", cfg.batch_size)
    print("grad_accum      :", cfg.gradient_accumulation_steps)
    print("num_epochs      :", cfg.num_epochs)
    print("learning_rate   :", cfg.learning_rate)
    print("load_4bit       :", cfg.load_4bit)
    print("lora_r          :", cfg.lora_r)
    print("lora_alpha      :", cfg.lora_alpha)
    print("lora_dropout    :", cfg.lora_dropout)
    print("target_modules  :", cfg.target_modules)

    dataset_path = root / cfg.dataset_path
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir = root / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(cfg)

    print("[train-v3] Loading dataset…")
    raw_dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=int(cfg.max_seq_len),
            truncation=True,
            padding="max_length",
        )

    print("[train-v3] Tokenizing dataset…")
    tokenized_ds = raw_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Cast config values safely
    lr = float(cfg.learning_rate)
    warmup = int(cfg.warmup_steps)
    weight_decay = float(cfg.weight_decay)
    save_steps = int(cfg.save_steps)
    logging_steps = int(cfg.logging_steps)
    num_epochs = float(cfg.num_epochs)
    batch_size = int(cfg.batch_size)
    grad_accum = int(cfg.gradient_accumulation_steps)

    # Optional max_steps override
    max_steps = None
    if cfg.max_steps is not None and str(cfg.max_steps).strip() != "":
        max_steps = int(cfg.max_steps)

    # If using QLoRA (4bit), fp16 training is normal and stable with GradScaler.
    # If not 4bit, fp16 + CPU offload can be unstable; still allow fp16 but keep clipping.
    use_fp16 = bool(getattr(model, "is_loaded_in_4bit", False))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=lr,
        warmup_steps=warmup,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,

        fp16=use_fp16,
        bf16=False,

        gradient_checkpointing=True,
        max_grad_norm=0.3,   # lower clipping helps avoid NaNs
        logging_dir=str(out_dir / "logs"),
        report_to="none",
        save_total_limit=2,
    )

    print("[train-v3] Starting Trainer…")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    trainer.train()

    print("[train-v3] Saving final LoRA adapter + tokenizer…")
    # With PEFT model, save_pretrained writes only adapter weights + config.
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print("[train-v3] Done. Adapters saved in:", out_dir)


if __name__ == "__main__":
    main()
