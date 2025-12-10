#!/usr/bin/env python

"""
04_train_lora_llama.py — FINAL STABLE VERSION
LoRA fine-tuning for LLaMA 3.1 8B (4-bit) on a single GPU using HF Trainer.
"""

import argparse
import yaml
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_training_text(example):
    """
    Build a single 'text' field for SFT.

    Supports two schemas:
    1) Old:  instruction / input / output
    2) New:  mode / article / output   (our v2 dataset)
    """

    if "instruction" in example:
        # OLD SCHEMA: keep backward compatible
        instruction = example["instruction"]
        inp = example.get("input", "")
    else:
        # NEW SCHEMA: use mode + article to build the instruction
        mode = (example.get("mode") or "A").upper()
        article = example.get("article", "")

        if mode == "A":
            instruction = (
                "Write ONE concise paragraph that summarises the ARTICLE in a "
                "neutral news tone. Maximum 3 sentences. Do NOT add any "
                "information that is not in the ARTICLE."
            )
        elif mode == "B":
            instruction = (
                "Produce a HEADLINE and a LEDE based ONLY on the ARTICLE.\n\n"
                "FORMAT:\n"
                "HEADLINE: <headline in sentence case, no more than 14 words>\n"
                "LEDE: <one sentence, no more than 40 words>\n"
                "Do NOT add numbers, dates or quotes that are not explicitly "
                "mentioned in the ARTICLE."
            )
        elif mode == "C":
            instruction = (
                "Produce a HEADLINE, a LEDE and exactly three factual bullets "
                "based ONLY on the ARTICLE.\n\n"
                "FORMAT:\n"
                "HEADLINE: <headline in sentence case, no more than 14 words>\n"
                "LEDE: <one sentence, no more than 40 words>\n"
                "BULLETS:\n"
                "- <fact 1>\n"
                "- <fact 2>\n"
                "- <fact 3>\n"
                "Each bullet must state a single fact from the ARTICLE. "
                "Do NOT invent new details."
            )
        else:
            instruction = (
                "Summarise the ARTICLE in a single paragraph using only the "
                "facts provided, in a neutral news tone."
            )

        inp = article

    out = example["output"]

    return {
        "text": (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Output:\n{out}"
        )
    }


def main():
    parser = argparse.ArgumentParser(description="Train LLaMA 3.1 8B with LoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_id = cfg["model_id"]
    dataset_path = cfg["dataset_path"]
    output_dir = cfg["output_dir"]
    max_seq_length = cfg.get("max_seq_length", 512)

    print("=== Config ===")
    print(f"model_id:     {model_id}")
    print(f"dataset_path: {dataset_path}")
    print(f"output_dir:   {output_dir}")
    print(f"max_seq_len:  {max_seq_length}")

    # ---------------- Tokenizer ----------------
    print("=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Model (4-bit) ----------------
    load_in_4bit = cfg.get("load_in_4bit", True)
    print(f"=== Loading base model (4bit={load_in_4bit}) ===")

    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # IMPORTANT: device_map=None → model init on CPU,
        # Trainer/Accelerate will move it to the correct device.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map=None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=None,
        )

    # Disable cache for training
    model.config.use_cache = False

    # ---------------- LoRA ----------------
    print("=== Applying LoRA ===")
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg.get("bias", "none"),
        task_type=cfg.get("task_type", "CAUSAL_LM"),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------- Dataset ----------------
    print("=== Loading dataset ===")
    raw_ds = load_dataset("json", data_files={"train": dataset_path})

    print("=== Building training text field ===")
    ds_with_text = raw_ds["train"].map(build_training_text)

    def tokenize_fn(example):
        # Pad & truncate to fixed length so the data collator can batch safely
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        # labels = input_ids (standard causal LM fine-tuning)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    print("=== Tokenizing dataset ===")
    tokenized_ds = ds_with_text.map(
        tokenize_fn,
        remove_columns=ds_with_text.column_names,
        desc="Tokenizing",
    )

    # ---------------- TrainingArguments ----------------
    print("=== Setting training arguments ===")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        logging_steps=cfg.get("logging_steps", 25),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 2),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=False,  # force off
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
    )

    data_collator = default_data_collator

    # ---------------- Trainer ----------------
    print("=== Starting Trainer ===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    # ---------------- Train ----------------
    print("=== Training starting ===")
    trainer.train()

    # ---------------- Save ----------------
    print("=== Saving LoRA adapter & tokenizer ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"=== Training complete. Adapter saved to: {output_dir} ===")


if __name__ == "__main__":
    main()
