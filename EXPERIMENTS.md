# Newsroom LLaMA – LoRA Training Experiments Log

This document tracks *every* LoRA fine-tuning experiment, the dataset used, config file used, output directory, and current status.  
It acts as the single source of truth for your team.

---

## ✅ CURRENT ACTIVE EXPERIMENTS

### **LoRA v2 – Current Production / Demo Model**
- **Date:** 2025-12-02  
- **Training Script:** `scripts/04_train_lora_llama.py`
- **Training Config:** `configs/llama31_lora_config_v2.yaml`
- **Dataset:** `data/train/newsroom_v2_train.jsonl`
- **Output Directory:** `outputs/llama31_lora_v2/`
- **Best Checkpoint:** `checkpoint-200`
- **Used By Inference:**
  - `scripts/06_inference_llama.py`
  - `scripts/07_run_flask_console.py`
  - `scripts/08_run_compare_ui.py`
- **Description:**  
  Supervised fine-tuning on “input → output” pairs created via earlier dataset generation.  
  Used as the **current LoRA adapter** for newsroom compare UI and demonstrations.
- **Status:** ✅ ACTIVE (Default)

---

## 🟡 LEGACY EXPERIMENTS

### **LoRA v1 – First Attempt (Legacy)**
- **Date:** 2025-11-27  
- **Training Script:** `scripts/04_train_lora_llama.py`
- **Training Config:** `configs/llama31_lora_config.yaml`
- **Dataset:** `data/cleaned/newsroom_finetune.jsonl`
- **Output Directory:** `outputs/llama31_lora_v1/`
- **Best Checkpoint:** `checkpoint-124`
- **Notes:**  
  Early prototype. We keep it for reference but do not use it in the UI or further training.
- **Status:** ❌ LEGACY (Not used)

---

## 🚧 UPCOMING EXPERIMENTS (Planned)

### **LoRA v3 – AJ Editorial Style Training (Planned)**
- **Goal:**  
  Train LoRA *only* on real Al Jazeera crawled articles to learn pure AJ writing style (domain adaptation).
- **Dataset:** `data/train/newsroom_v3_train.jsonl` (To be generated)
- **Training Config:** `configs/llama31_lora_config_v3.yaml` (To be created)
- **Output Directory:** `outputs/llama31_lora_v3/`
- **Will Improve:**
  - Tone
  - Structure
  - Neutrality
  - Grammar
  - Attribution
  - AJ-specific style patterns

When v3 is trained, add a block here just like v1 and v2.

---

## 📝 How to Add a New Experiment Entry

Every time a new LoRA training run is completed:

1. Create a new section (v4, v5…)
2. Add:
   - Date
   - Script used
   - Config used
   - Dataset used
   - Output checkpoints
   - Observations
3. Update **ACTIVE** or **LEGACY** status

This keeps the project clean, auditable, and understandable for the team.

---
