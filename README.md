# Newsroom LLaMA 3.1 LoRA Fine-Tuning Pipeline

A complete, lightweight, and production-ready pipeline for fine-tuning
**Meta LLaMA-3.1-8B-Instruct** using **LoRA**, optimized for generating
**Al Jazeera--style newsroom summaries** (headline, lede, and factual
bullets).

This repository includes everything end-to-end: - 🔎 Web crawling (RSS +
sitemaps + sections) - 🧹 Text cleaning & transformation - 📘 Dataset
preparation for supervised fine-tuning - 🧠 QLoRA training pipeline - 📊
Evaluation (baseline vs LoRA) - ⚡ Local inference engine - 🖥️ Minimal
Flask console UI

## 🚀 Features

-   **Deep Article Crawl** (RSS feeds + sitemap discovery + section
    pages)\
-   **Robust Cleaning** (HTML removal, normalization, deduplication,
    min-word filters)\
-   **Training** using QLoRA (4-bit quantization) for cost-efficient
    fine-tuning\
-   **Evaluation Scripts** with outputs stored in `data/eval`\
-   **Local Inference** using adapter injection\
-   **Modular Structure** --- configurable through YAML files

## 📂 Project Structure

    newsroom-llama31/
    │   README.md
    │   requirements.txt
    │   .gitignore
    │
    ├── configs/
    │   ├── llama31_inference.yaml
    │   └── llama31_lora_config.yaml
    │
    ├── data/
    │   ├── cleaned/
    │   └── eval/
    │
    ├── scripts/
    │   ├── 01_crawl_articles.py
    │   ├── 02_clean_articles.py
    │   ├── 03_prepare_dataset.py
    │   ├── 04_train_lora_llama.py
    │   ├── 05_evaluate_lora.py
    │   ├── 06_inference_llama.py
    │   ├── 07_run_flask_console.py
    │   ├── 08_batch_eval.py
    │   ├── 10_prepare_eval_sheet.py
    │   └── crawl_site_full_v4.py
    │
    └── static/
        ├── css/
        └── img/

## ⚙️ Installation

``` bash
git clone https://github.com/Nbeel89/newsroom-llama31.git
cd newsroom-llama31
pip install -r requirements.txt
```

Requirements: - Python 3.10+ - CUDA GPU recommended (for training) -
HuggingFace login (if model requires authentication)

## 🕸️ 1. Crawl News Articles

``` bash
python scripts/01_crawl_articles.py   --days 3650   --max-pages 50   --workers 6   --min-words 80   --out data/raw/articles.jsonl
```

## 🧹 2. Clean Articles

``` bash
python scripts/02_clean_articles.py   --input data/raw/articles.jsonl   --output data/cleaned/aljazeera_articles_cleaned.jsonl
```

## 📘 3. Prepare Fine-Tuning Dataset

``` bash
python scripts/03_prepare_dataset.py   --input data/cleaned/aljazeera_articles_cleaned.jsonl   --output data/cleaned/newsroom_finetune.jsonl
```

## 🧠 4. Train LoRA Adapter on LLaMA-3.1-8B

``` bash
python scripts/04_train_lora_llama.py   --config configs/llama31_lora_config.yaml
```

Outputs appear in:

    outputs/llama31_lora_v1/

## 📊 5. Evaluate Model (Baseline vs LoRA)

``` bash
python scripts/05_evaluate_lora.py
```

Evaluation results are stored under:

    data/eval/

## 🔍 6. Run Local Inference

``` bash
python scripts/06_inference_llama.py   --config configs/llama31_inference.yaml
```

## 🖥️ 7. Optional: Run the Flask Console UI

``` bash
python scripts/07_run_flask_console.py
```

This opens a minimal local web interface to test summaries.

## 🧾 License

MIT License --- free for use, modification, and commercial work.

## ⭐ Acknowledgments

-   Meta --- LLaMA 3.1\
-   HuggingFace Transformers\
-   LoRA (Hu et al. 2021)\
-   QLoRA (Dettmers et al., 2023)\
-   Al Jazeera article dataset generated via public web content
