import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "outputs/llama31_lora_v2/checkpoint-200"

ARTICLE = """Officials approved temporary power-rationing in three cities after unseasonal heat pushed demand to record highs; hospitals and water utilities are exempt."""

PROMPT = (
    "You are an Al Jazeera–style news editor. "
    "Use ONLY the information in ARTICLE. "
    "Do NOT add facts, numbers, places or background that are not explicitly mentioned.\n\n"
    f"ARTICLE:\n{ARTICLE}\n\n"
)

MAX_NEW_TOKENS = 120


def generate_with_model(model, tokenizer, label: str):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # deterministic
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Show only what the model added after the prompt
    if text.startswith(PROMPT):
        completion = text[len(PROMPT):].lstrip()
    else:
        completion = text

    print(f"\n===== {label} COMPLETION =====\n")
    print(completion)
    print("\n=============================\n")


def load_base_model(device: str):
    print("Loading base model in 4-bit (no device_map)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,  # same as training
    )
    if device == "cuda":
        model.to(device)
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\n=== ARTICLE ===\n")
    print(ARTICLE)

    # 1) PURE BASELINE MODEL
    print("\n--- BASELINE RUN (no LoRA) ---")
    base_model = load_base_model(device)
    generate_with_model(base_model, tokenizer, "BASE MODEL")

    # Free baseline model to avoid VRAM issues
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2) LoRA MODEL (base + adapter)
    print("\n--- LoRA v2 RUN ---")
    base_for_lora = load_base_model(device)
    lora_model = PeftModel.from_pretrained(base_for_lora, ADAPTER_PATH)
    lora_model.eval()
    generate_with_model(lora_model, tokenizer, "LoRA v2 MODEL")


if __name__ == "__main__":
    main()
