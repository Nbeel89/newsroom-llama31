import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "outputs/llama31_lora_v1"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    print("Loading 4-bit base model on GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="cuda",
        load_in_4bit=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    prompt = (
        "You are an Al Jazeera–style news editor. "
        "Write a neutral HEADLINE and a 30-word LEDE about: "
        "Officials approved temporary power rationing in three cities after "
        "unseasonal heat pushed electricity demand to record highs; "
        "hospitals and water utilities are exempt."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            temperature=0.3,
            top_p=0.9,
        )

    print("\n=== MODEL OUTPUT ===\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
