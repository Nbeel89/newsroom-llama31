import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

ARTICLE = (
    "Officials approved temporary power-rationing in three cities after "
    "unseasonal heat pushed electricity demand to record highs; hospitals "
    "and water utilities are exempt."
)

print("=== LOADING PURE BASE MODEL ON CPU (NO LORA) ===")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load model fully on CPU, no 4bit quantization
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,   # CPU-friendly
    device_map={"": "cpu"},      # force everything to CPU
)

prompt = ARTICLE
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

print("\n=== GENERATION START (this may take some seconds) ===\n")

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=120,       # small generation to keep it quick
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
print("\n=== END ===")
