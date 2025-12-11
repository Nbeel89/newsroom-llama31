import pathlib
import sys

# Base dir = project root (this file's folder)
BASE_DIR = pathlib.Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"

# Ensure scripts dir is on the Python path so we can import utils
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import generate_natural_text  # new "raw" generator


ARTICLE = (
    "Officials approved temporary power-rationing in three cities after "
    "unseasonal heat pushed electricity demand to record highs; hospitals "
    "and water utilities are exempt."
)

# Path to the BASE-ONLY config (no LoRA / adapters_path commented out)
BASE_CONFIG_PATH = "configs/llama31_inference_base.yaml"


def main():
    print("=== NATURAL LLM CONTINUATION (BASE MODEL ONLY) ===")
    print("PROMPT:")
    print(ARTICLE)
    print("\nOUTPUT:")

    out = generate_natural_text(
        prompt=ARTICLE,
        config_path=BASE_CONFIG_PATH,  # <-- important: use base-only config
        max_new_tokens=160,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.05,
    )

    print(out)
    print("=== END ===")


if __name__ == "__main__":
    main()
