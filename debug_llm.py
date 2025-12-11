import pathlib
import sys

# Base dir = project root (this file's folder)
BASE_DIR = pathlib.Path(__file__).resolve().parent

# Scripts dir, where utils.py lives
SCRIPTS_DIR = BASE_DIR / "scripts"

# Ensure scripts dir is on the Python path
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import generate_text  # uses your existing pipeline

ARTICLE = (
    "Officials approved temporary power-rationing in three cities after "
    "unseasonal heat pushed electricity demand to record highs; hospitals "
    "and water utilities are exempt."
)


def run_mode(mode: str) -> None:
    print("\n" + "=" * 40)
    print(f"MODE {mode}")
    print("-" * 40)
    out = generate_text(prompt=ARTICLE, mode=mode)
    print(out)
    print("=" * 40 + "\n")


if __name__ == "__main__":
    for m in ["A", "B", "C"]:
        run_mode(m)
