import argparse
from typing import Optional

from utils import generate_text


def main():
    parser = argparse.ArgumentParser(
        description="Run LLaMA 3.1 8B Instruct (newsroom-style) from the command line."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Article text or factual input.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="B",
        choices=["A", "B", "C", "a", "b", "c"],
        help="Generation mode: A=lede, B=brief, C=bullet headlines.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to inference YAML config (optional).",
    )

    args = parser.parse_args()

    text = generate_text(
        prompt=args.prompt,
        mode=args.mode,
        config_path=args.config,
    )

    print("\n=== OUTPUT ===\n")
    print(text)
    print("\n=============\n")


if __name__ == "__main__":
    main()
