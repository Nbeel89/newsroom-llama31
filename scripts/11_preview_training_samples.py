from pathlib import Path
import json

def main():
    path = Path("data/train/newsroom_v2_train.jsonl")
    if not path.exists():
        print("File not found:", path)
        return

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:  # show only first 3 samples
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            print("=" * 80)
            print("ID:", rec.get("id"))
            print("\nINPUT (first 400 chars):\n")
            print(rec["input"][:400], "...\n")
            print("OUTPUT:\n")
            print(rec["output"], "\n")

if __name__ == "__main__":
    main()
