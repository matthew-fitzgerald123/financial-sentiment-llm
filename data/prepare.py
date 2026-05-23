"""
Load financial sentiment data, format into instruction prompts, and write
train.jsonl / valid.jsonl that mlx_lm.lora expects.

Run once: python data/prepare.py

Dataset: nickmuchi/financial-classification
  columns: 'text' (string), 'labels' (int64: 0=negative, 1=neutral, 2=positive)
  splits:  train (4551), test (506)
"""
import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("./data")
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def format_prompt(example):
    label = LABEL_MAP[example["labels"]]
    return {
        "text": (
            f"<s>[INST] Classify the sentiment of the following financial statement "
            f"and briefly explain your reasoning:\n\n"
            f"\"{example['text']}\" [/INST]\n"
            f"Sentiment: {label}. This statement reflects "
            f"{'unfavorable' if label == 'negative' else 'neutral' if label == 'neutral' else 'favorable'} "
            f"financial conditions. </s>"
        )
    }


def write_jsonl(split, path):
    with open(path, "w") as f:
        for ex in split:
            f.write(json.dumps({"text": ex["text"]}) + "\n")
    print(f"Wrote {len(split)} examples → {path}")


def main():
    raw = load_dataset("nickmuchi/financial-classification")
    train = raw["train"].map(format_prompt)
    valid = raw["test"].map(format_prompt)

    write_jsonl(train, DATA_DIR / "train.jsonl")
    write_jsonl(valid, DATA_DIR / "valid.jsonl")

    print(f"\nSample:\n{train[0]['text']}")


if __name__ == "__main__":
    main()
