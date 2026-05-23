"""
Evaluate fine-tuned adapter vs base Mistral on the test set using ROUGE-L.
Prints a comparison table and saves full results to eval/results.json.

Usage: python eval/eval.py
"""
import json
from pathlib import Path
from rouge_score import rouge_scorer
from mlx_lm import load, generate

MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
ADAPTER_PATH = "./mistral-finetuned"
VALID_JSONL = "./data/valid.jsonl"
RESULTS_PATH = "./eval/results.json"
NUM_EXAMPLES = 50
MAX_TOKENS = 128


def load_examples(path, n):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def build_question(text):
    return text.split("[INST]")[1].split("[/INST]")[0].strip()


def build_ground_truth(text):
    return text.split("[/INST]")[1].replace("</s>", "").strip()


def run_generate(model, tokenizer, question):
    prompt = f"<s>[INST] {question} [/INST]"
    return generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS)


def main():
    examples = load_examples(VALID_JSONL, NUM_EXAMPLES)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    print("Loading base model...")
    base_model, base_tok = load(MODEL_ID)

    print("Loading fine-tuned model...")
    ft_model, ft_tok = load(MODEL_ID, adapter_path=ADAPTER_PATH)

    results = []
    for i, ex in enumerate(examples):
        question = build_question(ex["text"])
        ground_truth = build_ground_truth(ex["text"])

        base_answer = run_generate(base_model, base_tok, question)
        ft_answer = run_generate(ft_model, ft_tok, question)

        base_scores = scorer.score(ground_truth, base_answer)
        ft_scores = scorer.score(ground_truth, ft_answer)

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "base_model": base_answer,
            "finetuned": ft_answer,
            "base_rouge1": base_scores["rouge1"].fmeasure,
            "base_rougeL": base_scores["rougeL"].fmeasure,
            "ft_rouge1": ft_scores["rouge1"].fmeasure,
            "ft_rougeL": ft_scores["rougeL"].fmeasure,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(examples)} evaluated")

    Path(RESULTS_PATH).parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    avg = lambda key: sum(r[key] for r in results) / len(results)
    print(f"\n{'Model':<20} {'ROUGE-1':>10} {'ROUGE-L':>10}")
    print("-" * 42)
    print(f"{'Base Mistral-7B':<20} {avg('base_rouge1'):>10.3f} {avg('base_rougeL'):>10.3f}")
    print(f"{'Fine-tuned':<20} {avg('ft_rouge1'):>10.3f} {avg('ft_rougeL'):>10.3f}")
    print(f"\nFull results → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
