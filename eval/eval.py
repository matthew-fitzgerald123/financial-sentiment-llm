"""
Evaluate fine-tuned adapter vs base Mistral on the test set using ROUGE-L
and label accuracy.
Prints a comparison table, saves full per-example results to eval/results.json,
saves aggregate metrics to eval/summary.json, and logs metrics to the shared
MLflow experiment.

Usage:
    python eval/eval.py
    python eval/eval.py --data /path/to/ood_data.jsonl --n 100
"""
import argparse
import json
import re
from pathlib import Path
from rouge_score import rouge_scorer
import mlflow
from mlx_lm import load, generate

MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
ADAPTER_PATH = "./mistral-finetuned"
VALID_JSONL = "./data/valid.jsonl"
RESULTS_PATH = "./eval/results.json"
SUMMARY_PATH = "./eval/summary.json"
NUM_EXAMPLES = 50
MAX_TOKENS = 128
ROUGE_L_GATE = 0.85


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


def _parse_label(text):
    """Return the sentiment label from a model output, or 'unknown'."""
    m = re.search(r"Sentiment:\s*(positive|neutral|negative)", text, re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"


def label_accuracy(results):
    """Fraction of examples where fine-tuned label matches ground-truth label.

    Each element of *results* must have 'ground_truth' and 'finetuned' keys
    containing raw model-output strings in the expected structured format.
    Returns a float in [0, 1], or 0.0 for an empty list.
    """
    if not results:
        return 0.0
    correct = sum(
        1
        for r in results
        if _parse_label(r["finetuned"]) == _parse_label(r["ground_truth"])
    )
    return correct / len(results)


def save_summary(path, summary):
    """Write aggregate eval metrics to *path* as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def run_generate(model, tokenizer, question, max_tokens=MAX_TOKENS):
    prompt = f"<s>[INST] {question} [/INST]"
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def compute_averages(results: list[dict]) -> dict:
    """Compute average ROUGE metrics from a list of per-example result dicts.

    Returns a flat dict with keys:
      base_avg_rouge1, base_avg_rougeL, ft_avg_rouge1, ft_avg_rougeL,
      ft_rougeL_gate_passed (bool)
    """
    n = len(results)
    base_r1 = sum(r["base_rouge1"] for r in results) / n
    base_rL = sum(r["base_rougeL"] for r in results) / n
    ft_r1   = sum(r["ft_rouge1"]   for r in results) / n
    ft_rL   = sum(r["ft_rougeL"]   for r in results) / n
    return {
        "base_avg_rouge1":     base_r1,
        "base_avg_rougeL":     base_rL,
        "ft_avg_rouge1":       ft_r1,
        "ft_avg_rougeL":       ft_rL,
        "ft_rougeL_gate_passed": ft_rL >= ROUGE_L_GATE,
    }


ROUGE_L_THRESHOLD = 0.85


def check_gate(results: list, threshold: float = ROUGE_L_THRESHOLD) -> tuple:
    """Return (avg_ft_rougeL, passed) where passed is True iff avg >= threshold."""
    avg = sum(r["ft_rougeL"] for r in results) / len(results)
    return avg, avg >= threshold


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned vs base model")
    parser.add_argument("--data", default=VALID_JSONL, help="JSONL file to evaluate on")
    parser.add_argument("--n", type=int, default=NUM_EXAMPLES, help="Number of examples")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, dest="max_tokens")
    parser.add_argument("--adapter", default=ADAPTER_PATH, help="LoRA adapter path")
    args = parser.parse_args()

    examples = load_examples(args.data, args.n)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    print("Loading base model...")
    base_model, base_tok = load(MODEL_ID)

    print("Loading fine-tuned model...")
    ft_model, ft_tok = load(MODEL_ID, adapter_path=args.adapter)

    results = []
    for i, ex in enumerate(examples):
        question = build_question(ex["text"])
        ground_truth = build_ground_truth(ex["text"])

        base_answer = run_generate(base_model, base_tok, question, args.max_tokens)
        ft_answer = run_generate(ft_model, ft_tok, question, args.max_tokens)

        base_scores = scorer.score(ground_truth, base_answer)
        ft_scores = scorer.score(ground_truth, ft_answer)

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "gt_label": _parse_label(ground_truth),
            "base_model": base_answer,
            "base_label": _parse_label(base_answer),
            "finetuned": ft_answer,
            "ft_label": _parse_label(ft_answer),
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

    avgs = compute_averages(results)
    ft_acc = label_accuracy(results)
    base_acc = label_accuracy([
        {"ground_truth": r["ground_truth"], "finetuned": r["base_model"]}
        for r in results
    ])

    summary = {
        "n_examples": len(results),
        "data_path": args.data,
        "base_rouge1": avgs["base_avg_rouge1"],
        "base_rougeL": avgs["base_avg_rougeL"],
        "ft_rouge1": avgs["ft_avg_rouge1"],
        "ft_rougeL": avgs["ft_avg_rougeL"],
        "label_accuracy_base": base_acc,
        "label_accuracy_finetuned": ft_acc,
    }
    save_summary(SUMMARY_PATH, summary)

    mlflow.set_experiment("mistral-finance-mlx-lora")
    with mlflow.start_run(run_name="eval"):
        mlflow.set_tag("run_type", "eval")
        mlflow.log_metrics({
            "base_avg_rouge1":        avgs["base_avg_rouge1"],
            "base_avg_rougeL":        avgs["base_avg_rougeL"],
            "ft_avg_rouge1":          avgs["ft_avg_rouge1"],
            "ft_avg_rougeL":          avgs["ft_avg_rougeL"],
            "label_accuracy_base":    base_acc,
            "label_accuracy_finetuned": ft_acc,
        })
        mlflow.log_param("num_examples", len(results))
        mlflow.log_param("rouge_l_gate", ROUGE_L_GATE)
        mlflow.log_param("gate_passed", avgs["ft_rougeL_gate_passed"])
        mlflow.log_param("data_path", args.data)
        mlflow.log_artifact(RESULTS_PATH)
        mlflow.log_artifact(SUMMARY_PATH)

    print(f"\n{'Model':<20} {'ROUGE-1':>10} {'ROUGE-L':>10} {'Label Acc':>10}")
    print("-" * 54)
    print(f"{'Base Mistral-7B':<20} {avgs['base_avg_rouge1']:>10.3f} {avgs['base_avg_rougeL']:>10.3f} {base_acc:>10.3f}")
    print(f"{'Fine-tuned':<20} {avgs['ft_avg_rouge1']:>10.3f} {avgs['ft_avg_rougeL']:>10.3f} {ft_acc:>10.3f}")
    gate_status = "PASS" if avgs["ft_rougeL_gate_passed"] else "FAIL"
    print(f"\nROUGE-L gate (≥{ROUGE_L_GATE}): {gate_status}")
    print(f"Full results  → {RESULTS_PATH}")
    print(f"Summary       → {SUMMARY_PATH}")
    print("Metrics logged to MLflow experiment 'mistral-finance-mlx-lora'")
    if args.data != VALID_JSONL:
        print(f"(Evaluated on out-of-domain dataset: {args.data})")

    avg_rougeL, passed = check_gate(results)
    print(f"\nROUGE-L gate (>= {ROUGE_L_THRESHOLD}): avg={avg_rougeL:.3f} {'PASS' if passed else 'FAIL'}")
    if not passed:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
