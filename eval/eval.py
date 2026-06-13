"""
Evaluate fine-tuned adapter vs base Mistral on the test set using ROUGE-L.
Prints a comparison table, saves full results to eval/results.json, and
logs metrics to the shared MLflow experiment.

Usage: python eval/eval.py
"""
import json
from pathlib import Path
from rouge_score import rouge_scorer
import mlflow
from mlx_lm import load, generate

MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
ADAPTER_PATH = "./mistral-finetuned"
VALID_JSONL = "./data/valid.jsonl"
RESULTS_PATH = "./eval/results.json"
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


def run_generate(model, tokenizer, question):
    prompt = f"<s>[INST] {question} [/INST]"
    return generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS)


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

    avgs = compute_averages(results)

    mlflow.set_experiment("mistral-finance-mlx-lora")
    with mlflow.start_run(run_name="eval"):
        mlflow.set_tag("run_type", "eval")
        mlflow.log_metrics({
            "base_avg_rouge1": avgs["base_avg_rouge1"],
            "base_avg_rougeL": avgs["base_avg_rougeL"],
            "ft_avg_rouge1":   avgs["ft_avg_rouge1"],
            "ft_avg_rougeL":   avgs["ft_avg_rougeL"],
        })
        mlflow.log_param("num_examples", len(results))
        mlflow.log_param("rouge_l_gate", ROUGE_L_GATE)
        mlflow.log_param("gate_passed", avgs["ft_rougeL_gate_passed"])
        mlflow.log_artifact(RESULTS_PATH)

    print(f"\n{'Model':<20} {'ROUGE-1':>10} {'ROUGE-L':>10}")
    print("-" * 42)
    print(f"{'Base Mistral-7B':<20} {avgs['base_avg_rouge1']:>10.3f} {avgs['base_avg_rougeL']:>10.3f}")
    print(f"{'Fine-tuned':<20} {avgs['ft_avg_rouge1']:>10.3f} {avgs['ft_avg_rougeL']:>10.3f}")
    gate_status = "PASS" if avgs["ft_rougeL_gate_passed"] else "FAIL"
    print(f"\nROUGE-L gate (≥{ROUGE_L_GATE}): {gate_status}")
    print(f"Full results → {RESULTS_PATH}")
    print("Metrics logged to MLflow experiment 'mistral-finance-mlx-lora'")


if __name__ == "__main__":
    main()
