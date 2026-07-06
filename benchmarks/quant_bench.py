"""
Benchmark inference latency and ROUGE-L quality across LoRA scale
multipliers and quantization levels.

LoRA scale controls how strongly the fine-tuned adapter influences
output. We sweep scale multipliers [0.25, 0.5, 1.0, 2.0, 4.0] applied
on top of the trained adapter (base scale = 10.0), plus a no-adapter
baseline. For each configuration we record:
  - tokens/sec  (median over NUM_EXAMPLES runs)
  - ROUGE-L     (average over NUM_EXAMPLES examples)

Usage:
    python benchmarks/quant_bench.py [--examples N] [--max-tokens N] [--output PATH] [--adapter PATH] [--data PATH]

Results saved to benchmarks/bench_results.json, printed as a table, and
logged to the MLflow experiment 'mistral-finance-mlx-lora'.
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import mlx.core as mx
from mlx_lm import generate, load
from rouge_score import rouge_scorer

ADAPTER_PATH = "./mistral-finetuned"
MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
VALID_JSONL = "./data/valid.jsonl"
RESULTS_PATH = "./benchmarks/bench_results.json"
BASE_SCALE = 10.0

SCALE_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]


def load_examples(path, n):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def parse_example(text):
    question = text.split("[INST]")[1].split("[/INST]")[0].strip()
    ground_truth = text.split("[/INST]")[1].replace("</s>", "").strip()
    return question, ground_truth


def set_adapter_scale(adapter_dir: str, scale: float):
    """Patch adapter_config.json scale in-memory by writing a temp copy."""
    config_path = Path(adapter_dir) / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["lora_parameters"]["scale"] = scale
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def restore_adapter_scale(adapter_dir: str):
    set_adapter_scale(adapter_dir, BASE_SCALE)


def run_benchmark(model, tokenizer, examples, max_tokens):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    latencies = []
    rouges = []

    for ex in examples:
        question, ground_truth = parse_example(ex["text"])
        prompt = f"<s>[INST] {question} [/INST]"

        t0 = time.perf_counter()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        elapsed = time.perf_counter() - t0

        token_count = len(tokenizer.encode(output))
        tps = token_count / elapsed if elapsed > 0 else 0

        score = scorer.score(ground_truth, output)["rougeL"].fmeasure

        latencies.append(tps)
        rouges.append(score)

    return {
        "tps_median": sorted(latencies)[len(latencies) // 2],
        "tps_mean": sum(latencies) / len(latencies),
        "rougeL_mean": sum(rouges) / len(rouges),
    }


def config_metric_prefix(scale):
    """Turn a result's scale into an MLflow-metric-safe key prefix."""
    if scale is None:
        return "baseline"
    return f"scale_{scale:.2f}".replace(".", "_")


def log_to_mlflow(results, args):
    mlflow.set_experiment("mistral-finance-mlx-lora")
    with mlflow.start_run(run_name="benchmark"):
        mlflow.set_tag("run_type", "benchmark")
        mlflow.log_param("num_examples", args.examples)
        mlflow.log_param("max_tokens", args.max_tokens)
        mlflow.log_param("adapter_path", args.adapter)
        mlflow.log_param("data_path", args.data)

        metrics = {}
        for r in results:
            prefix = config_metric_prefix(r["scale"])
            metrics[f"{prefix}_tps_median"] = r["tps_median"]
            metrics[f"{prefix}_tps_mean"] = r["tps_mean"]
            metrics[f"{prefix}_rougeL_mean"] = r["rougeL_mean"]
        mlflow.log_metrics(metrics)

        mlflow.log_artifact(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--output",
        default=RESULTS_PATH,
        help="Path to write benchmark results JSON (default: %(default)s)",
    )
    parser.add_argument("--adapter", default=ADAPTER_PATH, help="Path to LoRA adapter directory")
    parser.add_argument("--data", default=VALID_JSONL, help="JSONL file to benchmark on")
    args = parser.parse_args()

    examples = load_examples(args.data, args.examples)
    results = []

    # --- Baseline: no adapter ---
    print("Loading base model (no adapter)...")
    model, tokenizer = load(MODEL_ID)
    print(f"  Running {args.examples} examples...")
    stats = run_benchmark(model, tokenizer, examples, args.max_tokens)
    results.append({"config": "base (no adapter)", "scale": None, **stats})
    print(f"  tps={stats['tps_median']:.1f}  rougeL={stats['rougeL_mean']:.3f}")
    del model, tokenizer
    mx.metal.clear_cache()

    # --- Sweep LoRA scale multipliers ---
    for mult in SCALE_MULTIPLIERS:
        effective_scale = BASE_SCALE * mult
        label = f"LoRA scale={effective_scale:.1f} ({mult}x)"
        print(f"\nLoading adapter at scale {effective_scale:.1f} ({mult}x base)...")

        set_adapter_scale(args.adapter, effective_scale)
        try:
            model, tokenizer = load(MODEL_ID, adapter_path=args.adapter)
            print(f"  Running {args.examples} examples...")
            stats = run_benchmark(model, tokenizer, examples, args.max_tokens)
            results.append({"config": label, "scale": effective_scale, **stats})
            print(f"  tps={stats['tps_median']:.1f}  rougeL={stats['rougeL_mean']:.3f}")
            del model, tokenizer
            mx.metal.clear_cache()
        finally:
            restore_adapter_scale(args.adapter)

    # --- Save results ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    log_to_mlflow(results, args)

    # --- Print table ---
    print(f"\n{'Configuration':<30} {'Scale':>8} {'Tok/s':>8} {'ROUGE-L':>8}")
    print("-" * 58)
    for r in results:
        scale_str = f"{r['scale']:.1f}" if r["scale"] is not None else "-"
        print(f"{r['config']:<30} {scale_str:>8} {r['tps_median']:>8.1f} {r['rougeL_mean']:>8.3f}")

    print(f"\nFull results → {args.output}")
    print("Metrics logged to MLflow experiment 'mistral-finance-mlx-lora'")


if __name__ == "__main__":
    main()
