"""
LoRA fine-tuning via mlx-lm on Apple Silicon.
Wraps mlx_lm.lora CLI with MLflow for experiment tracking.

Requires data prepared by: python data/prepare.py
Usage: python train/train.py
Expected runtime: ~45-90 min on M2/M3 Pro with 1000 iters.
"""
import sys
import subprocess
import mlflow
import mlx.core as mx

CONFIG = "./lora_config.yaml"
ADAPTER_PATH = "./mistral-finetuned"
MEMORY_LIMIT_GB = 10


def set_memory_limit():
    try:
        limit = MEMORY_LIMIT_GB * 1024 ** 3
        mx.metal.set_memory_limit(limit)
        print(f"Metal memory limit: {MEMORY_LIMIT_GB}GB")
    except AttributeError:
        print("Note: mx.metal.set_memory_limit not available in this mlx version — skipping")


def main():
    set_memory_limit()

    mlflow.set_experiment("mistral-finance-mlx-lora")

    with mlflow.start_run():
        mlflow.log_artifact(CONFIG)

        subprocess.run(
            [sys.executable, "-m", "mlx_lm", "lora", "-c", CONFIG],
            check=True,
        )

        mlflow.log_artifact(ADAPTER_PATH)
        print(f"\nAdapter saved → {ADAPTER_PATH}")


if __name__ == "__main__":
    main()
