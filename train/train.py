"""
LoRA fine-tuning via mlx-lm on Apple Silicon.
Wraps mlx_lm.lora CLI with MLflow for experiment tracking.

Requires data prepared by: python data/prepare.py
Usage: python train/train.py
Expected runtime: ~45-90 min on M2/M3 Pro with 1000 iters.
"""
import sys
import subprocess
import yaml
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
        print("Note: mx.metal.set_memory_limit not available in this mlx version, skipping")


def load_hparams(config_path: str) -> dict:
    """Extract loggable hyperparameters from a lora_config.yaml file.

    Returns a flat dict suitable for mlflow.log_params().
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    lora = cfg.get("lora_parameters", {})
    return {
        "model_id": cfg.get("model", ""),
        "iters": cfg.get("iters"),
        "batch_size": cfg.get("batch_size"),
        "learning_rate": cfg.get("learning_rate"),
        "lora_layers": cfg.get("num_layers"),
        "lora_rank": lora.get("rank"),
        "lora_scale": lora.get("scale"),
        "lora_dropout": lora.get("dropout"),
        "max_seq_length": cfg.get("max_seq_length"),
        "optimizer": cfg.get("optimizer"),
    }


def main():
    set_memory_limit()

    mlflow.set_experiment("mistral-finance-mlx-lora")

    with mlflow.start_run():
        hparams = load_hparams(CONFIG)
        mlflow.log_params(hparams)
        mlflow.log_artifact(CONFIG)

        subprocess.run(
            [sys.executable, "-m", "mlx_lm", "lora", "-c", CONFIG],
            check=True,
        )

        mlflow.log_artifact(ADAPTER_PATH)
        print(f"\nAdapter saved → {ADAPTER_PATH}")


if __name__ == "__main__":
    main()
