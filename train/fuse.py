"""
Fuse a trained LoRA adapter into the base model weights and re-quantize.

Running the adapter at inference adds a small per-layer overhead.  Fusing
bakes the adapter deltas into the base weights so the serving path loads a
single checkpoint with no adapter code path.

Workflow:
  1. mlx_lm fuse  — merge adapter + dequantize  → full-precision merged model
  2. mlx_lm convert -q  — re-quantize merged model back to 4-bit

Usage:
    python train/fuse.py
    python train/fuse.py --adapter ./mistral-finetuned --output ./mistral-fused-4bit
    python train/fuse.py --no-requantize   # fuse only, skip re-quantization
"""
import argparse
import subprocess
import sys
import mlflow

MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
ADAPTER_PATH = "./mistral-finetuned"
FUSED_PATH = "./mistral-fused-fp16"
OUTPUT_PATH = "./mistral-fused-4bit"
Q_BITS = 4


def run_fuse(
    model_id: str,
    adapter_path: str,
    fused_path: str,
    dequantize: bool = True,
) -> None:
    """Merge LoRA adapter weights into the base model.

    Passes --dequantize by default so the fused checkpoint is full-precision
    and safe to re-quantize cleanly in the next step.
    """
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_id,
        "--adapter-path", adapter_path,
        "--save-path", fused_path,
    ]
    if dequantize:
        cmd.append("--dequantize")
    subprocess.run(cmd, check=True)


def run_quantize(input_path: str, output_path: str, q_bits: int = Q_BITS) -> None:
    """Re-quantize a full-precision merged model back to q_bits-bit."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", input_path,
        "--mlx-path", output_path,
        "--quantize",
        "--q-bits", str(q_bits),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA adapter and re-quantize")
    parser.add_argument("--model", default=MODEL_ID, help="Base model ID or path")
    parser.add_argument("--adapter", default=ADAPTER_PATH, help="Trained adapter path")
    parser.add_argument("--fused-path", default=FUSED_PATH, dest="fused_path",
                        help="Intermediate full-precision fused model path")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Re-quantized output path")
    parser.add_argument("--q-bits", type=int, default=Q_BITS, dest="q_bits",
                        help="Bits per weight for re-quantization (default: 4)")
    parser.add_argument("--no-requantize", action="store_true", dest="no_requantize",
                        help="Skip re-quantization; keep fused model in full precision")
    args = parser.parse_args()

    mlflow.set_experiment("mistral-finance-mlx-lora")
    with mlflow.start_run(run_name="fuse"):
        mlflow.set_tag("run_type", "fuse")
        mlflow.log_params({
            "model_id": args.model,
            "adapter_path": args.adapter,
            "fused_path": args.fused_path,
            "output_path": args.output,
            "q_bits": args.q_bits,
            "requantize": not args.no_requantize,
        })

        print(f"Fusing adapter {args.adapter!r} into {args.model!r} → {args.fused_path!r}")
        run_fuse(args.model, args.adapter, args.fused_path, dequantize=True)
        print(f"Fused model saved to {args.fused_path!r}")

        if not args.no_requantize:
            print(f"Re-quantizing {args.fused_path!r} → {args.output!r} ({args.q_bits}-bit)")
            run_quantize(args.fused_path, args.output, args.q_bits)
            print(f"Re-quantized model saved to {args.output!r}")
            mlflow.log_artifact(args.output)
        else:
            mlflow.log_artifact(args.fused_path)

        print("Done. Metrics logged to MLflow experiment 'mistral-finance-mlx-lora'")


if __name__ == "__main__":
    main()
