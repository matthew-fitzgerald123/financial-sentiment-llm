"""
Merge LoRA adapter weights into the base model and optionally re-quantize.

Eliminating the adapter-loading step at serve time reduces inference overhead.
The merged model is loaded directly via mlx_lm.load() without an adapter_path.

Usage:
    # Default: merges ./mistral-finetuned into ./mistral-merged (4-bit quantized)
    python scripts/merge.py

    # Custom paths
    python scripts/merge.py --adapter ./mistral-finetuned --output ./mistral-merged

    # Merge only, skip re-quantization
    python scripts/merge.py --no-quantize

    # 8-bit output
    python scripts/merge.py --bits 8
"""
import argparse
import time
from pathlib import Path

BASE_MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
DEFAULT_ADAPTER = "./mistral-finetuned"
DEFAULT_OUTPUT = "./mistral-merged"
DEFAULT_QUANTIZE_BITS = 4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model weights and re-quantize"
    )
    parser.add_argument(
        "--model",
        default=BASE_MODEL_ID,
        help="Base model ID or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="Path to LoRA adapter directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Directory to save the merged model (default: %(default)s)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip re-quantization; save merged weights as float16",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=DEFAULT_QUANTIZE_BITS,
        choices=[4, 8],
        help="Quantization bit-width for the output model (default: %(default)s)",
    )
    return parser.parse_args(argv)


def validate_inputs(adapter_path: str, output_path: str) -> None:
    """Raise ValueError when the adapter is missing or the output already exists."""
    if not Path(adapter_path).exists():
        raise ValueError(
            f"Adapter not found at {adapter_path!r}. Run 'make train' first."
        )
    adapter_config = Path(adapter_path) / "adapter_config.json"
    if not adapter_config.exists():
        raise ValueError(
            f"adapter_config.json missing in {adapter_path!r}. "
            "Directory exists but does not look like a valid LoRA adapter."
        )
    if Path(output_path).exists():
        raise ValueError(
            f"Output path {output_path!r} already exists. "
            "Remove it or choose a different --output path."
        )


def merge(model_id: str, adapter_path: str, output_path: str) -> None:
    """Fuse LoRA adapter weights into the base model and save to output_path."""
    from mlx_lm import fuse
    fuse(
        model=model_id,
        adapter_path=adapter_path,
        save_path=output_path,
        de_quantize=False,
    )


def quantize(model_path: str, bits: int) -> None:
    """Re-quantize the merged model in place."""
    from mlx_lm import convert
    convert(
        hf_path=model_path,
        mlx_path=model_path,
        quantize=True,
        q_bits=bits,
    )


def main(argv=None):
    args = parse_args(argv)
    validate_inputs(args.adapter, args.output)

    print(f"Merging {args.model!r} + adapter {args.adapter!r} → {args.output!r}")
    t0 = time.perf_counter()
    merge(args.model, args.adapter, args.output)
    print(f"Merge complete in {time.perf_counter() - t0:.1f}s")

    if not args.no_quantize:
        print(f"Re-quantizing merged model to {args.bits}-bit …")
        t1 = time.perf_counter()
        quantize(args.output, args.bits)
        print(f"Quantization complete in {time.perf_counter() - t1:.1f}s")

    print(f"\nDone. Merged model saved to {args.output!r}")
    print(f"Serve with:  mlx_lm.load({args.output!r})")


if __name__ == "__main__":
    main()
