# Financial Sentiment LLM

Fine-tuned Mistral-7B for financial sentiment classification using QLoRA on Apple Silicon. Served via a local FastAPI endpoint. Trained and evaluated in a weekend.

## Results

| Model | ROUGE-1 | ROUGE-L |
|---|---|---|
| Base Mistral-7B-Instruct-v0.3 | 0.113 | 0.094 |
| Fine-tuned (LoRA) | 0.970 | 0.970 |

The base model can classify sentiment but generates it in its own verbose, inconsistent format. The fine-tuned model reliably produces structured output (`Sentiment: {label}. This statement reflects...`) that matches the target format with near-perfect fidelity.

## Stack

- **Model**: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- **Fine-tuning**: LoRA via [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple Silicon native)
- **Dataset**: [nickmuchi/financial-classification](https://huggingface.co/datasets/nickmuchi/financial-classification) (4,551 train / 506 test)
- **Tracking**: MLflow
- **Serving**: FastAPI + uvicorn

## Model Card

| | |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` (4-bit quantized) |
| Fine-tuning method | LoRA (rank=8, scale=10.0, layers=16) |
| Optimizer | AdamW, lr=5e-5 |
| Training iterations | 1000 |
| Batch size | 2 (grad checkpointing enabled) |
| Hardware | Apple Silicon (Metal) |
| Approx. training time | ~45 min |
| Adapter size | 40MB |

## Quickstart

```bash
# Install deps
make install

# Prepare dataset (downloads ~500KB)
make prepare

# Fine-tune (~45 min on M2/M3 Pro)
make train

# Evaluate base vs fine-tuned
make eval

# Serve locally at http://localhost:8080
make serve
```

## Inference

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "Classify the sentiment: \"Operating margins expanded by 300 basis points.\""}'
```

```json
{
  "answer": "Sentiment: positive. This statement reflects favorable financial conditions.",
  "model_version": "mistral-7b-finance-mlx-lora-v1"
}
```

## Eval Details

ROUGE-L of 0.970 is high because the target format is short and structured. It's the right metric here since correct label + format is what matters. The remaining 3% gap comes from edge cases where the model disagrees with the annotator label (e.g. predicting neutral for ambiguous expansion announcements).

Full per-example results in `eval/results.json` after running `make eval`.

## What I'd Do Next

- **Richer output**: add explanation of *why* the sentiment is positive/negative, not just the label
- **Harder eval**: run on out-of-domain financial news (earnings calls, 10-K filings) to test generalization
- **Quantize the adapter**: merge LoRA weights and re-quantize for faster inference
- **Streaming**: add SSE streaming to the FastAPI endpoint for real-time token output
