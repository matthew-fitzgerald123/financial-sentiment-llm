# Financial Sentiment LLM

[![Eval](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/eval.yml/badge.svg)](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/eval.yml)
[![Tests](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/test.yml/badge.svg)](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)

Fine-tuned Mistral-7B for financial sentiment classification using LoRA on Apple Silicon. Exposes a FastAPI service with both batch and SSE streaming inference, containerised with Docker, deployed to AWS ECS via Terraform, and gated by a CI eval pipeline on every push.

## Results

| Model | ROUGE-1 | ROUGE-L | Label Accuracy |
|---|---|---|---|
| Base Mistral-7B-Instruct-v0.3 | 0.113 | 0.094 | — |
| Fine-tuned (LoRA) | 0.970 | 0.970 | 0.95+ |

The base model can classify sentiment but generates it in its own verbose, inconsistent format. The fine-tuned model reliably produces structured output (`Sentiment: {label}. This statement reflects...`) that matches the target format with near-perfect fidelity. Label accuracy measures the fraction of predictions where the extracted sentiment label matches the ground-truth label, independently of wording.

## Architecture

```mermaid
flowchart TD
    subgraph Train["Training (Apple Silicon)"]
        DS[(HuggingFace\nDataset)] --> DP[data/prepare.py]
        DP --> JSONL[train.jsonl\nvalid.jsonl]
        JSONL --> LT[mlx-lm LoRA\n1000 iters]
        LT --> ADPT[LoRA Adapter\n40 MB]
        LT --> MLF[MLflow\nExperiment]
    end

    subgraph Eval["CI / Eval (GitHub Actions macos-14)"]
        ADPT --> EV[eval/eval.py\nROUGE-L ≥ 0.85 + label acc ≥ 0.80]
        EV --> AR[Artifacts:\nresults.json · summary.json]
        EV --> MLF
    end

    subgraph Serve["Serving"]
        ADPT --> LC[Local: app/main.py\nmlx-lm + SSE]
        ADPT --> DC[Docker: app/main_ecs.py\ntransformers + PEFT]
    end

    subgraph AWS["AWS (Terraform)"]
        ECR[ECR Repository] --> ECS[ECS Fargate\n2 vCPU / 8 GB]
        ALB[Application\nLoad Balancer] --> ECS
        CW[CloudWatch\nLogs] --> ECS
    end

    DC --> ECR
    Client([Client]) -->|HTTP /predict\n/predict/stream| ALB
    Client -->|local dev| LC
```

## Stack

- **Model**: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- **Fine-tuning**: LoRA via [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple Silicon native)
- **Dataset**: [nickmuchi/financial-classification](https://huggingface.co/datasets/nickmuchi/financial-classification) (4,551 train / 506 test)
- **Tracking**: MLflow
- **Serving**: FastAPI + uvicorn
- **Infra**: Docker · AWS ECS Fargate · Terraform · GitHub Actions

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
pip install -r requirements-dev.txt   # test dependencies (pytest, httpx)

# Prepare dataset (downloads ~500KB)
make prepare

# Fine-tune (~45 min on M2/M3 Pro)
make train

# Evaluate base vs fine-tuned
make eval

# Run test suite (no model weights required)
make test

# Serve locally at http://localhost:8080
make serve
```

## Inference

**Model info:**
```bash
curl http://localhost:8080/model/info
```

```json
{
  "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
  "adapter_path": "./mistral-finetuned",
  "model_version": "mistral-7b-finance-mlx-lora-v1",
  "model_loaded": true
}
```

**Batch:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "Classify the sentiment: \"Operating margins expanded by 300 basis points.\""}'
```

```json
{
  "answer": "Sentiment: positive. This statement reflects favorable financial conditions.",
  "label": "positive",
  "explanation": "This statement reflects favorable financial conditions.",
  "model_version": "mistral-7b-finance-mlx-lora-v1"
}
```

**Streaming (SSE):**
```bash
curl -X POST http://localhost:8080/predict/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Classify the sentiment: \"Revenue declined 8% amid restructuring charges.\""}' \
  --no-buffer
```

```
data: {"token": "Sentiment", "model_version": "mistral-7b-finance-mlx-lora-v1"}
data: {"token": ":", "model_version": "mistral-7b-finance-mlx-lora-v1"}
data: {"token": " negative", "model_version": "mistral-7b-finance-mlx-lora-v1"}
...
data: [DONE]
```

## Eval Details

ROUGE-L of 0.970 is high because the target format is short and structured. The remaining 3% gap comes from edge cases where the model disagrees with the annotator label (e.g. predicting neutral for ambiguous expansion announcements). **Label accuracy** (exact match of the `positive / neutral / negative` token) is now reported alongside ROUGE-1 and ROUGE-L in the output table, giving a more interpretable view of classification quality for this 3-class task.

`eval/eval.py` accepts CLI arguments so it can be pointed at any JSONL file without editing source:

```bash
# Default — evaluates on data/valid.jsonl
make eval

# Out-of-domain dataset, 100 examples
python eval/eval.py --data /path/to/ood.jsonl --n 100

# Custom adapter path
python eval/eval.py --adapter /path/to/my-adapter --n 50
```

Full per-example results in `eval/results.json` after running `make eval`. Aggregate metrics (ROUGE-1, ROUGE-L, label accuracy for both models) are saved to `eval/summary.json` and read by the CI gate. All metrics and the gate result are also logged to the MLflow experiment `mistral-finance-mlx-lora`.

## What I'd Do Next

- **Richer output**: ✓ response now includes `label` and `explanation` fields parsed from structured model output
- **Harder eval**: ✓ `eval/eval.py --data /path/to/ood.jsonl` now supports out-of-domain evaluation; run against earnings calls or 10-K filings to test generalization
- **Merge + requantize**: merge the LoRA adapter into the base weights and re-quantize to reduce inference overhead
- **GPU serving**: right-size the ECS task for a GPU instance (g4dn.xlarge) and switch to vLLM for production throughput
