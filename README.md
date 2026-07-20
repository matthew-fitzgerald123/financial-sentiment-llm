# Financial Sentiment LLM

[![Eval](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/eval.yml/badge.svg)](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/eval.yml)
[![Tests](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/test.yml/badge.svg)](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/test.yml)
[![Terraform](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/terraform.yml/badge.svg)](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/terraform.yml)
[![Docker](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/docker.yml/badge.svg)](https://github.com/matthew-fitzgerald123/financial-sentiment-llm/actions/workflows/docker.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)

Fine-tuned Mistral-7B for financial sentiment classification using LoRA on Apple Silicon. Exposes a FastAPI service with both batch and SSE streaming inference plus a built-in streaming web UI, containerised with Docker, deployed to AWS ECS via Terraform, and gated by a CI eval pipeline on every push.

<p align="center">
  <img src="docs/demo.gif" alt="Duel mode: the LoRA fine-tune locks a structured sentiment label in under a second, while base Mistral-7B reads the tone correctly but answers in verbose prose the parser scores as unparseable" width="820" />
  <br />
  <em>Duel mode — same prompt, fine-tune vs raw base model, streamed live.</em>
</p>

## Results

| Model | ROUGE-1 | ROUGE-L | Label Accuracy |
|---|---|---|---|
| Base Mistral-7B-Instruct-v0.3 | 0.113 | 0.094 | n/a |
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
        ADPT --> VL[ECS GPU: app/main_vllm.py\nvLLM AsyncEngine + LoRA]
    end

    subgraph AWS["AWS (Terraform)"]
        ECR[ECR Repository] --> ECS[ECS EC2 g4dn.xlarge\nNVIDIA T4 16 GB VRAM]
        ALB[Application\nLoad Balancer] --> ECS
        CW[CloudWatch\nLogs] --> ECS
        ALARMS[CloudWatch Alarms\nALB health/5xx, ECS CPU/mem] --> SNS[SNS Topic]
    end

    DC --> ECR
    VL --> ECR
    Client([Client]) -->|Web UI at /\n/predict · /predict/stream| ALB
    Client -->|local dev| LC
```

## Stack

- **Model**: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- **Fine-tuning**: LoRA via [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple Silicon native)
- **Dataset**: [nickmuchi/financial-classification](https://huggingface.co/datasets/nickmuchi/financial-classification) (4,551 train / 506 test)
- **Tracking**: MLflow
- **Serving**: FastAPI + uvicorn
- **Infra**: Docker (build + smoke-test gated in CI) · AWS ECS EC2 (g4dn.xlarge, NVIDIA T4) · Terraform (fmt + validate gated in CI) · GitHub Actions
- **Alerting**: CloudWatch alarms on ALB unhealthy hosts, ALB 5xx rate, ECS CPU/memory utilization, and application-level ERROR log lines (via a CloudWatch Logs metric filter on the app's structured logs, e.g. model load failures), all published to an SNS topic (subscribe via the `alarm_email` Terraform variable)
- **Logging**: all three serving entrypoints emit structured logs (timestamp, level, logger, message) via the standard `logging` module; set `LOG_LEVEL` (default `INFO`) to control verbosity

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

# Evaluate on bundled out-of-domain fixture (earnings calls, 10-K filings)
make eval-ood

# Run test suite (no model weights required)
make test

# Merge LoRA adapter into base weights and re-quantize (requires trained adapter)
make merge

# Benchmark LoRA scale vs latency/quality (requires trained adapter)
make benchmark

# Benchmark on bundled OOD fixture (earnings calls, 10-K filings)
make benchmark-ood

# Benchmark a non-default adapter path (mirrors eval.py --adapter)
python benchmarks/quant_bench.py --examples 20 --adapter /path/to/my-adapter

# Serve locally at http://localhost:8080 (Apple Silicon / mlx-lm)
make serve

# Serve merged model (no adapter overhead, run 'make merge' first)
make serve-merged

# Serve ECS-compatible backend (transformers + PEFT; MOCK_MODE avoids loading weights)
make serve-ecs

# Serve with vLLM GPU backend (requires CUDA; use MOCK_MODE=true without a GPU)
MOCK_MODE=true make serve-vllm

# Serve the ECS-compatible backend in Docker (requires a trained adapter)
docker compose up --build

# Check Terraform formatting and validate the infra/ config (no AWS credentials needed)
make terraform-validate
```

## Web UI

Every serving entrypoint serves a zero-dependency single-page UI at `/` (e.g. http://localhost:8080 after `make serve`) with three modes:

- **Classify** — stream one statement's classification token by token, with the label detected mid-stream, the parsed verdict surfaced above the raw stream, and per-request time-to-first-token / total latency / tokens-per-sec.
- **Duel** — send the same prompt to the fine-tune and to the raw base model side by side (`adapter: false` on the API). The base 7B usually reads the tone correctly, but answers in verbose prose the production parser scores as `unknown` every time; the fine-tune emits a locked `Sentiment:` schema in ~17 tokens instead of ~100+. That schema is what makes label accuracy measurable (0.95), eval-gateable in CI, and cheap at scale — the value of the 40 MB adapter, made visible.
- **Tape** — paste several headlines (one per line) and get a per-line label plus an aggregate bullish / bearish / mixed market-mood readout with a sentiment distribution bar.

Input is wrapped client-side in the same instruction template the adapter was trained on (`data/prepare.py`) before being sent, keeping the model on-format. The page follows the system light/dark preference live (no manual toggle), keeps a session history of past classifications, and ships inside the Docker image, so the ECS deployment serves it behind the ALB with no extra infrastructure.

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
  "model_loaded": true,
  "merged": false
}
```

Both `/predict` and `/predict/stream` also accept `"adapter": false` to generate with the raw base model instead of the fine-tune (used by the UI's Duel mode; mlx lazy-loads the base copy on first use, PEFT disables the adapter in place, vLLM drops the LoRA request).

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
# Default: evaluates on data/valid.jsonl
make eval

# Bundled OOD fixture (earnings calls, 10-K filings), no extra data needed
make eval-ood
# Equivalent: python eval/eval.py --data data/ood_sample.jsonl --n 10 --no-gate

# Custom out-of-domain dataset
python eval/eval.py --data /path/to/ood.jsonl --n 100

# Custom adapter path
python eval/eval.py --adapter /path/to/my-adapter --n 50

# Skip the ROUGE-L exit gate (informational / OOD runs where high variance is expected)
python eval/eval.py --data data/ood_sample.jsonl --n 10 --no-gate

# Write OOD results to separate files (avoids overwriting main eval output)
python eval/eval.py \
    --data data/ood_sample.jsonl --n 10 --no-gate \
    --output-results eval/ood_results.json \
    --output-summary eval/ood_summary.json
```

Full per-example results in `eval/results.json` after running `make eval`. Aggregate metrics (ROUGE-1, ROUGE-L, label accuracy for both models) and gate-passed boolean flags (`ft_rougeL_gate_passed`, `label_accuracy_gate_passed`) are saved to `eval/summary.json`; the CI gate reads the pre-computed flags directly from the artifact so threshold logic lives in exactly one place (`eval/eval.py`). All metrics and the gate result are also logged to the MLflow experiment `mistral-finance-mlx-lora`.

`benchmarks/quant_bench.py` logs the same way: tok/s and ROUGE-L for the baseline and every LoRA scale multiplier are recorded as MLflow metrics (one run per `make benchmark` invocation, tagged `run_type=benchmark`), alongside the adapter path, data path, and example/token-count params, with `bench_results.json` attached as a run artifact.

## What I'd Do Next

- **Richer output**: ✓ response now includes `label` and `explanation` fields parsed from structured model output
- **Harder eval**: ✓ `data/ood_sample.jsonl` bundles 10 earnings-call / 10-K examples; `make eval-ood` runs the full OOD evaluation in one command
- **Merge + requantize**: ✓ `scripts/merge.py` fuses the LoRA adapter into the base weights and re-quantizes (`make merge`); `make serve-merged` loads the fused model directly via `MERGED_MODEL_PATH` with no adapter overhead, and `/model/info` reports `merged: true` in that mode
- **GPU serving**: ✓ `app/main_vllm.py` serves via `vllm.AsyncLLMEngine` with LoRA support; Terraform switches the ECS cluster from Fargate to an EC2 Auto Scaling Group of `g4dn.xlarge` GPU instances using the ECS-optimized GPU AMI; run locally with `make serve-vllm` (set `MOCK_MODE=true` without a GPU)
