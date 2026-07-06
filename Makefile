.PHONY: install prepare train eval eval-ood mlflow serve serve-merged serve-ecs serve-vllm merge benchmark benchmark-ood test terraform-validate

install:
	pip install -r requirements.txt
	pip install -r app/requirements.txt

prepare:
	python data/prepare.py

train:
	caffeinate -i python train/train.py

eval:
	python eval/eval.py

eval-ood:
	python eval/eval.py --data data/ood_sample.jsonl --n 10 --no-gate

mlflow:
	mlflow ui --host 127.0.0.1 --port 5000

# Local dev — Apple Silicon, mlx-lm backend
serve:
	uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload

# Serve merged model (no adapter overhead) — run 'make merge' first
serve-merged:
	MERGED_MODEL_PATH=./mistral-merged uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload

# ECS-compatible — transformers + PEFT backend
serve-ecs:
	MOCK_MODE=true uvicorn app.main_ecs:app --host 127.0.0.1 --port 8080 --reload

# GPU serving — vLLM backend (requires CUDA; use MOCK_MODE=true for local smoke-test)
serve-vllm:
	uvicorn app.main_vllm:app --host 127.0.0.1 --port 8080 --reload

test:
	python -m pytest tests/ -v

# Merge LoRA adapter into base weights and re-quantize (requires trained adapter)
merge:
	python scripts/merge.py --adapter ./mistral-finetuned --output ./mistral-merged

# Benchmark LoRA scale vs latency/quality (requires trained adapter)
benchmark:
	python benchmarks/quant_bench.py --examples 20

# Benchmark on bundled OOD fixture (earnings calls, 10-K filings) — saves to a separate file
benchmark-ood:
	python benchmarks/quant_bench.py --data data/ood_sample.jsonl --examples 10 --output benchmarks/bench_results_ood.json

# Check Terraform formatting and validate config (no AWS credentials required)
terraform-validate:
	cd infra && terraform fmt -check -recursive -diff
	cd infra && terraform init -backend=false -input=false
	cd infra && terraform validate
