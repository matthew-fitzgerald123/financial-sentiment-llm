.PHONY: install prepare train eval eval-ood mlflow serve serve-ecs serve-vllm docker-build docker-build-vllm docker-run-vllm benchmark test

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
	python eval/eval.py --data data/ood_sample.jsonl --n 10

mlflow:
	mlflow ui --host 127.0.0.1 --port 5000

# Local dev — Apple Silicon, mlx-lm backend
serve:
	uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload

# ECS-compatible — transformers + PEFT backend
serve-ecs:
	MOCK_MODE=true uvicorn app.main_ecs:app --host 127.0.0.1 --port 8080 --reload

# GPU serving — vLLM backend (requires CUDA; use MOCK_MODE=true for local smoke-test)
serve-vllm:
	uvicorn app.main_vllm:app --host 127.0.0.1 --port 8080 --reload

test:
	python -m pytest tests/ -v

# CPU/ECS image (transformers + PEFT backend)
docker-build:
	docker build -t financial-sentiment-llm:cpu .

# GPU image (vLLM backend, requires CUDA runtime on the host)
docker-build-vllm:
	docker build -f Dockerfile.vllm -t financial-sentiment-llm:gpu .

# Smoke-test the GPU image locally using MOCK_MODE (no GPU required)
docker-run-vllm:
	docker run --rm -p 8080:8080 -e MOCK_MODE=true financial-sentiment-llm:gpu

# Benchmark LoRA scale vs latency/quality (requires trained adapter)
benchmark:
	python benchmarks/quant_bench.py --examples 20
