.PHONY: install prepare train eval eval-ood mlflow serve serve-ecs merge benchmark test

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

test:
	python -m pytest tests/ -v

# Merge LoRA adapter into base weights and re-quantize (requires trained adapter)
merge:
	python scripts/merge.py --adapter ./mistral-finetuned --output ./mistral-merged

# Benchmark LoRA scale vs latency/quality (requires trained adapter)
benchmark:
	python benchmarks/quant_bench.py --examples 20
