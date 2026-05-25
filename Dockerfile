FROM python:3.11-slim

WORKDIR /app

# Linux/ECS backend: transformers + PEFT (mlx-lm is Apple Silicon only)
COPY app/requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ ./app/
COPY mistral-finetuned/ ./mistral-finetuned/

ENV BASE_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
ENV ADAPTER_PATH=./mistral-finetuned
ENV MODEL_VERSION=mistral-7b-finance-mlx-lora-v1

EXPOSE 8080

CMD ["uvicorn", "app.main_ecs:app", "--host", "0.0.0.0", "--port", "8080"]
