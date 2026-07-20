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

# Container-native healthcheck so orchestrators (ECS, plain `docker run`) can
# detect and restart a stuck container independently of the ALB health check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=5).status == 200 else 1)"

CMD ["uvicorn", "app.main_ecs:app", "--host", "0.0.0.0", "--port", "8080"]
