import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from mlx_lm import generate as mlx_generate
from mlx_lm import load
from pydantic import BaseModel, ConfigDict

MODEL_ID = os.getenv("BASE_MODEL_ID", "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./mistral-finetuned")
MODEL_VERSION = os.getenv("MODEL_VERSION", "mistral-7b-finance-mlx-lora-v1")

model = None
tokenizer = None
executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    adapter = ADAPTER_PATH if Path(ADAPTER_PATH).exists() else None
    if adapter is None:
        print(f"No adapter found at {ADAPTER_PATH!r} — loading base model only")
    model, tokenizer = load(MODEL_ID, adapter_path=adapter)
    yield


app = FastAPI(title="Financial Sentiment LLM API", lifespan=lifespan)


class Query(BaseModel):
    question: str
    max_tokens: int = 256


class Response(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    answer: str
    model_version: str


def _generate(question: str, max_tokens: int) -> str:
    prompt = f"<s>[INST] {question} [/INST]"
    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


@app.post("/predict", response_model=Response)
async def predict(query: Query):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import asyncio
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(executor, _generate, query.question, query.max_tokens)
    return Response(answer=answer, model_version=MODEL_VERSION)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
