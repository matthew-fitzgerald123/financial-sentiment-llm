"""
vLLM-based serving entrypoint for GPU deployment on ECS EC2 (g4dn.xlarge).
Provides high-throughput async inference with continuous batching.

Backends by target environment:
  Local Apple Silicon  →  app/main.py       (mlx-lm)
  ECS CPU / Graviton   →  app/main_ecs.py   (transformers + PEFT)
  ECS GPU (g4dn.xlarge)→  this file         (vLLM)

MOCK_MODE=true skips engine init for CI / infra validation.
"""
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils import configure_logging, parse_sentiment_explanation, parse_sentiment_label

configure_logging()
logger = logging.getLogger(__name__)

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./mistral-finetuned")
MODEL_VERSION = os.getenv("MODEL_VERSION", "mistral-7b-finance-mlx-lora-v1")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if MOCK_MODE:
        engine = {"mock": True}
        logger.info("MOCK_MODE enabled, skipping vLLM engine init")
        yield
        return

    from vllm import AsyncLLMEngine, AsyncEngineArgs
    engine_args = AsyncEngineArgs(
        model=BASE_MODEL_ID,
        dtype="auto",
        max_model_len=2048,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield


app = FastAPI(title="Financial Sentiment LLM API (vLLM)", lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(256, gt=0, le=2048)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class Response(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    answer: str
    label: str
    explanation: str
    model_version: str


def _build_prompt(question: str) -> str:
    return f"<s>[INST] {question} [/INST]"


@app.post("/predict", response_model=Response)
async def predict(query: Query):
    if engine is None:
        logger.warning("Rejecting /predict: engine not loaded")
        raise HTTPException(status_code=503, detail="Engine not loaded")
    if MOCK_MODE:
        answer = "Sentiment: positive. This statement reflects favorable financial conditions."
        return Response(
            answer=answer,
            label=parse_sentiment_label(answer),
            explanation=parse_sentiment_explanation(answer),
            model_version=MODEL_VERSION,
        )

    from vllm import SamplingParams, LoRARequest
    params = SamplingParams(temperature=0.0, max_tokens=query.max_tokens)
    lora_request = (
        LoRARequest("finance-lora", 1, ADAPTER_PATH) if os.path.exists(ADAPTER_PATH) else None
    )
    request_id = str(uuid.uuid4())

    answer = ""
    async for output in engine.generate(
        _build_prompt(query.question), params, request_id, lora_request=lora_request
    ):
        answer = output.outputs[0].text

    return Response(
        answer=answer,
        label=parse_sentiment_label(answer),
        explanation=parse_sentiment_explanation(answer),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/stream")
async def predict_stream(query: Query):
    if engine is None:
        logger.warning("Rejecting /predict/stream: engine not loaded")
        raise HTTPException(status_code=503, detail="Engine not loaded")

    if MOCK_MODE:
        async def mock_generator():
            for token in ["Sentiment", ":", " positive", ".", " Mock", " response", "."]:
                yield f"data: {json.dumps({'token': token, 'model_version': MODEL_VERSION})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(mock_generator(), media_type="text/event-stream")

    from vllm import SamplingParams, LoRARequest
    params = SamplingParams(temperature=0.0, max_tokens=query.max_tokens)
    lora_request = (
        LoRARequest("finance-lora", 1, ADAPTER_PATH) if os.path.exists(ADAPTER_PATH) else None
    )
    request_id = str(uuid.uuid4())

    async def event_generator():
        prev_len = 0
        async for output in engine.generate(
            _build_prompt(query.question), params, request_id, lora_request=lora_request
        ):
            new_text = output.outputs[0].text
            token = new_text[prev_len:]
            prev_len = len(new_text)
            if token:
                yield f"data: {json.dumps({'token': token, 'model_version': MODEL_VERSION})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/model/info")
async def model_info():
    return {
        "model_id":      BASE_MODEL_ID,
        "adapter_path":  ADAPTER_PATH,
        "model_version": MODEL_VERSION,
        "model_loaded":  engine is not None,
        "backend":       "vllm",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None, "model_version": MODEL_VERSION}
