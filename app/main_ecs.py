"""
ECS/Linux-compatible serving entrypoint.
Uses HuggingFace transformers + PEFT (works on CPU/GPU/Graviton).
Local Apple Silicon dev uses app/main.py (mlx-lm backend).
"""
import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Event

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ui import mount_ui
from app.utils import configure_logging, parse_sentiment_explanation, parse_sentiment_label

configure_logging()
logger = logging.getLogger(__name__)

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./mistral-finetuned")
MODEL_VERSION = os.getenv("MODEL_VERSION", "mistral-7b-finance-mlx-lora-v1")

pipeline = None
executor = ThreadPoolExecutor(max_workers=1)


MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

_MOCK_RESPONSE = "Sentiment: positive. This statement reflects favorable financial conditions."


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if MOCK_MODE:
        # Stub for infra validation / CI, no weights downloaded
        pipeline = {"model": None, "tokenizer": None}
        logger.info("MOCK_MODE enabled, skipping model load")
        yield
        return

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

        if Path(ADAPTER_PATH).exists():
            base = PeftModel.from_pretrained(base, ADAPTER_PATH)
            logger.info("Loaded adapter from %r", ADAPTER_PATH)
        else:
            logger.warning("No adapter at %r, using base model", ADAPTER_PATH)
    except Exception:
        logger.error(
            "Failed to load model %r (adapter_path=%r)",
            BASE_MODEL_ID, ADAPTER_PATH, exc_info=True,
        )
        raise

    pipeline = {"model": base, "tokenizer": tokenizer}
    yield


app = FastAPI(title="Financial Sentiment LLM API (ECS)", lifespan=lifespan)
mount_ui(app)


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


def _generate(question: str, max_tokens: int) -> str:
    if MOCK_MODE:
        return _MOCK_RESPONSE
    from transformers import pipeline as hf_pipeline
    model = pipeline["model"]
    tokenizer = pipeline["tokenizer"]
    prompt = f"<s>[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("[/INST]")[-1].strip()


def _stream_into_queue(question: str, max_tokens: int, q: Queue, done: Event):
    if MOCK_MODE:
        for token in _MOCK_RESPONSE.split():
            q.put(token + " ")
        done.set()
        return
    from transformers import TextIteratorStreamer
    import threading
    model = pipeline["model"]
    tokenizer = pipeline["tokenizer"]
    prompt = f"<s>[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, streamer=streamer, do_sample=False)
    t = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    t.start()
    try:
        for token in streamer:
            q.put(token)
    finally:
        t.join()
        done.set()


@app.post("/predict", response_model=Response)
async def predict(query: Query):
    if pipeline is None:
        logger.warning("Rejecting /predict: model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(executor, _generate, query.question, query.max_tokens)
    return Response(
        answer=answer,
        label=parse_sentiment_label(answer),
        explanation=parse_sentiment_explanation(answer),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/stream")
async def predict_stream(query: Query):
    if pipeline is None:
        logger.warning("Rejecting /predict/stream: model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    q: Queue = Queue()
    done = Event()
    executor.submit(_stream_into_queue, query.question, query.max_tokens, q, done)

    async def event_generator():
        loop = asyncio.get_event_loop()
        while not (done.is_set() and q.empty()):
            try:
                token = await loop.run_in_executor(None, q.get, True, 0.05)
                yield f"data: {json.dumps({'token': token, 'model_version': MODEL_VERSION})}\n\n"
            except Empty:
                continue
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/model/info")
async def model_info():
    return {
        "model_id":      BASE_MODEL_ID,
        "adapter_path":  ADAPTER_PATH,
        "model_version": MODEL_VERSION,
        "model_loaded":  pipeline is not None,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None, "model_version": MODEL_VERSION}
