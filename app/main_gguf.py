"""
CPU-only serving entrypoint for GPU-less cloud hosts (e.g. Cloud Run).
Runs the LoRA fine-tune merged into the base weights and quantized to 4-bit
GGUF via llama.cpp (llama-cpp-python); no GPU or Apple Silicon required.
Same API and web UI as the other entrypoints.
Local Apple Silicon dev uses app/main.py (mlx-lm backend).

MOCK_MODE=true skips loading the GGUF weights for CI / infra validation.
"""
import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Event

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ui import mount_ui
from app.utils import configure_logging, parse_sentiment_explanation, parse_sentiment_label

configure_logging()
logger = logging.getLogger(__name__)

FINETUNED_GGUF = os.getenv("FINETUNED_GGUF", "./models/mistral-7b-finance-Q4_K_M.gguf")
BASE_GGUF = os.getenv("BASE_GGUF", "./models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")
MODEL_VERSION = os.getenv("MODEL_VERSION", "mistral-7b-finance-mlx-lora-v1")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
GENERATION_TIMEOUT_SECONDS = float(os.getenv("GENERATION_TIMEOUT_SECONDS", "120"))
N_CTX = int(os.getenv("N_CTX", "2048"))
# 0 = let llama.cpp pick (all physical cores)
N_THREADS = int(os.getenv("N_THREADS", "0")) or None

model = None
base_model = None
executor = ThreadPoolExecutor(max_workers=1)


class _MockLLM:
    """Stands in for llama_cpp.Llama in MOCK_MODE so no GGUF weights are required."""

    _ANSWER = "Sentiment: positive. This statement reflects favorable financial conditions."
    _TOKENS = ["Sentiment", ":", " positive", ".", " Mock", " response", "."]

    def __call__(self, prompt, max_tokens=256, temperature=0.0, stream=False):
        if stream:
            return iter({"choices": [{"text": tok}]} for tok in self._TOKENS)
        return {"choices": [{"text": self._ANSWER}]}


def _load(gguf_path: str):
    from llama_cpp import Llama

    return Llama(model_path=gguf_path, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)


def _get_base():
    """Lazy-load the base-model GGUF for comparison (adapter=false) requests.

    Only runs inside the single-worker executor, so no lock is needed."""
    global base_model
    if MOCK_MODE:
        return model
    if base_model is None:
        logger.info("Lazy-loading base GGUF %r for adapter=false requests", BASE_GGUF)
        base_model = _load(BASE_GGUF)
    return base_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MOCK_MODE:
        model = _MockLLM()
        logger.info("MOCK_MODE enabled, skipping GGUF load")
        yield
        return

    try:
        if not Path(FINETUNED_GGUF).exists():
            raise FileNotFoundError(f"No fine-tuned GGUF at {FINETUNED_GGUF!r}")
        logger.info("Loading fine-tuned GGUF from %r", FINETUNED_GGUF)
        model = _load(FINETUNED_GGUF)
    except Exception:
        logger.error(
            "Failed to load model (finetuned_gguf=%r, base_gguf=%r)",
            FINETUNED_GGUF, BASE_GGUF, exc_info=True,
        )
        raise
    yield


app = FastAPI(title="Financial Sentiment LLM API (GGUF/CPU)", lifespan=lifespan)
mount_ui(app)


class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(256, gt=0, le=2048)
    adapter: bool = Field(True, description="False generates with the raw base model")

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


# Unlike the other backends, llama.cpp prepends BOS during tokenization, so the
# prompt omits the literal "<s>" the sibling entrypoints include.
def _generate(question: str, max_tokens: int, use_adapter: bool = True) -> str:
    llm = model if use_adapter else _get_base()
    prompt = f"[INST] {question} [/INST]"
    out = llm(prompt, max_tokens=max_tokens, temperature=0.0)
    return out["choices"][0]["text"].strip()


def _stream_into_queue(
    question: str, max_tokens: int, use_adapter: bool, q: Queue, done: Event, request_id: str
):
    prompt = f"[INST] {question} [/INST]"
    start = time.perf_counter()
    logger.info(
        "predict/stream request start request_id=%s adapter=%s max_tokens=%d",
        request_id, use_adapter, max_tokens,
    )
    try:
        llm = model if use_adapter else _get_base()
        for chunk in llm(prompt, max_tokens=max_tokens, temperature=0.0, stream=True):
            q.put(chunk["choices"][0]["text"])
        logger.info(
            "predict/stream request done request_id=%s latency_ms=%.1f",
            request_id, (time.perf_counter() - start) * 1000,
        )
    except Exception:
        logger.error("predict/stream request failed request_id=%s", request_id, exc_info=True)
    finally:
        done.set()


@app.post("/predict", response_model=Response)
async def predict(query: Query):
    if model is None:
        logger.warning("Rejecting /predict: model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    logger.info(
        "predict request start request_id=%s adapter=%s max_tokens=%d",
        request_id, query.adapter, query.max_tokens,
    )
    loop = asyncio.get_event_loop()
    try:
        answer = await asyncio.wait_for(
            loop.run_in_executor(
                executor, _generate, query.question, query.max_tokens, query.adapter
            ),
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "predict request timed out request_id=%s after %.0fs",
            request_id, GENERATION_TIMEOUT_SECONDS,
        )
        raise HTTPException(status_code=504, detail="Inference timed out") from None
    except Exception:
        logger.error("predict request failed request_id=%s", request_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed") from None
    logger.info(
        "predict request done request_id=%s latency_ms=%.1f",
        request_id, (time.perf_counter() - start) * 1000,
    )
    return Response(
        answer=answer,
        label=parse_sentiment_label(answer),
        explanation=parse_sentiment_explanation(answer),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/stream")
async def predict_stream(query: Query):
    """SSE endpoint: streams tokens as they are generated."""
    if model is None:
        logger.warning("Rejecting /predict/stream: model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    q: Queue = Queue()
    done = Event()
    executor.submit(
        _stream_into_queue, query.question, query.max_tokens, query.adapter, q, done, request_id
    )

    async def event_generator():
        loop = asyncio.get_event_loop()
        last_activity = loop.time()
        while not (done.is_set() and q.empty()):
            if loop.time() - last_activity > GENERATION_TIMEOUT_SECONDS:
                logger.error(
                    "Streaming inference timed out for /predict/stream after %.0fs of inactivity",
                    GENERATION_TIMEOUT_SECONDS,
                )
                payload = json.dumps({"error": "Inference timed out", "model_version": MODEL_VERSION})
                yield f"data: {payload}\n\n"
                break
            try:
                token = await loop.run_in_executor(None, q.get, True, 0.05)
                last_activity = loop.time()
                yield f"data: {json.dumps({'token': token, 'model_version': MODEL_VERSION})}\n\n"
            except Empty:
                continue
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/model/info")
async def model_info():
    return {
        "model_id":      FINETUNED_GGUF,
        "adapter_path":  None,  # adapter is merged into the GGUF weights
        "model_version": MODEL_VERSION,
        "model_loaded":  model is not None,
        "merged":        True,
    }


@app.get("/health")
async def health():
    loaded = model is not None
    body = {"status": "ok" if loaded else "unhealthy", "model_loaded": loaded, "model_version": MODEL_VERSION}
    return JSONResponse(content=body, status_code=200 if loaded else 503)
