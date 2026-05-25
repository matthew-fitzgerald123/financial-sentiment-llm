"""
ECS/Linux-compatible serving entrypoint.
Uses HuggingFace transformers + PEFT (works on CPU/GPU/Graviton).
Local Apple Silicon dev uses app/main.py (mlx-lm backend).
"""
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Event

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./mistral-finetuned")
MODEL_VERSION = os.getenv("MODEL_VERSION", "mistral-7b-finance-mlx-lora-v1")

pipeline = None
executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
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
        print(f"Loaded adapter from {ADAPTER_PATH!r}")
    else:
        print(f"No adapter at {ADAPTER_PATH!r} — using base model")

    pipeline = {"model": base, "tokenizer": tokenizer}
    yield


app = FastAPI(title="Financial Sentiment LLM API (ECS)", lifespan=lifespan)


class Query(BaseModel):
    question: str
    max_tokens: int = 256


class Response(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    answer: str
    model_version: str


def _generate(question: str, max_tokens: int) -> str:
    from transformers import pipeline as hf_pipeline
    model = pipeline["model"]
    tokenizer = pipeline["tokenizer"]
    prompt = f"<s>[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("[/INST]")[-1].strip()


def _stream_into_queue(question: str, max_tokens: int, q: Queue, done: Event):
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
        raise HTTPException(status_code=503, detail="Model not loaded")
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(executor, _generate, query.question, query.max_tokens)
    return Response(answer=answer, model_version=MODEL_VERSION)


@app.post("/predict/stream")
async def predict_stream(query: Query):
    if pipeline is None:
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


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None}
