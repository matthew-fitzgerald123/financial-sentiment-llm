"""
Tests for app/main_vllm.py non-MOCK inference paths.

test_api_vllm.py forces MOCK_MODE=true at module level, so the real async
vLLM paths (engine.generate iteration, delta-token extraction, LoRA request
construction) have no coverage there.  This file patches MOCK_MODE=False and
stubs the vLLM AsyncLLMEngine so those paths can be exercised without a GPU.
"""
import json
import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Register a vLLM stub before any module import so that lazy
# 'from vllm import SamplingParams, LoRARequest' calls inside route handlers
# succeed.  conftest.py only stubs mlx/mlx_lm; vllm must be added here.
_vllm_stub = MagicMock()
sys.modules.setdefault("vllm", _vllm_stub)

# MOCK_MODE must be "true" at import time so the lifespan uses the stub path
# and does not attempt to start a real vLLM engine.
os.environ.setdefault("MOCK_MODE", "true")

from app.main_vllm import app  # noqa: E402  -- env var must be set first


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output(text: str):
    """Minimal fake vLLM RequestOutput with the shape the handlers expect."""
    out = MagicMock()
    out.outputs = [MagicMock(text=text)]
    return out


async def _async_gen(*outputs):
    """Async generator that yields fake vLLM output objects."""
    for o in outputs:
        yield o


def _make_engine(*outputs):
    """Return a mock engine whose .generate() produces a fresh async generator per call."""
    engine = MagicMock()
    engine.generate.side_effect = lambda *a, **kw: _async_gen(*outputs)
    return engine


# ---------------------------------------------------------------------------
# /predict — real (non-mock) path
# ---------------------------------------------------------------------------

def test_predict_real_path_returns_200(monkeypatch):
    answer = "Sentiment: positive. This statement reflects favorable financial conditions."
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(_make_output(answer)))
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post("/predict", json={"question": "Classify: 'Revenue rose 5%.'", "max_tokens": 64})
    assert r.status_code == 200


def test_predict_real_path_response_keys(monkeypatch):
    answer = "Sentiment: positive. Favorable conditions."
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(_make_output(answer)))
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post("/predict", json={"question": "Classify this."})
    for key in ("answer", "label", "explanation", "model_version"):
        assert key in r.json(), f"Missing response key: {key}"


def test_predict_real_path_label_extracted(monkeypatch):
    answer = "Sentiment: negative. This statement reflects unfavorable financial conditions."
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(_make_output(answer)))
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post("/predict", json={"question": "Classify."})
    assert r.json()["label"] == "negative"


def test_predict_real_path_neutral_label(monkeypatch):
    answer = "Sentiment: neutral. This statement reflects neutral financial conditions."
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(_make_output(answer)))
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post("/predict", json={"question": "Classify."})
    assert r.json()["label"] == "neutral"


def test_predict_real_path_uses_last_accumulated_output(monkeypatch):
    """The handler accumulates outputs; /predict must return the final text."""
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(
        _make_output("Sentiment"),
        _make_output("Sentiment: positive"),
        _make_output("Sentiment: positive. Favorable financial conditions."),
    ))
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post("/predict", json={"question": "Classify."})
    assert r.json()["answer"] == "Sentiment: positive. Favorable financial conditions."


def test_predict_real_path_calls_engine_generate_once(monkeypatch):
    """engine.generate must be called exactly once per /predict request."""
    answer = "Sentiment: neutral. Neutral conditions."
    mock_engine = _make_engine(_make_output(answer))
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", mock_engine)
    c = TestClient(app, raise_server_exceptions=True)
    c.post("/predict", json={"question": "Classify."})
    mock_engine.generate.assert_called_once()


def test_predict_real_path_no_lora_request_when_adapter_missing(monkeypatch, tmp_path):
    """engine.generate must receive lora_request=None when adapter path does not exist."""
    answer = "Sentiment: neutral. Neutral conditions."
    missing = str(tmp_path / "no-adapter-here")
    mock_engine = _make_engine(_make_output(answer))
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", mock_engine)
    monkeypatch.setattr("app.main_vllm.ADAPTER_PATH", missing)
    c = TestClient(app, raise_server_exceptions=True)
    c.post("/predict", json={"question": "Classify."})
    call_kwargs = mock_engine.generate.call_args.kwargs
    assert call_kwargs.get("lora_request") is None


def test_predict_real_path_lora_request_set_when_adapter_exists(monkeypatch, tmp_path):
    """engine.generate must receive a non-None lora_request when adapter path exists."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    answer = "Sentiment: positive. Favorable conditions."
    mock_engine = _make_engine(_make_output(answer))
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", mock_engine)
    monkeypatch.setattr("app.main_vllm.ADAPTER_PATH", str(adapter_dir))
    c = TestClient(app, raise_server_exceptions=True)
    c.post("/predict", json={"question": "Classify."})
    call_kwargs = mock_engine.generate.call_args.kwargs
    assert call_kwargs.get("lora_request") is not None


def test_predict_real_path_explanation_extracted(monkeypatch):
    answer = "Sentiment: positive. This statement reflects favorable financial conditions."
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(_make_output(answer)))
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post("/predict", json={"question": "Classify."})
    assert "favorable" in r.json()["explanation"].lower()


def test_predict_real_path_model_version_present(monkeypatch):
    answer = "Sentiment: positive. Favorable conditions."
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(_make_output(answer)))
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post("/predict", json={"question": "Classify."})
    assert len(r.json()["model_version"]) > 0


def test_predict_real_path_prompt_includes_question(monkeypatch):
    """engine.generate must be called with a prompt that contains the question text."""
    answer = "Sentiment: positive. Favorable."
    mock_engine = _make_engine(_make_output(answer))
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", mock_engine)
    c = TestClient(app, raise_server_exceptions=True)
    question = "Classify: 'Operating margins expanded 300bps.'"
    c.post("/predict", json={"question": question})
    call_args = mock_engine.generate.call_args
    prompt = call_args.args[0]
    assert question in prompt
    assert "[INST]" in prompt
    assert "[/INST]" in prompt


# ---------------------------------------------------------------------------
# /predict/stream — delta-token extraction (real path)
# ---------------------------------------------------------------------------

def test_predict_stream_real_path_done_sentinel(monkeypatch):
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(
        _make_output("Sentiment: positive. Favorable.")
    ))
    c = TestClient(app)
    r = c.post("/predict/stream", json={"question": "Classify."})
    assert "data: [DONE]" in r.text


def test_predict_stream_real_path_content_type(monkeypatch):
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(
        _make_output("Sentiment: positive. Favorable.")
    ))
    c = TestClient(app)
    r = c.post("/predict/stream", json={"question": "Classify."})
    assert "text/event-stream" in r.headers["content-type"]


def test_predict_stream_real_path_first_delta_is_initial_text(monkeypatch):
    """First emitted token must be the first chunk of accumulated text (prev_len=0)."""
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(
        _make_output("Sentiment"),
        _make_output("Sentiment: positive"),
        _make_output("Sentiment: positive. Favorable."),
    ))
    c = TestClient(app)
    r = c.post("/predict/stream", json={"question": "Classify."})
    data_lines = [
        ln for ln in r.text.splitlines()
        if ln.startswith("data:") and "[DONE]" not in ln
    ]
    assert len(data_lines) >= 1
    first = json.loads(data_lines[0].removeprefix("data: "))
    assert first["token"] == "Sentiment"


def test_predict_stream_real_path_subsequent_deltas_are_incremental(monkeypatch):
    """Each subsequent token must be the new suffix, not a repeat of prior text."""
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(
        _make_output("Sentiment"),
        _make_output("Sentiment: positive"),
        _make_output("Sentiment: positive. Favorable."),
    ))
    c = TestClient(app)
    r = c.post("/predict/stream", json={"question": "Classify."})
    data_lines = [
        ln for ln in r.text.splitlines()
        if ln.startswith("data:") and "[DONE]" not in ln
    ]
    assert len(data_lines) >= 2
    second = json.loads(data_lines[1].removeprefix("data: "))
    # Second delta is ": positive" (new_text[9:] where prev_len=len("Sentiment")=9)
    assert second["token"] == ": positive"


def test_predict_stream_real_path_no_empty_token_on_duplicate_output(monkeypatch):
    """An empty delta from a duplicate output must not produce a token SSE event."""
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(
        _make_output("Hello"),
        _make_output("Hello"),       # duplicate → delta = "" → must be skipped
        _make_output("Hello world"),
    ))
    c = TestClient(app)
    r = c.post("/predict/stream", json={"question": "Classify."})
    data_lines = [
        ln for ln in r.text.splitlines()
        if ln.startswith("data:") and "[DONE]" not in ln
    ]
    for ln in data_lines:
        payload = json.loads(ln.removeprefix("data: "))
        assert payload["token"] != "", "Empty delta must not be emitted as a token event"


def test_predict_stream_real_path_token_payload_has_required_keys(monkeypatch):
    """Every SSE token event must contain 'token' and 'model_version'."""
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(
        _make_output("Sentiment: positive. Favorable."),
    ))
    c = TestClient(app)
    r = c.post("/predict/stream", json={"question": "Classify."})
    data_lines = [
        ln for ln in r.text.splitlines()
        if ln.startswith("data:") and "[DONE]" not in ln
    ]
    assert len(data_lines) >= 1
    for ln in data_lines:
        payload = json.loads(ln.removeprefix("data: "))
        assert "token" in payload
        assert "model_version" in payload


def test_predict_stream_real_path_all_deltas_concatenate_to_full_text(monkeypatch):
    """Concatenating all emitted token deltas must reconstruct the final model output."""
    full_text = "Sentiment: positive. Favorable financial conditions."
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine(
        _make_output("Sentiment"),
        _make_output("Sentiment: positive"),
        _make_output("Sentiment: positive. Favorable financial conditions."),
    ))
    c = TestClient(app)
    r = c.post("/predict/stream", json={"question": "Classify."})
    data_lines = [
        ln for ln in r.text.splitlines()
        if ln.startswith("data:") and "[DONE]" not in ln
    ]
    combined = "".join(
        json.loads(ln.removeprefix("data: "))["token"] for ln in data_lines
    )
    assert combined == full_text


def test_predict_stream_real_path_empty_question_rejected(monkeypatch):
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr("app.main_vllm.engine", _make_engine())
    c = TestClient(app)
    r = c.post("/predict/stream", json={"question": ""})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# lifespan — engine init failure
# ---------------------------------------------------------------------------

def test_lifespan_logs_and_reraises_on_engine_init_failure(monkeypatch, caplog):
    """An engine-init failure at startup must be logged with context, not swallowed."""
    monkeypatch.setattr("app.main_vllm.MOCK_MODE", False)
    monkeypatch.setattr(
        _vllm_stub.AsyncLLMEngine, "from_engine_args",
        MagicMock(side_effect=RuntimeError("out of memory")),
    )
    with caplog.at_level("ERROR", logger="app.main_vllm"):
        with pytest.raises(RuntimeError, match="out of memory"):
            with TestClient(app):
                pass
    assert "failed to initialize vllm engine" in caplog.text.lower()
