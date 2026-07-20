"""
Tests for app/main_vllm.py (vLLM GPU serving entrypoint).
MOCK_MODE is forced via env so no GPU or model weights are required.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ["MOCK_MODE"] = "true"

from app.main_vllm import app, _build_prompt  # noqa: E402 — env must be set first


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_has_model_version(client):
    r = client.get("/health")
    assert "model_version" in r.json()


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

def test_model_info_fields(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    for key in ("model_id", "adapter_path", "model_version", "model_loaded", "backend"):
        assert key in data, f"missing key: {key}"


def test_model_info_backend_is_vllm(client):
    r = client.get("/model/info")
    assert r.json()["backend"] == "vllm"


def test_model_info_model_loaded(client):
    r = client.get("/model/info")
    assert r.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

def test_predict_returns_200(client):
    r = client.post("/predict", json={"question": "Classify: 'Earnings beat expectations.'"})
    assert r.status_code == 200


def test_predict_response_shape(client):
    r = client.post("/predict", json={"question": "Classify: 'Revenue declined 8%.'", "max_tokens": 64})
    data = r.json()
    for key in ("answer", "label", "explanation", "model_version"):
        assert key in data, f"missing key: {key}"


def test_predict_label_is_positive_in_mock_mode(client):
    r = client.post("/predict", json={"question": "Classify sentiment."})
    assert r.json()["label"] == "positive"


@pytest.mark.parametrize("max_tokens", [0, -1, 2049])
def test_predict_rejects_out_of_range_max_tokens(client, max_tokens):
    """max_tokens must be a positive integer within the allowed cap."""
    r = client.post(
        "/predict",
        json={"question": "Classify: 'Revenue rose 12%.'", "max_tokens": max_tokens},
    )
    assert r.status_code == 422


def test_predict_answer_non_empty(client):
    r = client.post("/predict", json={"question": "Classify this."})
    assert len(r.json()["answer"]) > 0


def test_predict_missing_question(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_empty_question(client):
    r = client.post("/predict", json={"question": ""})
    assert r.status_code == 422


def test_predict_rejects_oversized_question(client):
    """question longer than the allowed cap must be rejected with 422."""
    r = client.post("/predict", json={"question": "x" * 4097})
    assert r.status_code == 422


def test_predict_rejects_blank_question(client):
    """whitespace-only question must be rejected with 422."""
    r = client.post("/predict", json={"question": "   "})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /predict/stream
# ---------------------------------------------------------------------------

def test_predict_stream_200(client):
    r = client.post("/predict/stream", json={"question": "Classify: 'Revenue fell.'"})
    assert r.status_code == 200


def test_predict_stream_content_type(client):
    r = client.post("/predict/stream", json={"question": "Classify."})
    assert "text/event-stream" in r.headers["content-type"]


def test_predict_stream_done_sentinel(client):
    r = client.post("/predict/stream", json={"question": "Classify: 'Revenue fell.'"})
    assert "data: [DONE]" in r.text


def test_predict_stream_token_format(client):
    r = client.post("/predict/stream", json={"question": "Classify: 'EPS beat by 15%."})
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:") and "[DONE]" not in ln]
    assert len(lines) >= 1
    payload = json.loads(lines[0].removeprefix("data: "))
    assert "token" in payload
    assert "model_version" in payload


def test_predict_stream_multiple_tokens(client):
    r = client.post("/predict/stream", json={"question": "Classify this statement."})
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:") and "[DONE]" not in ln]
    assert len(lines) > 1


def test_predict_stream_empty_question(client):
    r = client.post("/predict/stream", json={"question": ""})
    assert r.status_code == 422


def test_predict_stream_missing_question(client):
    r = client.post("/predict/stream", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 503 when engine is None
# ---------------------------------------------------------------------------

def test_predict_503_when_engine_none(monkeypatch):
    import app.main_vllm as m
    monkeypatch.setattr(m, "engine", None)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/predict", json={"question": "Classify this."})
    assert r.status_code == 503


def test_predict_stream_503_when_engine_none(monkeypatch):
    import app.main_vllm as m
    monkeypatch.setattr(m, "engine", None)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/predict/stream", json={"question": "Classify this."})
    assert r.status_code == 503


def test_predict_503_logs_warning(monkeypatch, caplog):
    """A rejected /predict call should be observable in the logs, not silent."""
    import app.main_vllm as m
    monkeypatch.setattr(m, "engine", None)
    c = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level("WARNING", logger="app.main_vllm"):
        c.post("/predict", json={"question": "Classify this."})
    assert "engine not loaded" in caplog.text.lower()


def test_health_503_when_engine_none(monkeypatch):
    """/health must return 503 so the ALB/ECS health check detects a failed engine load."""
    import app.main_vllm as m
    monkeypatch.setattr(m, "engine", None)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/health")
    assert r.status_code == 503
    data = r.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_wraps_in_inst_tags():
    prompt = _build_prompt("What is the sentiment?")
    assert "[INST]" in prompt
    assert "[/INST]" in prompt
    assert "What is the sentiment?" in prompt


def test_build_prompt_starts_with_bos():
    prompt = _build_prompt("Any question")
    assert prompt.startswith("<s>")


def test_build_prompt_question_between_tags():
    question = "Classify this financial statement."
    prompt = _build_prompt(question)
    start = prompt.index("[INST]") + len("[INST]")
    end = prompt.index("[/INST]")
    assert question in prompt[start:end]


# ---------------------------------------------------------------------------
# Per-request observability: request_id + latency logging
# ---------------------------------------------------------------------------

def test_predict_logs_start_and_done_with_request_id(client, caplog):
    """A /predict call (even in MOCK_MODE) must log a start and done line sharing one request_id."""
    with caplog.at_level("INFO", logger="app.main_vllm"):
        client.post("/predict", json={"question": "Classify: 'Revenue rose 12%.'"})

    start_lines = [r for r in caplog.records if "predict request start" in r.message]
    done_lines = [r for r in caplog.records if "predict request done" in r.message]
    assert len(start_lines) == 1
    assert len(done_lines) == 1

    import re
    start_id = re.search(r"request_id=(\S+)", start_lines[0].message).group(1)
    done_id = re.search(r"request_id=(\S+)", done_lines[0].message).group(1)
    assert start_id == done_id
    assert "latency_ms=" in done_lines[0].message


def test_predict_stream_logs_start_and_done_with_request_id(client, caplog):
    """A /predict/stream call (even in MOCK_MODE) must log a start and done line sharing one request_id."""
    with caplog.at_level("INFO", logger="app.main_vllm"):
        client.post("/predict/stream", json={"question": "Classify: 'Revenue fell.'"})

    start_lines = [r for r in caplog.records if "predict/stream request start" in r.message]
    done_lines = [r for r in caplog.records if "predict/stream request done" in r.message]
    assert len(start_lines) == 1
    assert len(done_lines) == 1

    import re
    start_id = re.search(r"request_id=(\S+)", start_lines[0].message).group(1)
    done_id = re.search(r"request_id=(\S+)", done_lines[0].message).group(1)
    assert start_id == done_id
