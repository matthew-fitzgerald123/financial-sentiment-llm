"""
Tests for the CPU/GGUF FastAPI inference service (app/main_gguf.py).
Model loading is patched out so no GGUF weights are required.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_MOCK_ANSWER = "Sentiment: positive. This statement reflects favorable financial conditions."


class _FakeLLM:
    """Mimics the llama_cpp.Llama callable interface used by _generate/_stream_into_queue."""

    def __call__(self, prompt, max_tokens=256, temperature=0.0, stream=False):
        if stream:
            return iter([{"choices": [{"text": " positive"}]}])
        return {"choices": [{"text": _MOCK_ANSWER}]}


@pytest.fixture()
def client(monkeypatch):
    import app.main_gguf as m
    monkeypatch.setattr(m, "MOCK_MODE", False)
    with (
        patch("app.main_gguf.Path", MagicMock()),
        patch("app.main_gguf._load", return_value=_FakeLLM()),
    ):
        from app.main_gguf import app

        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "model_id" in data
    assert "model_version" in data
    assert data["model_loaded"] is True
    assert data["merged"] is True


def test_predict_returns_answer(client):
    r = client.post(
        "/predict",
        json={"question": "Classify the sentiment: 'Operating margins expanded.'"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "label" in data
    assert "explanation" in data
    assert "model_version" in data
    assert len(data["answer"]) > 0
    assert data["label"] == "positive"
    assert "favorable" in data["explanation"]


def test_predict_missing_question(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_empty_question(client):
    """Empty question string must be rejected with 422."""
    r = client.post("/predict", json={"question": ""})
    assert r.status_code == 422


def test_predict_rejects_blank_question(client):
    """whitespace-only question must be rejected with 422."""
    r = client.post("/predict", json={"question": "   "})
    assert r.status_code == 422


def test_predict_rejects_oversized_question(client):
    """question longer than the allowed cap must be rejected with 422."""
    r = client.post("/predict", json={"question": "x" * 4097})
    assert r.status_code == 422


@pytest.mark.parametrize("max_tokens", [0, -1, 2049])
def test_predict_rejects_out_of_range_max_tokens(client, max_tokens):
    """max_tokens must be a positive integer within the allowed cap."""
    r = client.post(
        "/predict",
        json={"question": "Classify: 'Revenue rose 12%.'", "max_tokens": max_tokens},
    )
    assert r.status_code == 422


def test_predict_stream_done_event(client):
    r = client.post(
        "/predict/stream",
        json={"question": "Classify sentiment: 'Revenue fell.'"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: [DONE]" in r.text


# ---------------------------------------------------------------------------
# Per-request observability: request_id + latency logging, error handling
# ---------------------------------------------------------------------------

def test_predict_logs_start_and_done_with_request_id(client, caplog):
    """A successful /predict call must log a start and done line sharing one request_id."""
    with caplog.at_level("INFO", logger="app.main_gguf"):
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


def test_predict_generation_failure_returns_500_and_logs_error(client, caplog):
    """A generation exception must be logged with the request_id and surfaced as a 500, not crash unhandled."""
    with (
        patch("app.main_gguf._generate", side_effect=RuntimeError("out of memory")),
        caplog.at_level("ERROR", logger="app.main_gguf"),
    ):
        r = client.post("/predict", json={"question": "Classify: 'Revenue rose 12%.'"})

    assert r.status_code == 500
    error_lines = [rec for rec in caplog.records if "predict request failed" in rec.message]
    assert len(error_lines) == 1
    assert error_lines[0].exc_info is not None


def test_predict_stream_logs_start_and_done_with_request_id(client, caplog):
    """A successful /predict/stream call must log a start and done line sharing one request_id."""
    with caplog.at_level("INFO", logger="app.main_gguf"):
        client.post("/predict/stream", json={"question": "Classify: 'Revenue fell.'"})

    start_lines = [r for r in caplog.records if "predict/stream request start" in r.message]
    done_lines = [r for r in caplog.records if "predict/stream request done" in r.message]
    assert len(start_lines) == 1
    assert len(done_lines) == 1

    import re
    start_id = re.search(r"request_id=(\S+)", start_lines[0].message).group(1)
    done_id = re.search(r"request_id=(\S+)", done_lines[0].message).group(1)
    assert start_id == done_id


def test_predict_stream_generation_failure_logs_error(client, caplog):
    """A mid-stream generation failure must be logged, and the stream must still terminate with [DONE]."""
    with (
        patch("app.main_gguf.model", side_effect=RuntimeError("gpu error")),
        caplog.at_level("ERROR", logger="app.main_gguf"),
    ):
        r = client.post("/predict/stream", json={"question": "Classify: 'Revenue fell.'"})

    assert r.status_code == 200
    assert "data: [DONE]" in r.text
    error_lines = [rec for rec in caplog.records if "predict/stream request failed" in rec.message]
    assert len(error_lines) == 1
    assert error_lines[0].exc_info is not None


# ---------------------------------------------------------------------------
# 503 guard, lifespan load failure, adapter=false routing
# ---------------------------------------------------------------------------

def test_health_503_when_model_none(client, monkeypatch):
    """/health must return 503 so Cloud Run's health check detects a failed model load."""
    import app.main_gguf as m
    monkeypatch.setattr(m, "model", None)
    r = client.get("/health")
    assert r.status_code == 503
    data = r.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False


def test_predict_503_when_model_none(client, monkeypatch):
    """/predict must return 503 if the model was never loaded."""
    import app.main_gguf as m
    monkeypatch.setattr(m, "model", None)
    r = client.post("/predict", json={"question": "Classify this."})
    assert r.status_code == 503


def test_predict_stream_503_when_model_none(client, monkeypatch):
    """/predict/stream must return 503 if the model was never loaded."""
    import app.main_gguf as m
    monkeypatch.setattr(m, "model", None)
    r = client.post("/predict/stream", json={"question": "Classify this."})
    assert r.status_code == 503


def test_predict_503_logs_warning(client, monkeypatch, caplog):
    """A rejected /predict call should be observable in the logs, not silent."""
    import app.main_gguf as m
    monkeypatch.setattr(m, "model", None)
    with caplog.at_level("WARNING", logger="app.main_gguf"):
        client.post("/predict", json={"question": "Classify this."})
    assert "model not loaded" in caplog.text.lower()


def test_lifespan_logs_and_reraises_on_load_failure(caplog, monkeypatch):
    """A model-load failure at startup must be logged with context, not swallowed."""
    import app.main_gguf as m
    monkeypatch.setattr(m, "MOCK_MODE", False)
    with (
        patch("app.main_gguf.Path", MagicMock()),
        patch("app.main_gguf._load", side_effect=RuntimeError("weights corrupted")),
    ):
        from app.main_gguf import app as gguf_app
        with caplog.at_level("ERROR", logger="app.main_gguf"):
            with pytest.raises(RuntimeError, match="weights corrupted"):
                with TestClient(gguf_app):
                    pass
    assert "failed to load model" in caplog.text.lower()


def test_predict_with_adapter_false_uses_base_model(client):
    """adapter=false must be accepted and served by the lazily loaded base model."""
    r = client.post(
        "/predict",
        json={"question": "Classify the sentiment: 'Margins expanded.'", "adapter": False},
    )
    assert r.status_code == 200
    assert r.json()["label"] == "positive"


# ---------------------------------------------------------------------------
# Generation timeout
# ---------------------------------------------------------------------------

def test_predict_timeout_returns_504(monkeypatch):
    """A generation call that exceeds GENERATION_TIMEOUT_SECONDS must return 504."""
    import time as _time

    class _SlowLLM:
        def __call__(self, prompt, max_tokens=256, temperature=0.0, stream=False):
            if stream:
                return iter([{"choices": [{"text": " positive"}]}])
            _time.sleep(0.2)
            return {"choices": [{"text": _MOCK_ANSWER}]}

    import app.main_gguf as m
    monkeypatch.setattr(m, "MOCK_MODE", False)
    with (
        patch("app.main_gguf.Path", MagicMock()),
        patch("app.main_gguf._load", return_value=_SlowLLM()),
    ):
        from app.main_gguf import app
        monkeypatch.setattr(m, "GENERATION_TIMEOUT_SECONDS", 0.05)
        with TestClient(app) as c:
            r = c.post("/predict", json={"question": "Classify: 'Revenue rose.'"})

    assert r.status_code == 504
    assert "timed out" in r.json()["detail"].lower()


def test_predict_stream_timeout_emits_error_event(monkeypatch):
    """A streaming call that stalls past GENERATION_TIMEOUT_SECONDS must emit an SSE error."""
    import time as _time

    class _SlowStreamLLM:
        def __call__(self, prompt, max_tokens=256, temperature=0.0, stream=False):
            if not stream:
                return {"choices": [{"text": _MOCK_ANSWER}]}

            def _gen():
                yield {"choices": [{"text": "Sentiment"}]}
                _time.sleep(0.2)
                yield {"choices": [{"text": ": positive"}]}

            return _gen()

    import app.main_gguf as m
    monkeypatch.setattr(m, "MOCK_MODE", False)
    with (
        patch("app.main_gguf.Path", MagicMock()),
        patch("app.main_gguf._load", return_value=_SlowStreamLLM()),
    ):
        from app.main_gguf import app
        monkeypatch.setattr(m, "GENERATION_TIMEOUT_SECONDS", 0.05)
        with TestClient(app) as c:
            r = c.post("/predict/stream", json={"question": "Classify: 'Revenue fell.'"})

    assert r.status_code == 200
    assert '"error"' in r.text
    assert "timed out" in r.text.lower()
    assert "data: [DONE]" in r.text
