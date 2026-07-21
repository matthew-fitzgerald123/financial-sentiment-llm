"""
Tests for the ECS/Docker FastAPI inference service (app/main_ecs.py).
MOCK_MODE is patched to True so no model weights or GPU are required.
"""
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_MOCK_ANSWER = "Sentiment: positive. This statement reflects favorable financial conditions."


def _mock_stream(question, max_tokens, use_adapter, q, done, request_id):
    done.set()


@pytest.fixture()
def client():
    with (
        patch("app.main_ecs.MOCK_MODE", True),
        patch("app.main_ecs._generate", return_value=_MOCK_ANSWER),
        patch("app.main_ecs._stream_into_queue", side_effect=_mock_stream),
    ):
        from app.main_ecs import app

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


def test_predict_stream_done_event(client):
    r = client.post(
        "/predict/stream",
        json={"question": "Classify sentiment: 'Revenue fell.'"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: [DONE]" in r.text


# ---------------------------------------------------------------------------
# Generation timeout
# ---------------------------------------------------------------------------

def _slow_generate(question, max_tokens, use_adapter=True):
    time.sleep(0.2)
    return _MOCK_ANSWER


def _slow_stream(question, max_tokens, use_adapter, q, done, request_id):
    q.put("Sentiment:")
    time.sleep(0.2)
    q.put(" positive.")
    done.set()


def test_predict_timeout_returns_504(monkeypatch):
    with (
        patch("app.main_ecs.MOCK_MODE", True),
        patch("app.main_ecs._generate", side_effect=_slow_generate),
        patch("app.main_ecs._stream_into_queue", side_effect=_mock_stream),
    ):
        from app.main_ecs import app
        import app.main_ecs as m
        monkeypatch.setattr(m, "GENERATION_TIMEOUT_SECONDS", 0.05)
        with TestClient(app) as c:
            r = c.post("/predict", json={"question": "Classify: 'Revenue rose.'"})

    assert r.status_code == 504
    assert "timed out" in r.json()["detail"].lower()


def test_predict_stream_timeout_emits_error_event(monkeypatch):
    with (
        patch("app.main_ecs.MOCK_MODE", True),
        patch("app.main_ecs._generate", return_value=_MOCK_ANSWER),
        patch("app.main_ecs._stream_into_queue", side_effect=_slow_stream),
    ):
        from app.main_ecs import app
        import app.main_ecs as m
        monkeypatch.setattr(m, "GENERATION_TIMEOUT_SECONDS", 0.05)
        with TestClient(app) as c:
            r = c.post("/predict/stream", json={"question": "Classify: 'Revenue fell.'"})

    assert r.status_code == 200
    assert '"error"' in r.text
    assert "timed out" in r.text.lower()
    assert "data: [DONE]" in r.text
