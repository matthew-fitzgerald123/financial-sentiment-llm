"""
Tests for the CPU/GGUF FastAPI inference service (app/main_gguf.py).
Model loading is patched out so no GGUF weights are required.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_MOCK_ANSWER = "Sentiment: positive. This statement reflects favorable financial conditions."


def _mock_stream(question, max_tokens, use_adapter, q, done):
    done.set()


@pytest.fixture()
def client():
    with (
        patch("app.main_gguf.Path", MagicMock()),
        patch("app.main_gguf._load", return_value=object()),
        patch("app.main_gguf._generate", return_value=_MOCK_ANSWER),
        patch("app.main_gguf._stream_into_queue", side_effect=_mock_stream),
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


def test_predict_stream_done_event(client):
    r = client.post(
        "/predict/stream",
        json={"question": "Classify sentiment: 'Revenue fell.'"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: [DONE]" in r.text
