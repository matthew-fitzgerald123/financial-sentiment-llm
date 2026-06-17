"""
Tests for the ECS/Docker FastAPI inference service (app/main_ecs.py).
MOCK_MODE is patched to True so no model weights or GPU are required.
"""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_MOCK_ANSWER = "Sentiment: positive. This statement reflects favorable financial conditions."


def _mock_stream(question, max_tokens, q, done):
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


def test_predict_respects_max_tokens(client):
    """max_tokens field is accepted without error."""
    r = client.post(
        "/predict",
        json={"question": "Classify: 'Revenue rose 12%.'", "max_tokens": 64},
    )
    assert r.status_code == 200
    assert "answer" in r.json()


def test_predict_stream_done_event(client):
    r = client.post(
        "/predict/stream",
        json={"question": "Classify sentiment: 'Revenue fell.'"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: [DONE]" in r.text


def test_predict_stream_empty_question(client):
    """Empty question string must be rejected on the streaming endpoint too."""
    r = client.post("/predict/stream", json={"question": ""})
    assert r.status_code == 422


def test_predict_stream_token_format(client):
    """Tokens emitted before [DONE] must be valid JSON with 'token' and 'model_version'."""

    def _emit_one_token(question, max_tokens, q, done):
        q.put("positive")
        done.set()

    with patch("app.main_ecs._stream_into_queue", side_effect=_emit_one_token):
        r = client.post(
            "/predict/stream",
            json={"question": "Classify: 'EPS beat estimates by 15%.'"},
        )

    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:") and "[DONE]" not in ln]
    assert len(lines) >= 1
    payload = json.loads(lines[0].removeprefix("data: "))
    assert "token" in payload
    assert "model_version" in payload
