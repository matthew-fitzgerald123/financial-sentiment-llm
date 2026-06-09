"""
Tests for the ECS/Docker FastAPI inference service (app/main_ecs.py).
MOCK_MODE is patched to True so no model weights or GPU are required.
"""
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
    assert "model_version" in data
    assert len(data["answer"]) > 0
    assert data["label"] == "positive"


def test_predict_missing_question(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_stream_done_event(client):
    r = client.post(
        "/predict/stream",
        json={"question": "Classify sentiment: 'Revenue fell.'"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: [DONE]" in r.text
