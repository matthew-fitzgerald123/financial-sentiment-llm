"""
Tests for the FastAPI inference service (app/main.py).
The MLX model is mocked so these run without GPU or model weights.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Return a TestClient with the MLX model mocked out."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with (
        patch("app.main.load", return_value=(mock_model, mock_tokenizer)),
        patch(
            "app.main.mlx_generate",
            return_value="Sentiment: positive. This statement reflects favorable financial conditions.",
        ),
        patch("app.main.stream_generate", return_value=iter([])),
    ):
        # Import inside the patch context so lifespan runs with mocks
        from app.main import app

        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


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
    """Streaming endpoint should always emit the [DONE] sentinel."""
    with patch("app.main.stream_generate", return_value=iter([])):
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
    import json as _json

    class _FakeChunk:
        def __init__(self, text):
            self.text = text

    with patch("app.main.stream_generate", return_value=iter([_FakeChunk("positive")])):
        r = client.post(
            "/predict/stream",
            json={"question": "Classify: 'EPS beat estimates by 15%.'"},
        )

    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:") and "[DONE]" not in ln]
    assert len(lines) >= 1
    payload = _json.loads(lines[0].removeprefix("data: "))
    assert "token" in payload
    assert "model_version" in payload
