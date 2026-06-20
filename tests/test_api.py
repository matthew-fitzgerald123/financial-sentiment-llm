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


# ---------------------------------------------------------------------------
# Merged model path
# ---------------------------------------------------------------------------

def test_merged_model_loads_without_adapter(tmp_path):
    """When MERGED_MODEL_PATH is set and exists, load() is called with no adapter_path arg."""
    merged_dir = tmp_path / "mistral-merged"
    merged_dir.mkdir()

    with (
        patch("app.main.MERGED_MODEL_PATH", str(merged_dir)),
        patch("app.main.load", return_value=(MagicMock(), MagicMock())) as mock_load,
        patch("app.main.mlx_generate", return_value="Sentiment: positive. Favorable conditions."),
        patch("app.main.stream_generate", return_value=iter([])),
    ):
        from app.main import app
        with TestClient(app):
            pass

    mock_load.assert_called_once_with(str(merged_dir))


def test_model_info_merged_true_when_merged_path_set(tmp_path):
    """model/info returns merged=True when MERGED_MODEL_PATH points to an existing directory."""
    merged_dir = tmp_path / "mistral-merged"
    merged_dir.mkdir()

    with (
        patch("app.main.MERGED_MODEL_PATH", str(merged_dir)),
        patch("app.main.load", return_value=(MagicMock(), MagicMock())),
        patch("app.main.mlx_generate", return_value="Sentiment: positive. Favorable conditions."),
        patch("app.main.stream_generate", return_value=iter([])),
    ):
        from app.main import app
        with TestClient(app) as c:
            r = c.get("/model/info")

    data = r.json()
    assert data["merged"] is True
    assert data["model_id"] == str(merged_dir)
    assert data["adapter_path"] is None


def test_model_info_merged_false_by_default(client):
    """model/info returns merged=False when serving the base model + adapter path."""
    r = client.get("/model/info")
    assert r.json()["merged"] is False


def test_predict_works_with_merged_model(tmp_path):
    """Predictions succeed when the merged model is loaded."""
    merged_dir = tmp_path / "mistral-merged"
    merged_dir.mkdir()

    with (
        patch("app.main.MERGED_MODEL_PATH", str(merged_dir)),
        patch("app.main.load", return_value=(MagicMock(), MagicMock())),
        patch(
            "app.main.mlx_generate",
            return_value="Sentiment: positive. This statement reflects favorable financial conditions.",
        ),
        patch("app.main.stream_generate", return_value=iter([])),
    ):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/predict", json={"question": "Classify: 'Margins expanded 200bps.'"})

    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "positive"
    assert "favorable" in data["explanation"]


def test_merged_path_not_used_when_dir_missing(tmp_path):
    """If MERGED_MODEL_PATH is set but the directory does not exist, falls back to base+adapter."""
    missing_dir = str(tmp_path / "does-not-exist")

    with (
        patch("app.main.MERGED_MODEL_PATH", missing_dir),
        patch("app.main.load", return_value=(MagicMock(), MagicMock())) as mock_load,
        patch("app.main.mlx_generate", return_value="Sentiment: neutral. Neutral conditions."),
        patch("app.main.stream_generate", return_value=iter([])),
    ):
        from app.main import app
        with TestClient(app):
            pass

    # load() must be called with model_id + adapter_path kwargs, not the missing merged path
    call_args = mock_load.call_args
    assert call_args[0][0] != missing_dir


# ---------------------------------------------------------------------------
# 503 guard and health model_version
# ---------------------------------------------------------------------------

def test_health_has_model_version(client):
    """/health response must include model_version."""
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_predict_503_when_model_none(monkeypatch):
    """/predict must return 503 if the model was never loaded."""
    import app.main as m
    from app.main import app as mlx_app
    monkeypatch.setattr(m, "model", None)
    c = TestClient(mlx_app, raise_server_exceptions=False)
    r = c.post("/predict", json={"question": "Classify this."})
    assert r.status_code == 503


def test_predict_stream_503_when_model_none(monkeypatch):
    """/predict/stream must return 503 if the model was never loaded."""
    import app.main as m
    from app.main import app as mlx_app
    monkeypatch.setattr(m, "model", None)
    c = TestClient(mlx_app, raise_server_exceptions=False)
    r = c.post("/predict/stream", json={"question": "Classify this."})
    assert r.status_code == 503
