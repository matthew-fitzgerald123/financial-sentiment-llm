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


@pytest.mark.parametrize("max_tokens", [0, -1, 2049])
def test_predict_rejects_out_of_range_max_tokens(client, max_tokens):
    """max_tokens must be a positive integer within the allowed cap."""
    r = client.post(
        "/predict",
        json={"question": "Classify: 'Revenue rose 12%.'", "max_tokens": max_tokens},
    )
    assert r.status_code == 422


def test_predict_rejects_oversized_question(client):
    """question longer than the allowed cap must be rejected with 422."""
    r = client.post("/predict", json={"question": "x" * 4097})
    assert r.status_code == 422


def test_predict_rejects_blank_question(client):
    """whitespace-only question must be rejected with 422."""
    r = client.post("/predict", json={"question": "   "})
    assert r.status_code == 422


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


def test_predict_503_logs_warning(monkeypatch, caplog):
    """A rejected /predict call should be observable in the logs, not silent."""
    import app.main as m
    from app.main import app as mlx_app
    monkeypatch.setattr(m, "model", None)
    c = TestClient(mlx_app, raise_server_exceptions=False)
    with caplog.at_level("WARNING", logger="app.main"):
        c.post("/predict", json={"question": "Classify this."})
    assert "model not loaded" in caplog.text.lower()


def test_lifespan_logs_and_reraises_on_load_failure(caplog):
    """A model-load failure at startup must be logged with context, not swallowed."""
    with patch("app.main.load", side_effect=RuntimeError("weights corrupted")):
        from app.main import app as mlx_app
        with caplog.at_level("ERROR", logger="app.main"):
            with pytest.raises(RuntimeError, match="weights corrupted"):
                with TestClient(mlx_app):
                    pass
    assert "failed to load model" in caplog.text.lower()


def test_predict_with_adapter_false_uses_base_model(client):
    """adapter=false must be accepted and served by the lazily loaded base model."""
    r = client.post(
        "/predict",
        json={"question": "Classify the sentiment: 'Margins expanded.'", "adapter": False},
    )
    assert r.status_code == 200
    assert "label" in r.json()


# ---------------------------------------------------------------------------
# Per-request observability: request_id + latency logging, error handling
# ---------------------------------------------------------------------------

def test_predict_logs_start_and_done_with_request_id(client, caplog):
    """A successful /predict call must log a start and done line sharing one request_id."""
    with caplog.at_level("INFO", logger="app.main"):
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
        patch("app.main.mlx_generate", side_effect=RuntimeError("out of memory")),
        caplog.at_level("ERROR", logger="app.main"),
    ):
        r = client.post("/predict", json={"question": "Classify: 'Revenue rose 12%.'"})

    assert r.status_code == 500
    error_lines = [rec for rec in caplog.records if "predict request failed" in rec.message]
    assert len(error_lines) == 1
    assert error_lines[0].exc_info is not None


def test_predict_stream_logs_start_and_done_with_request_id(client, caplog):
    """A successful /predict/stream call must log a start and done line sharing one request_id."""

    class _FakeChunk:
        def __init__(self, text):
            self.text = text

    with (
        patch("app.main.stream_generate", return_value=iter([_FakeChunk("positive")])),
        caplog.at_level("INFO", logger="app.main"),
    ):
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
        patch("app.main.stream_generate", side_effect=RuntimeError("gpu error")),
        caplog.at_level("ERROR", logger="app.main"),
    ):
        r = client.post("/predict/stream", json={"question": "Classify: 'Revenue fell.'"})

    assert r.status_code == 200
    assert "data: [DONE]" in r.text
    error_lines = [rec for rec in caplog.records if "predict/stream request failed" in rec.message]
    assert len(error_lines) == 1
    assert error_lines[0].exc_info is not None


def test_health_503_when_model_none(monkeypatch):
    """/health must return 503 so the ALB/ECS health check detects a failed model load."""
    import app.main as m
    from app.main import app as mlx_app
    monkeypatch.setattr(m, "model", None)
    c = TestClient(mlx_app, raise_server_exceptions=False)
    r = c.get("/health")
    assert r.status_code == 503
    data = r.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# Generation timeout
# ---------------------------------------------------------------------------

def test_predict_timeout_returns_504(monkeypatch):
    """A generation call that exceeds GENERATION_TIMEOUT_SECONDS must return 504."""
    import time as _time

    def _slow_generate(*args, **kwargs):
        _time.sleep(0.2)
        return "Sentiment: positive. Favorable conditions."

    with (
        patch("app.main.load", return_value=(MagicMock(), MagicMock())),
        patch("app.main.mlx_generate", side_effect=_slow_generate),
        patch("app.main.stream_generate", return_value=iter([])),
    ):
        from app.main import app
        import app.main as m
        monkeypatch.setattr(m, "GENERATION_TIMEOUT_SECONDS", 0.05)
        with TestClient(app) as c:
            r = c.post("/predict", json={"question": "Classify: 'Revenue rose.'"})

    assert r.status_code == 504
    assert "timed out" in r.json()["detail"].lower()


def test_predict_stream_timeout_emits_error_event(monkeypatch):
    """A streaming call that stalls past GENERATION_TIMEOUT_SECONDS must emit an SSE error."""
    import time as _time

    class _Chunk:
        def __init__(self, text):
            self.text = text

    def _slow_stream(*args, **kwargs):
        yield _Chunk("Sentiment")
        _time.sleep(0.2)
        yield _Chunk(": positive")

    with (
        patch("app.main.load", return_value=(MagicMock(), MagicMock())),
        patch("app.main.mlx_generate", return_value="Sentiment: positive."),
        patch("app.main.stream_generate", side_effect=_slow_stream),
    ):
        from app.main import app
        import app.main as m
        monkeypatch.setattr(m, "GENERATION_TIMEOUT_SECONDS", 0.05)
        with TestClient(app) as c:
            r = c.post("/predict/stream", json={"question": "Classify: 'Revenue fell.'"})

    assert r.status_code == 200
    assert '"error"' in r.text
    assert "timed out" in r.text.lower()
    assert "data: [DONE]" in r.text
