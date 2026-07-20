"""
Tests for the web UI (app/ui.py + app/static/index.html).
The UI is a static single-page app, so these run without model weights.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ui import STATIC_DIR, mount_ui


def _client() -> TestClient:
    app = FastAPI()
    mount_ui(app)
    return TestClient(app)


def test_root_serves_html():
    r = _client().get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Financial Sentiment" in r.text


def test_root_excluded_from_openapi_schema():
    app = FastAPI()
    mount_ui(app)
    assert "/" not in app.openapi()["paths"]


def test_index_targets_real_api_endpoints():
    html = (STATIC_DIR / "index.html").read_text()
    assert "/predict/stream" in html
    assert "/health" in html


def test_index_wraps_input_in_training_prompt_template():
    """The UI must send input wrapped in the instruction template the adapter
    was trained on (data/prepare.py), or the model drifts off-format."""
    html = (STATIC_DIR / "index.html").read_text()
    assert "Classify the sentiment of the following financial statement" in html


def test_index_supports_light_and_dark_themes():
    html = (STATIC_DIR / "index.html").read_text()
    assert 'data-theme="light"' in html
    assert "prefers-color-scheme" in html


def test_all_entrypoints_mount_ui():
    """Each serving entrypoint must wire in the UI (checked at source level so
    this does not require transformers/vllm to be importable)."""
    app_dir = STATIC_DIR.parent
    for entrypoint in ("main.py", "main_ecs.py", "main_vllm.py"):
        source = (app_dir / entrypoint).read_text()
        assert "mount_ui(app)" in source, f"{entrypoint} does not mount the UI"
