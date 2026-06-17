"""
Unit tests for benchmarks/quant_bench.py pure utility functions.
No model weights or Apple Silicon required.
"""
import json
import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the benchmark module (mlx and mlx_lm are stubbed by conftest.py)
# ---------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location(
    "quant_bench", Path(__file__).parent.parent / "benchmarks" / "quant_bench.py"
)
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)


# ---------------------------------------------------------------------------
# parse_example
# ---------------------------------------------------------------------------

_SAMPLE_TEXT = (
    "<s>[INST] Classify the sentiment: 'Revenue declined 8%.' [/INST]"
    "Sentiment: negative. This statement reflects unfavorable financial conditions.</s>"
)


def test_parse_example_extracts_question():
    question, _ = _bench.parse_example(_SAMPLE_TEXT)
    assert "Revenue declined 8%" in question
    assert "[INST]" not in question
    assert "[/INST]" not in question


def test_parse_example_extracts_ground_truth():
    _, gt = _bench.parse_example(_SAMPLE_TEXT)
    assert gt.startswith("Sentiment: negative")
    assert "</s>" not in gt
    assert "[/INST]" not in gt


def test_parse_example_question_is_non_empty():
    question, _ = _bench.parse_example(_SAMPLE_TEXT)
    assert len(question) > 0


def test_parse_example_ground_truth_is_non_empty():
    _, gt = _bench.parse_example(_SAMPLE_TEXT)
    assert len(gt) > 0


@pytest.mark.parametrize("label", ["positive", "negative", "neutral"])
def test_parse_example_preserves_label(label):
    text = (
        f"<s>[INST] Classify this. [/INST]"
        f"Sentiment: {label}. Some explanation.</s>"
    )
    _, gt = _bench.parse_example(text)
    assert label in gt


# ---------------------------------------------------------------------------
# set_adapter_scale / restore_adapter_scale
# ---------------------------------------------------------------------------

def _make_adapter_config(tmp_path: Path, scale: float = 10.0) -> Path:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config = {
        "lora_parameters": {"rank": 8, "dropout": 0.05, "scale": scale},
        "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
    return adapter_dir


def test_set_adapter_scale_writes_scale(tmp_path):
    adapter_dir = _make_adapter_config(tmp_path)
    _bench.set_adapter_scale(str(adapter_dir), 25.0)
    loaded = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert loaded["lora_parameters"]["scale"] == pytest.approx(25.0)


def test_set_adapter_scale_preserves_other_lora_keys(tmp_path):
    adapter_dir = _make_adapter_config(tmp_path)
    _bench.set_adapter_scale(str(adapter_dir), 5.0)
    loaded = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert loaded["lora_parameters"]["rank"] == 8
    assert loaded["lora_parameters"]["dropout"] == pytest.approx(0.05)


def test_set_adapter_scale_preserves_top_level_keys(tmp_path):
    adapter_dir = _make_adapter_config(tmp_path)
    _bench.set_adapter_scale(str(adapter_dir), 5.0)
    loaded = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert "model" in loaded


def test_set_adapter_scale_zero(tmp_path):
    adapter_dir = _make_adapter_config(tmp_path)
    _bench.set_adapter_scale(str(adapter_dir), 0.0)
    loaded = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert loaded["lora_parameters"]["scale"] == pytest.approx(0.0)


def test_restore_adapter_scale_writes_base_scale(tmp_path):
    adapter_dir = _make_adapter_config(tmp_path, scale=25.0)
    _bench.restore_adapter_scale(str(adapter_dir))
    loaded = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert loaded["lora_parameters"]["scale"] == pytest.approx(_bench.BASE_SCALE)


def test_restore_adapter_scale_after_set(tmp_path):
    adapter_dir = _make_adapter_config(tmp_path)
    _bench.set_adapter_scale(str(adapter_dir), 40.0)
    _bench.restore_adapter_scale(str(adapter_dir))
    loaded = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert loaded["lora_parameters"]["scale"] == pytest.approx(_bench.BASE_SCALE)
