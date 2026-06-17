"""
Unit tests for benchmarks/quant_bench.py pure utility functions.
No model weights or Apple Silicon required.
"""
import json
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# load_examples
# ---------------------------------------------------------------------------

_SAMPLE_TEXT = (
    "<s>[INST] Classify the sentiment: 'Revenue declined 8%.' [/INST]"
    "Sentiment: negative. This statement reflects unfavorable financial conditions.</s>"
)


def test_load_examples_returns_requested_count(tmp_path):
    jsonl = tmp_path / "valid.jsonl"
    lines = [json.dumps({"text": _SAMPLE_TEXT}) for _ in range(10)]
    jsonl.write_text("\n".join(lines))
    examples = _bench.load_examples(str(jsonl), 5)
    assert len(examples) == 5


def test_load_examples_does_not_exceed_file_length(tmp_path):
    jsonl = tmp_path / "valid.jsonl"
    lines = [json.dumps({"text": _SAMPLE_TEXT}) for _ in range(3)]
    jsonl.write_text("\n".join(lines))
    examples = _bench.load_examples(str(jsonl), 100)
    assert len(examples) == 3


def test_load_examples_each_entry_has_text_key(tmp_path):
    jsonl = tmp_path / "valid.jsonl"
    lines = [json.dumps({"text": _SAMPLE_TEXT}) for _ in range(4)]
    jsonl.write_text("\n".join(lines))
    examples = _bench.load_examples(str(jsonl), 4)
    for ex in examples:
        assert "text" in ex


def test_load_examples_exactly_one(tmp_path):
    jsonl = tmp_path / "valid.jsonl"
    jsonl.write_text(json.dumps({"text": _SAMPLE_TEXT}) + "\n" + json.dumps({"text": _SAMPLE_TEXT}))
    examples = _bench.load_examples(str(jsonl), 1)
    assert len(examples) == 1


# ---------------------------------------------------------------------------
# run_benchmark (stat-aggregation logic, model/tokenizer mocked)
# ---------------------------------------------------------------------------

def _make_example(label="negative"):
    text = (
        f"<s>[INST] Classify the sentiment: 'Revenue fell.' [/INST]"
        f"Sentiment: {label}. This statement reflects financial conditions.</s>"
    )
    return {"text": text}


def _make_mock_tokenizer(token_count: int):
    tok = MagicMock()
    tok.encode.return_value = list(range(token_count))
    return tok


def test_run_benchmark_returns_expected_keys():
    model = MagicMock()
    tokenizer = _make_mock_tokenizer(10)
    examples = [_make_example()]

    with patch.object(_bench, "generate", return_value="Sentiment: negative. Some explanation."):
        result = _bench.run_benchmark(model, tokenizer, examples, max_tokens=64)

    assert set(result.keys()) == {"tps_median", "tps_mean", "rougeL_mean"}


def test_run_benchmark_single_example_median_equals_mean():
    model = MagicMock()
    tokenizer = _make_mock_tokenizer(20)
    examples = [_make_example()]

    with patch.object(_bench, "generate", return_value="Sentiment: negative. Some explanation."):
        result = _bench.run_benchmark(model, tokenizer, examples, max_tokens=64)

    assert result["tps_median"] == pytest.approx(result["tps_mean"])


def test_run_benchmark_perfect_rougeL_on_exact_match():
    model = MagicMock()
    tokenizer = _make_mock_tokenizer(10)
    ground_truth = "Sentiment: negative. This statement reflects financial conditions."
    text = (
        "<s>[INST] Classify the sentiment: 'Revenue fell.' [/INST]"
        f"{ground_truth}</s>"
    )
    examples = [{"text": text}]

    with patch.object(_bench, "generate", return_value=ground_truth):
        result = _bench.run_benchmark(model, tokenizer, examples, max_tokens=64)

    assert result["rougeL_mean"] == pytest.approx(1.0)


def test_run_benchmark_rougeL_mean_averaged_over_examples():
    model = MagicMock()
    tokenizer = _make_mock_tokenizer(10)
    ground_truth = "Sentiment: negative. This statement reflects financial conditions."
    text = (
        "<s>[INST] Classify. [/INST]"
        f"{ground_truth}</s>"
    )
    examples = [{"text": text}, {"text": text}]
    outputs = [ground_truth, "completely wrong output xyz abc def"]

    with patch.object(_bench, "generate", side_effect=outputs):
        result = _bench.run_benchmark(model, tokenizer, examples, max_tokens=64)

    assert 0.0 < result["rougeL_mean"] < 1.0


def test_run_benchmark_tps_non_negative():
    model = MagicMock()
    tokenizer = _make_mock_tokenizer(8)
    examples = [_make_example()]

    with patch.object(_bench, "generate", return_value="Sentiment: negative. Explanation."):
        result = _bench.run_benchmark(model, tokenizer, examples, max_tokens=64)

    assert result["tps_median"] >= 0.0
    assert result["tps_mean"] >= 0.0


def test_run_benchmark_median_of_odd_count():
    model = MagicMock()
    tokenizer = _make_mock_tokenizer(10)
    examples = [_make_example() for _ in range(3)]

    with patch.object(_bench, "generate", return_value="Sentiment: negative. Explanation."):
        result = _bench.run_benchmark(model, tokenizer, examples, max_tokens=64)

    assert isinstance(result["tps_median"], float)
