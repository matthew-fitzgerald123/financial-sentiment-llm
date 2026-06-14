"""
Tests for benchmarks/quant_bench.py helper functions.
No GPU or model weights are required — mlx_lm is stubbed by conftest.py
and inference is patched with controlled return values.
"""
import json
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_spec = importlib.util.spec_from_file_location(
    "bench_module", Path(__file__).parent.parent / "benchmarks" / "quant_bench.py"
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

_SAMPLE_TEXT = (
    "<s>[INST] Classify the sentiment: 'Revenue declined 8%.' [/INST]"
    "Sentiment: negative. This statement reflects unfavorable financial conditions.</s>"
)
_CANNED_ANSWER = "Sentiment: negative. This statement reflects unfavorable financial conditions."


# ---------------------------------------------------------------------------
# load_examples
# ---------------------------------------------------------------------------

def test_load_examples_returns_requested_count(tmp_path):
    jsonl = tmp_path / "valid.jsonl"
    lines = [json.dumps({"text": _SAMPLE_TEXT}) for _ in range(10)]
    jsonl.write_text("\n".join(lines))
    examples = bench.load_examples(str(jsonl), 5)
    assert len(examples) == 5


def test_load_examples_does_not_exceed_file_size(tmp_path):
    jsonl = tmp_path / "valid.jsonl"
    lines = [json.dumps({"text": _SAMPLE_TEXT}) for _ in range(3)]
    jsonl.write_text("\n".join(lines))
    examples = bench.load_examples(str(jsonl), 100)
    assert len(examples) == 3


def test_load_examples_each_entry_has_text_key(tmp_path):
    jsonl = tmp_path / "valid.jsonl"
    jsonl.write_text(json.dumps({"text": _SAMPLE_TEXT}) + "\n")
    examples = bench.load_examples(str(jsonl), 1)
    assert "text" in examples[0]


# ---------------------------------------------------------------------------
# parse_example
# ---------------------------------------------------------------------------

def test_parse_example_extracts_question():
    question, _ = bench.parse_example(_SAMPLE_TEXT)
    assert "Revenue declined 8%" in question
    assert "[INST]" not in question
    assert "[/INST]" not in question


def test_parse_example_extracts_ground_truth():
    _, ground_truth = bench.parse_example(_SAMPLE_TEXT)
    assert ground_truth.startswith("Sentiment: negative")
    assert "</s>" not in ground_truth
    assert "[/INST]" not in ground_truth


def test_parse_example_returns_tuple_of_two():
    result = bench.parse_example(_SAMPLE_TEXT)
    assert len(result) == 2


def test_parse_example_strips_whitespace():
    question, ground_truth = bench.parse_example(_SAMPLE_TEXT)
    assert question == question.strip()
    assert ground_truth == ground_truth.strip()


# ---------------------------------------------------------------------------
# set_adapter_scale / restore_adapter_scale
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter_dir(tmp_path):
    config = {
        "lora_parameters": {"rank": 8, "dropout": 0.05, "scale": 10.0},
        "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_set_adapter_scale_writes_new_scale(adapter_dir):
    bench.set_adapter_scale(adapter_dir, 5.0)
    config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    assert config["lora_parameters"]["scale"] == pytest.approx(5.0)


def test_set_adapter_scale_preserves_other_fields(adapter_dir):
    bench.set_adapter_scale(adapter_dir, 20.0)
    config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    assert config["lora_parameters"]["rank"] == 8
    assert config["lora_parameters"]["dropout"] == pytest.approx(0.05)
    assert config["model"] == "mlx-community/Mistral-7B-Instruct-v0.3-4bit"


def test_restore_adapter_scale_resets_to_base(adapter_dir):
    bench.set_adapter_scale(adapter_dir, 99.0)
    bench.restore_adapter_scale(adapter_dir)
    config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    assert config["lora_parameters"]["scale"] == pytest.approx(bench.BASE_SCALE)


def test_set_adapter_scale_multiple_times(adapter_dir):
    for scale in [2.5, 5.0, 1.0]:
        bench.set_adapter_scale(adapter_dir, scale)
        config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
        assert config["lora_parameters"]["scale"] == pytest.approx(scale)


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(token_count: int = 10) -> MagicMock:
    tok = MagicMock()
    tok.encode.return_value = list(range(token_count))
    return tok


def _make_examples(n: int = 3) -> list[dict]:
    return [{"text": _SAMPLE_TEXT} for _ in range(n)]


def test_run_benchmark_returns_expected_keys():
    mock_model = MagicMock()
    mock_tokenizer = _make_mock_tokenizer()
    with patch.object(bench, "generate", return_value=_CANNED_ANSWER):
        stats = bench.run_benchmark(mock_model, mock_tokenizer, _make_examples(2), 64)
    assert set(stats.keys()) == {"tps_median", "tps_mean", "rougeL_mean"}


def test_run_benchmark_perfect_rouge_on_exact_match():
    mock_model = MagicMock()
    mock_tokenizer = _make_mock_tokenizer()
    with patch.object(bench, "generate", return_value=_CANNED_ANSWER):
        stats = bench.run_benchmark(mock_model, mock_tokenizer, _make_examples(3), 64)
    assert stats["rougeL_mean"] == pytest.approx(1.0)


def test_run_benchmark_tps_values_are_positive():
    mock_model = MagicMock()
    mock_tokenizer = _make_mock_tokenizer(token_count=20)
    with patch.object(bench, "generate", return_value=_CANNED_ANSWER):
        stats = bench.run_benchmark(mock_model, mock_tokenizer, _make_examples(4), 64)
    assert stats["tps_median"] > 0
    assert stats["tps_mean"] > 0


def test_run_benchmark_median_within_mean_bounds():
    mock_model = MagicMock()
    mock_tokenizer = _make_mock_tokenizer(token_count=10)
    with patch.object(bench, "generate", return_value=_CANNED_ANSWER):
        stats = bench.run_benchmark(mock_model, mock_tokenizer, _make_examples(5), 64)
    assert stats["tps_median"] > 0
    assert stats["tps_mean"] > 0


def test_run_benchmark_zero_rouge_on_mismatch():
    mock_model = MagicMock()
    mock_tokenizer = _make_mock_tokenizer()
    with patch.object(bench, "generate", return_value="xyz foo bar baz qux"):
        stats = bench.run_benchmark(mock_model, mock_tokenizer, _make_examples(2), 64)
    assert stats["rougeL_mean"] == pytest.approx(0.0)


def test_run_benchmark_single_example():
    mock_model = MagicMock()
    mock_tokenizer = _make_mock_tokenizer(token_count=8)
    with patch.object(bench, "generate", return_value=_CANNED_ANSWER):
        stats = bench.run_benchmark(mock_model, mock_tokenizer, _make_examples(1), 64)
    assert stats["tps_median"] == stats["tps_mean"]
