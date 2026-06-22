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


# ---------------------------------------------------------------------------
# main() — orchestration (model loading and file I/O are mocked)
# ---------------------------------------------------------------------------

_MOCK_STATS = {"tps_median": 50.0, "tps_mean": 48.0, "rougeL_mean": 0.95}
_MAIN_EXAMPLES = [_make_example()]


def _run_main(results_p, extra_argv=None):
    argv = ["quant_bench.py"] + (extra_argv or [])
    with (
        patch.object(_bench, "load_examples", return_value=_MAIN_EXAMPLES),
        patch.object(_bench, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_bench, "run_benchmark", return_value=_MOCK_STATS),
        patch.object(_bench, "set_adapter_scale"),
        patch.object(_bench, "restore_adapter_scale"),
        patch.object(_bench, "RESULTS_PATH", str(results_p)),
        patch("sys.argv", argv),
    ):
        _bench.main()


def test_main_writes_results_json(tmp_path):
    """main() must write bench_results.json at RESULTS_PATH."""
    results_p = tmp_path / "bench_results.json"
    _run_main(results_p)
    assert results_p.exists()


def test_main_results_json_is_valid_json(tmp_path):
    """The results file must contain valid JSON."""
    results_p = tmp_path / "bench_results.json"
    _run_main(results_p)
    data = json.loads(results_p.read_text())
    assert isinstance(data, list)


def test_main_results_include_baseline_and_sweep(tmp_path):
    """Results must have one baseline entry plus one per SCALE_MULTIPLIER."""
    results_p = tmp_path / "bench_results.json"
    _run_main(results_p)
    results = json.loads(results_p.read_text())
    assert len(results) == 1 + len(_bench.SCALE_MULTIPLIERS)


def test_main_baseline_entry_has_none_scale(tmp_path):
    """The first result entry (baseline, no adapter) must have scale=None."""
    results_p = tmp_path / "bench_results.json"
    _run_main(results_p)
    results = json.loads(results_p.read_text())
    assert results[0]["scale"] is None


def test_main_sweep_entries_have_numeric_scale(tmp_path):
    """All sweep entries after the baseline must have a numeric scale."""
    results_p = tmp_path / "bench_results.json"
    _run_main(results_p)
    results = json.loads(results_p.read_text())
    for entry in results[1:]:
        assert isinstance(entry["scale"], (int, float)), (
            f"Expected numeric scale, got {entry['scale']!r}"
        )


def test_main_each_result_has_required_keys(tmp_path):
    """Every result entry must have config, scale, tps_median, tps_mean, rougeL_mean."""
    results_p = tmp_path / "bench_results.json"
    _run_main(results_p)
    results = json.loads(results_p.read_text())
    for entry in results:
        for key in ("config", "scale", "tps_median", "tps_mean", "rougeL_mean"):
            assert key in entry, f"Result entry missing key: {key!r}"


def test_main_calls_restore_adapter_scale_per_multiplier(tmp_path):
    """restore_adapter_scale must be called exactly once per SCALE_MULTIPLIER."""
    results_p = tmp_path / "bench_results.json"
    argv = ["quant_bench.py"]
    with (
        patch.object(_bench, "load_examples", return_value=_MAIN_EXAMPLES),
        patch.object(_bench, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_bench, "run_benchmark", return_value=_MOCK_STATS),
        patch.object(_bench, "set_adapter_scale"),
        patch.object(_bench, "restore_adapter_scale") as mock_restore,
        patch.object(_bench, "RESULTS_PATH", str(results_p)),
        patch("sys.argv", argv),
    ):
        _bench.main()
    assert mock_restore.call_count == len(_bench.SCALE_MULTIPLIERS)


def test_main_calls_load_for_baseline_and_each_sweep(tmp_path):
    """load() must be called once for the baseline and once per sweep config."""
    results_p = tmp_path / "bench_results.json"
    argv = ["quant_bench.py"]
    with (
        patch.object(_bench, "load_examples", return_value=_MAIN_EXAMPLES),
        patch.object(_bench, "load", return_value=(MagicMock(), MagicMock())) as mock_load,
        patch.object(_bench, "run_benchmark", return_value=_MOCK_STATS),
        patch.object(_bench, "set_adapter_scale"),
        patch.object(_bench, "restore_adapter_scale"),
        patch.object(_bench, "RESULTS_PATH", str(results_p)),
        patch("sys.argv", argv),
    ):
        _bench.main()
    assert mock_load.call_count == 1 + len(_bench.SCALE_MULTIPLIERS)


def test_main_baseline_load_has_no_adapter(tmp_path):
    """The baseline load() call must NOT pass an adapter_path."""
    results_p = tmp_path / "bench_results.json"
    argv = ["quant_bench.py"]
    with (
        patch.object(_bench, "load_examples", return_value=_MAIN_EXAMPLES),
        patch.object(_bench, "load", return_value=(MagicMock(), MagicMock())) as mock_load,
        patch.object(_bench, "run_benchmark", return_value=_MOCK_STATS),
        patch.object(_bench, "set_adapter_scale"),
        patch.object(_bench, "restore_adapter_scale"),
        patch.object(_bench, "RESULTS_PATH", str(results_p)),
        patch("sys.argv", argv),
    ):
        _bench.main()
    baseline_call = mock_load.call_args_list[0]
    assert baseline_call.kwargs.get("adapter_path") is None


def test_main_sweep_loads_use_adapter_path(tmp_path):
    """Each sweep load() call must pass adapter_path=ADAPTER_PATH."""
    results_p = tmp_path / "bench_results.json"
    argv = ["quant_bench.py"]
    with (
        patch.object(_bench, "load_examples", return_value=_MAIN_EXAMPLES),
        patch.object(_bench, "load", return_value=(MagicMock(), MagicMock())) as mock_load,
        patch.object(_bench, "run_benchmark", return_value=_MOCK_STATS),
        patch.object(_bench, "set_adapter_scale"),
        patch.object(_bench, "restore_adapter_scale"),
        patch.object(_bench, "RESULTS_PATH", str(results_p)),
        patch("sys.argv", argv),
    ):
        _bench.main()
    sweep_calls = mock_load.call_args_list[1:]
    for call in sweep_calls:
        assert call.kwargs.get("adapter_path") == _bench.ADAPTER_PATH
