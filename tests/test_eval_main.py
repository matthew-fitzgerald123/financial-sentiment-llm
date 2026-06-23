"""
Integration tests for eval/eval.py main().

These tests exercise the full orchestration path — argument parsing,
example loading, inference, metric computation, file I/O, and MLflow
logging — without loading any model weights or Apple Silicon hardware.
They complement the helper-level unit tests in test_eval_utils.py.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "eval_main_module", Path(__file__).parent.parent / "eval" / "eval.py"
)
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_GT = "Sentiment: negative. This statement reflects unfavorable financial conditions."
_SAMPLE_TEXT = f"<s>[INST] Classify the sentiment: 'Revenue fell 8%.' [/INST]{_GT}</s>"

_JUNK = "xyz abc def completely unrelated output"


@pytest.fixture()
def data_file(tmp_path):
    """One-example JSONL file in the format eval.py expects."""
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps({"text": _SAMPLE_TEXT}) + "\n")
    return str(p)


def _mlflow_ctx():
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock())
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# File I/O: results.json
# ---------------------------------------------------------------------------

def test_main_writes_results_json(data_file, tmp_path):
    """main() must write results.json at RESULTS_PATH."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    assert Path(results_p).exists()
    results = json.loads(Path(results_p).read_text())
    assert isinstance(results, list)
    assert len(results) == 1


def test_main_results_json_has_expected_keys(data_file, tmp_path):
    """Each entry in results.json must have the keys that the CI gate and MLflow logging rely on."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    entry = json.loads(Path(results_p).read_text())[0]
    for key in (
        "question", "ground_truth", "base_model", "finetuned",
        "base_rouge1", "base_rougeL", "ft_rouge1", "ft_rougeL",
    ):
        assert key in entry, f"results.json entry missing key: {key}"


# ---------------------------------------------------------------------------
# File I/O: summary.json
# ---------------------------------------------------------------------------

def test_main_writes_summary_json(data_file, tmp_path):
    """main() must write summary.json at SUMMARY_PATH."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    assert Path(summary_p).exists()


def test_main_summary_contains_ci_gate_keys(data_file, tmp_path):
    """summary.json must contain the keys the eval.yml CI gate reads."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    summary = json.loads(Path(summary_p).read_text())
    for key in ("ft_rougeL", "label_accuracy_finetuned", "n_examples", "data_path"):
        assert key in summary, f"CI gate key missing from summary.json: {key}"


def test_main_summary_data_path_matches_arg(data_file, tmp_path):
    """summary['data_path'] must record the --data argument used."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    summary = json.loads(Path(summary_p).read_text())
    assert summary["data_path"] == data_file


def test_main_summary_n_examples_matches_evaluated_count(data_file, tmp_path):
    """summary['n_examples'] must equal the number of examples actually evaluated."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    summary = json.loads(Path(summary_p).read_text())
    assert summary["n_examples"] == 1


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------

def test_main_sets_correct_mlflow_experiment(data_file, tmp_path):
    """main() must target the shared MLflow experiment."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment") as mock_set_exp,
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    mock_set_exp.assert_called_once_with("mistral-finance-mlx-lora")


def test_main_calls_log_metrics_with_rouge_keys(data_file, tmp_path):
    """main() must log ROUGE-1, ROUGE-L, and label accuracy metrics to MLflow."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics") as mock_log_metrics,
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    mock_log_metrics.assert_called_once()
    logged = mock_log_metrics.call_args[0][0]
    for key in ("ft_avg_rouge1", "ft_avg_rougeL", "label_accuracy_finetuned"):
        assert key in logged, f"Expected MLflow metric key missing: {key}"


def test_main_logs_mlflow_artifacts(data_file, tmp_path):
    """main() must log both results.json and summary.json as MLflow artifacts."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact") as mock_log_artifact,
    ):
        _eval.main()

    logged_artifacts = [c[0][0] for c in mock_log_artifact.call_args_list]
    assert results_p in logged_artifacts
    assert summary_p in logged_artifacts


# ---------------------------------------------------------------------------
# CI gate behaviour
# ---------------------------------------------------------------------------

def test_main_gate_does_not_exit_when_rougeL_passes(data_file, tmp_path):
    """main() must not call sys.exit when the average fine-tuned ROUGE-L >= 0.85."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    # Exact ground-truth output → ROUGE-L = 1.0 → gate passes
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()  # must not raise SystemExit


def test_main_gate_exits_one_when_rougeL_fails(data_file, tmp_path):
    """main() must call sys.exit(1) when the average fine-tuned ROUGE-L < 0.85."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    # Junk output → ROUGE-L ≈ 0.0 → gate fails
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_JUNK),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            _eval.main()

    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Argument forwarding
# ---------------------------------------------------------------------------

def test_main_respects_n_arg(tmp_path):
    """--n must limit the number of examples evaluated."""
    # Write a file with 3 examples
    p = tmp_path / "multi.jsonl"
    lines = "\n".join(json.dumps({"text": _SAMPLE_TEXT}) for _ in range(3))
    p.write_text(lines + "\n")
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", str(p), "--n", "2"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    results = json.loads(Path(results_p).read_text())
    assert len(results) == 2


def test_main_calls_load_twice(data_file, tmp_path):
    """main() must load the base model and the fine-tuned adapter as separate calls."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())) as mock_load,
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    assert mock_load.call_count == 2


def test_main_passes_adapter_arg_to_load(data_file, tmp_path):
    """--adapter must be forwarded to the fine-tuned model load() call."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    custom_adapter = "/path/to/my/adapter"
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())) as mock_load,
        patch.object(_eval, "generate", return_value=_GT),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", [
            "eval.py", "--data", data_file, "--n", "1",
            "--adapter", custom_adapter,
        ]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    adapter_calls = [
        c for c in mock_load.call_args_list
        if c.kwargs.get("adapter_path") == custom_adapter
    ]
    assert len(adapter_calls) == 1, "load() must be called once with the custom adapter path"


# ---------------------------------------------------------------------------
# --no-gate flag
# ---------------------------------------------------------------------------

_JUNK = "xyz abc def completely unrelated output"


def test_no_gate_skips_exit_on_failing_rougeL(data_file, tmp_path):
    """--no-gate must prevent sys.exit(1) even when ROUGE-L is below the threshold."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    # Junk output gives ROUGE-L ≈ 0.0 — gate would normally fail
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_JUNK),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1", "--no-gate"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()  # must not raise SystemExit


def test_no_gate_still_writes_results_and_summary(data_file, tmp_path):
    """--no-gate must still write both output files."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_JUNK),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1", "--no-gate"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()

    assert Path(results_p).exists()
    assert Path(summary_p).exists()


def test_gate_still_exits_without_no_gate_flag(data_file, tmp_path):
    """Without --no-gate, sys.exit(1) must still be called when ROUGE-L fails."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_JUNK),
        patch.object(_eval, "RESULTS_PATH", results_p),
        patch.object(_eval, "SUMMARY_PATH", summary_p),
        patch("sys.argv", ["eval.py", "--data", data_file, "--n", "1"]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            _eval.main()

    assert exc_info.value.code == 1
