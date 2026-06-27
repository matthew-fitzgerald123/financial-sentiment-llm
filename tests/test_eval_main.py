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

# Same structure and wording as _GT but with the wrong label — ROUGE-L will be
# high (> 0.85) because most tokens match, but label accuracy will be 0.
_WRONG_LABEL = "Sentiment: positive. This statement reflects unfavorable financial conditions."

# Correct label but completely different wording — label accuracy is 1.0
# (label "negative" matches) but ROUGE-L is low because the explanation
# shares almost no tokens with the ground-truth explanation.
_CORRECT_LABEL_WRONG_WORDING = "Sentiment: negative. Lorem ipsum dolor sit amet consectetur."


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
        "question", "ground_truth", "gt_label",
        "base_model", "base_label",
        "finetuned", "ft_label",
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
    for key in (
        "ft_rougeL", "label_accuracy_finetuned", "n_examples", "data_path",
        "ft_rougeL_gate_passed", "label_accuracy_gate_passed",
    ):
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
    for key in (
        "base_avg_rouge1", "base_avg_rougeL",
        "ft_avg_rouge1", "ft_avg_rougeL",
        "label_accuracy_base", "label_accuracy_finetuned",
    ):
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


# ---------------------------------------------------------------------------
# Label accuracy gate
# ---------------------------------------------------------------------------

def test_main_gate_exits_when_label_accuracy_below_threshold(data_file, tmp_path):
    """main() must call sys.exit(1) when label accuracy < 0.80, even if ROUGE-L passes."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    # _WRONG_LABEL has the same wording as _GT except for the sentiment token, so
    # ROUGE-L is > 0.85 (gate passes) while label accuracy is 0.0 (gate fails).
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_WRONG_LABEL),
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


def test_main_gate_exits_code_1_not_zero_on_label_accuracy_failure(data_file, tmp_path):
    """Exit code must be 1 (not 0) when the label accuracy gate fails."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_WRONG_LABEL),
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

    assert exc_info.value.code != 0


def test_main_gate_does_not_exit_when_both_thresholds_pass(data_file, tmp_path):
    """main() must not call sys.exit when ROUGE-L >= 0.85 and label accuracy >= 0.80."""
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    # Exact ground-truth output → ROUGE-L = 1.0, label accuracy = 1.0
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


def test_main_label_accuracy_threshold_constant_matches_ci_gate():
    """LABEL_ACCURACY_THRESHOLD must equal 0.80 to match the CI eval.yml gate."""
    assert _eval.LABEL_ACCURACY_THRESHOLD == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Gate-passed flags in summary.json
# ---------------------------------------------------------------------------

def _run_main_for_summary(data_file, tmp_path, generate_return):
    results_p = str(tmp_path / "results.json")
    summary_p = str(tmp_path / "summary.json")
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=generate_return),
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
    return json.loads(Path(summary_p).read_text())


def test_summary_contains_ft_rougeL_gate_passed(data_file, tmp_path):
    """summary.json must include the ft_rougeL_gate_passed boolean flag."""
    summary = _run_main_for_summary(data_file, tmp_path, _GT)
    assert "ft_rougeL_gate_passed" in summary


def test_summary_contains_label_accuracy_gate_passed(data_file, tmp_path):
    """summary.json must include the label_accuracy_gate_passed boolean flag."""
    summary = _run_main_for_summary(data_file, tmp_path, _GT)
    assert "label_accuracy_gate_passed" in summary


def test_summary_rougeL_gate_passed_true_on_good_output(data_file, tmp_path):
    """ft_rougeL_gate_passed must be True when the fine-tuned ROUGE-L >= 0.85."""
    summary = _run_main_for_summary(data_file, tmp_path, _GT)
    assert summary["ft_rougeL_gate_passed"] is True


def test_summary_rougeL_gate_passed_false_on_junk_output(data_file, tmp_path):
    """ft_rougeL_gate_passed must be False when the fine-tuned ROUGE-L < 0.85."""
    summary = _run_main_for_summary(data_file, tmp_path, _JUNK)
    assert summary["ft_rougeL_gate_passed"] is False


def test_summary_label_accuracy_gate_passed_true_on_good_output(data_file, tmp_path):
    """label_accuracy_gate_passed must be True when label accuracy >= 0.80."""
    summary = _run_main_for_summary(data_file, tmp_path, _GT)
    assert summary["label_accuracy_gate_passed"] is True


def test_summary_label_accuracy_gate_passed_false_on_wrong_label(data_file, tmp_path):
    """label_accuracy_gate_passed must be False when label accuracy < 0.80."""
    summary = _run_main_for_summary(data_file, tmp_path, _WRONG_LABEL)
    assert summary["label_accuracy_gate_passed"] is False


def test_summary_gate_flags_are_booleans(data_file, tmp_path):
    """Gate-passed flags must be JSON booleans, not strings or ints."""
    summary = _run_main_for_summary(data_file, tmp_path, _GT)
    assert isinstance(summary["ft_rougeL_gate_passed"], bool)
    assert isinstance(summary["label_accuracy_gate_passed"], bool)


def test_summary_rougeL_fails_but_label_accuracy_passes(data_file, tmp_path):
    """When the label is correct but wording differs, ROUGE-L fails while label accuracy passes.

    This covers the case where the model learns the right classification but
    generates its own explanation — a realistic degradation pattern.  The two
    gates must be evaluated independently so neither masks the other.
    """
    summary = _run_main_for_summary(data_file, tmp_path, _CORRECT_LABEL_WRONG_WORDING)
    assert summary["ft_rougeL_gate_passed"] is False
    assert summary["label_accuracy_gate_passed"] is True


# ---------------------------------------------------------------------------
# --output-results and --output-summary arguments
# ---------------------------------------------------------------------------

def _run_main_custom_outputs(data_file, results_p, summary_p, extra_argv=None):
    argv = ["eval.py", "--data", data_file, "--n", "1", "--no-gate"]
    argv += ["--output-results", str(results_p), "--output-summary", str(summary_p)]
    if extra_argv:
        argv += extra_argv
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch("sys.argv", argv),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact"),
    ):
        _eval.main()


def test_output_results_arg_writes_to_custom_path(data_file, tmp_path):
    """--output-results must write results JSON to the specified path."""
    results_p = tmp_path / "custom" / "results.json"
    summary_p = tmp_path / "custom" / "summary.json"
    _run_main_custom_outputs(data_file, results_p, summary_p)
    assert results_p.exists(), "--output-results path was not written"


def test_output_summary_arg_writes_to_custom_path(data_file, tmp_path):
    """--output-summary must write summary JSON to the specified path."""
    results_p = tmp_path / "custom" / "results.json"
    summary_p = tmp_path / "custom" / "summary.json"
    _run_main_custom_outputs(data_file, results_p, summary_p)
    assert summary_p.exists(), "--output-summary path was not written"


def test_output_results_arg_creates_parent_dirs(data_file, tmp_path):
    """--output-results must create any missing parent directories."""
    results_p = tmp_path / "a" / "b" / "c" / "results.json"
    summary_p = tmp_path / "summary.json"
    _run_main_custom_outputs(data_file, results_p, summary_p)
    assert results_p.exists()


def test_output_results_contains_valid_list(data_file, tmp_path):
    """Custom --output-results path must contain a valid JSON list."""
    results_p = tmp_path / "results.json"
    summary_p = tmp_path / "summary.json"
    _run_main_custom_outputs(data_file, results_p, summary_p)
    data = json.loads(results_p.read_text())
    assert isinstance(data, list)
    assert len(data) == 1


def test_output_summary_contains_gate_flags(data_file, tmp_path):
    """Custom --output-summary path must contain gate-passed flags."""
    results_p = tmp_path / "results.json"
    summary_p = tmp_path / "summary.json"
    _run_main_custom_outputs(data_file, results_p, summary_p)
    summary = json.loads(summary_p.read_text())
    assert "ft_rougeL_gate_passed" in summary
    assert "label_accuracy_gate_passed" in summary


def test_ood_pattern_writes_to_separate_files(tmp_path):
    """OOD eval pattern: --data + --no-gate + custom output paths.

    Verifies that OOD results land in dedicated files so they do not overwrite
    the main eval output, which is the intended CI workflow design.
    """
    ood_data = tmp_path / "ood.jsonl"
    ood_data.write_text(json.dumps({"text": _SAMPLE_TEXT}) + "\n")
    main_results = tmp_path / "results.json"
    main_summary = tmp_path / "summary.json"
    ood_results = tmp_path / "ood_results.json"
    ood_summary = tmp_path / "ood_summary.json"

    # Simulate main eval — write main output files
    _run_main_custom_outputs(str(ood_data), main_results, main_summary)
    main_summary_content = main_summary.read_text()

    # Simulate OOD eval — must NOT touch main output files
    _run_main_custom_outputs(str(ood_data), ood_results, ood_summary)

    assert ood_results.exists(), "OOD results file not written"
    assert ood_summary.exists(), "OOD summary file not written"
    assert main_summary.read_text() == main_summary_content, (
        "OOD eval must not overwrite main eval summary.json"
    )


def test_default_output_paths_match_module_constants(data_file, tmp_path):
    """Without --output-results / --output-summary the defaults match RESULTS_PATH and SUMMARY_PATH."""
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

    assert Path(results_p).exists(), "Default results path not written"
    assert Path(summary_p).exists(), "Default summary path not written"


def test_mlflow_logs_custom_output_results_artifact(data_file, tmp_path):
    """MLflow log_artifact must be called with the custom --output-results path."""
    results_p = tmp_path / "my_results.json"
    summary_p = tmp_path / "my_summary.json"
    with (
        patch.object(_eval, "load", return_value=(MagicMock(), MagicMock())),
        patch.object(_eval, "generate", return_value=_GT),
        patch("sys.argv", [
            "eval.py", "--data", data_file, "--n", "1", "--no-gate",
            "--output-results", str(results_p),
            "--output-summary", str(summary_p),
        ]),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=_mlflow_ctx()),
        patch("mlflow.set_tag"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_param"),
        patch("mlflow.log_artifact") as mock_log_artifact,
    ):
        _eval.main()

    logged = [c[0][0] for c in mock_log_artifact.call_args_list]
    assert str(results_p) in logged, "Custom results path not logged to MLflow"
    assert str(summary_p) in logged, "Custom summary path not logged to MLflow"
