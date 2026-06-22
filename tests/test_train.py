"""
Tests for train/train.py helper functions (no GPU or model weights required).
"""
import json
import textwrap
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

# load_hparams is a pure function — import it directly without executing main()
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "train_module", Path(__file__).parent.parent / "train" / "train.py"
)
_train = importlib.util.module_from_spec(_spec)


def _load_train():
    _spec.loader.exec_module(_train)


_load_train()


_SAMPLE_CONFIG = textwrap.dedent("""\
    model: mlx-community/Mistral-7B-Instruct-v0.3-4bit
    train: true
    data: ./data
    adapter_path: ./mistral-finetuned

    iters: 1000
    batch_size: 2
    num_layers: 16
    learning_rate: 5.0e-5
    optimizer: adamw
    grad_checkpoint: true
    max_seq_length: 512

    lora_parameters:
      rank: 8
      dropout: 0.05
      scale: 10.0
""")


@pytest.fixture()
def config_file(tmp_path):
    p = tmp_path / "lora_config.yaml"
    p.write_text(_SAMPLE_CONFIG)
    return str(p)


def test_load_hparams_extracts_top_level_fields(config_file):
    hparams = _train.load_hparams(config_file)
    assert hparams["model_id"] == "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    assert hparams["iters"] == 1000
    assert hparams["batch_size"] == 2
    assert hparams["lora_layers"] == 16
    assert hparams["optimizer"] == "adamw"
    assert hparams["max_seq_length"] == 512


def test_load_hparams_learning_rate(config_file):
    hparams = _train.load_hparams(config_file)
    assert abs(hparams["learning_rate"] - 5e-5) < 1e-10


def test_load_hparams_extracts_lora_parameters(config_file):
    hparams = _train.load_hparams(config_file)
    assert hparams["lora_rank"] == 8
    assert abs(hparams["lora_scale"] - 10.0) < 1e-9
    assert abs(hparams["lora_dropout"] - 0.05) < 1e-9


def test_load_hparams_returns_flat_dict(config_file):
    hparams = _train.load_hparams(config_file)
    for v in hparams.values():
        assert not isinstance(v, dict), "load_hparams must return a flat dict"


def test_load_hparams_missing_optional_fields(tmp_path):
    minimal = tmp_path / "minimal.yaml"
    minimal.write_text("model: some-model\niters: 500\n")
    hparams = _train.load_hparams(str(minimal))
    assert hparams["model_id"] == "some-model"
    assert hparams["iters"] == 500
    assert hparams["lora_rank"] is None
    assert hparams["lora_scale"] is None


def test_load_hparams_keys_match_mlruns_schema(config_file):
    hparams = _train.load_hparams(config_file)
    expected_keys = {
        "model_id", "iters", "batch_size", "learning_rate",
        "lora_layers", "lora_rank", "lora_scale", "lora_dropout",
        "max_seq_length", "optimizer",
    }
    assert set(hparams.keys()) == expected_keys


# ---------------------------------------------------------------------------
# set_memory_limit
# ---------------------------------------------------------------------------

def test_set_memory_limit_calls_metal_api():
    """set_memory_limit must call mx.metal.set_memory_limit with the correct byte value."""
    mock_metal = MagicMock()
    with patch.object(_train.mx, "metal", mock_metal):
        _train.set_memory_limit()
    expected_bytes = _train.MEMORY_LIMIT_GB * 1024 ** 3
    mock_metal.set_memory_limit.assert_called_once_with(expected_bytes)


def test_set_memory_limit_handles_attribute_error(capsys):
    """set_memory_limit must not raise when mx.metal.set_memory_limit is absent."""
    mock_metal = MagicMock()
    mock_metal.set_memory_limit.side_effect = AttributeError("not available")
    with patch.object(_train.mx, "metal", mock_metal):
        _train.set_memory_limit()  # must not raise
    captured = capsys.readouterr()
    assert "skipping" in captured.out.lower()


def test_set_memory_limit_uses_config_constant():
    """The byte value passed must equal MEMORY_LIMIT_GB * 1024**3."""
    mock_metal = MagicMock()
    with patch.object(_train.mx, "metal", mock_metal):
        _train.set_memory_limit()
    actual = mock_metal.set_memory_limit.call_args[0][0]
    assert actual == _train.MEMORY_LIMIT_GB * 1024 ** 3


# ---------------------------------------------------------------------------
# main() — subprocess and MLflow integration
# ---------------------------------------------------------------------------

def test_main_invokes_lora_subprocess(config_file):
    """main() must call subprocess.run with the mlx_lm lora command."""
    import sys

    with (
        patch("subprocess.run") as mock_run,
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run"),
        patch("mlflow.log_params"),
        patch("mlflow.log_artifact"),
        patch.object(_train, "CONFIG", config_file),
        patch.object(_train, "set_memory_limit"),
    ):
        _train.main()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert sys.executable in cmd
    assert "-m" in cmd
    assert "mlx_lm" in cmd
    assert "lora" in cmd
    assert "-c" in cmd


def test_main_passes_config_to_lora(config_file):
    """main() must pass CONFIG as the -c argument to mlx_lm lora."""
    with (
        patch("subprocess.run") as mock_run,
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run"),
        patch("mlflow.log_params"),
        patch("mlflow.log_artifact"),
        patch.object(_train, "CONFIG", config_file),
        patch.object(_train, "set_memory_limit"),
    ):
        _train.main()

    cmd = mock_run.call_args[0][0]
    assert config_file in cmd


def test_main_logs_hparams_to_mlflow(config_file):
    """main() must call mlflow.log_params with the hyperparameters from the config."""
    with (
        patch("subprocess.run"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run") as mock_run_ctx,
        patch("mlflow.log_params") as mock_log_params,
        patch("mlflow.log_artifact"),
        patch.object(_train, "CONFIG", config_file),
        patch.object(_train, "set_memory_limit"),
    ):
        mock_run_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_run_ctx.return_value.__exit__ = MagicMock(return_value=False)
        _train.main()

    mock_log_params.assert_called_once()
    logged = mock_log_params.call_args[0][0]
    assert "model_id" in logged
    assert "iters" in logged
    assert "learning_rate" in logged


def test_main_logs_config_artifact(config_file):
    """main() must log the lora_config.yaml as an MLflow artifact."""
    with (
        patch("subprocess.run"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run") as mock_run_ctx,
        patch("mlflow.log_params"),
        patch("mlflow.log_artifact") as mock_log_artifact,
        patch.object(_train, "CONFIG", config_file),
        patch.object(_train, "set_memory_limit"),
    ):
        mock_run_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_run_ctx.return_value.__exit__ = MagicMock(return_value=False)
        _train.main()

    artifact_paths = [c[0][0] for c in mock_log_artifact.call_args_list]
    assert config_file in artifact_paths


def test_main_logs_adapter_artifact(config_file):
    """main() must log the adapter directory as an MLflow artifact after training."""
    with (
        patch("subprocess.run"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run") as mock_run_ctx,
        patch("mlflow.log_params"),
        patch("mlflow.log_artifact") as mock_log_artifact,
        patch.object(_train, "CONFIG", config_file),
        patch.object(_train, "set_memory_limit"),
    ):
        mock_run_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_run_ctx.return_value.__exit__ = MagicMock(return_value=False)
        _train.main()

    artifact_paths = [c[0][0] for c in mock_log_artifact.call_args_list]
    assert _train.ADAPTER_PATH in artifact_paths


def test_main_sets_mlflow_experiment(config_file):
    """main() must target the shared MLflow experiment."""
    with (
        patch("subprocess.run"),
        patch("mlflow.set_experiment") as mock_set_exp,
        patch("mlflow.start_run") as mock_run_ctx,
        patch("mlflow.log_params"),
        patch("mlflow.log_artifact"),
        patch.object(_train, "CONFIG", config_file),
        patch.object(_train, "set_memory_limit"),
    ):
        mock_run_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_run_ctx.return_value.__exit__ = MagicMock(return_value=False)
        _train.main()

    mock_set_exp.assert_called_once_with("mistral-finance-mlx-lora")
