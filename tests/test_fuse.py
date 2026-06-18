"""
Tests for train/fuse.py — fuse + re-quantize helpers (no GPU or weights required).
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import call, patch

import pytest

_spec = importlib.util.spec_from_file_location(
    "fuse_module", Path(__file__).parent.parent / "train" / "fuse.py"
)
fuse_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fuse_module)
sys.modules["fuse_module"] = fuse_module


# ---------------------------------------------------------------------------
# run_fuse
# ---------------------------------------------------------------------------

def test_run_fuse_calls_subprocess(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_fuse("some-model", "./adapter", str(tmp_path / "fused"))
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "fuse" in cmd
    assert "--model" in cmd
    assert "some-model" in cmd
    assert "--adapter-path" in cmd
    assert "--save-path" in cmd


def test_run_fuse_includes_dequantize_flag_by_default(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_fuse("m", "./a", str(tmp_path / "out"), dequantize=True)
    cmd = mock_run.call_args[0][0]
    assert "--dequantize" in cmd


def test_run_fuse_omits_dequantize_when_false(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_fuse("m", "./a", str(tmp_path / "out"), dequantize=False)
    cmd = mock_run.call_args[0][0]
    assert "--dequantize" not in cmd


def test_run_fuse_passes_adapter_path(tmp_path):
    adapter = "./custom-adapter"
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_fuse("m", adapter, str(tmp_path / "out"))
    cmd = mock_run.call_args[0][0]
    assert adapter in cmd


def test_run_fuse_passes_save_path(tmp_path):
    save = str(tmp_path / "merged")
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_fuse("m", "./a", save)
    cmd = mock_run.call_args[0][0]
    assert save in cmd


def test_run_fuse_uses_check_true(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_fuse("m", "./a", str(tmp_path / "out"))
    _, kwargs = mock_run.call_args
    assert kwargs.get("check") is True


def test_run_fuse_invokes_mlx_lm_fuse(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_fuse("m", "./a", str(tmp_path / "out"))
    cmd = mock_run.call_args[0][0]
    assert "mlx_lm" in cmd
    assert "fuse" in cmd


# ---------------------------------------------------------------------------
# run_quantize
# ---------------------------------------------------------------------------

def test_run_quantize_calls_subprocess(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_quantize(str(tmp_path / "in"), str(tmp_path / "out"))
    mock_run.assert_called_once()


def test_run_quantize_includes_quantize_flag(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_quantize(str(tmp_path / "in"), str(tmp_path / "out"))
    cmd = mock_run.call_args[0][0]
    assert "--quantize" in cmd


def test_run_quantize_passes_q_bits(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_quantize(str(tmp_path / "in"), str(tmp_path / "out"), q_bits=8)
    cmd = mock_run.call_args[0][0]
    assert "--q-bits" in cmd
    idx = cmd.index("--q-bits")
    assert cmd[idx + 1] == "8"


def test_run_quantize_default_q_bits_is_4(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_quantize(str(tmp_path / "in"), str(tmp_path / "out"))
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("--q-bits")
    assert cmd[idx + 1] == "4"


def test_run_quantize_passes_input_and_output_paths(tmp_path):
    input_path = str(tmp_path / "merged")
    output_path = str(tmp_path / "quantized")
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_quantize(input_path, output_path)
    cmd = mock_run.call_args[0][0]
    assert input_path in cmd
    assert output_path in cmd


def test_run_quantize_uses_check_true(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_quantize(str(tmp_path / "in"), str(tmp_path / "out"))
    _, kwargs = mock_run.call_args
    assert kwargs.get("check") is True


def test_run_quantize_invokes_mlx_lm_convert(tmp_path):
    with patch("fuse_module.subprocess.run") as mock_run:
        fuse_module.run_quantize(str(tmp_path / "in"), str(tmp_path / "out"))
    cmd = mock_run.call_args[0][0]
    assert "mlx_lm" in cmd
    assert "convert" in cmd


# ---------------------------------------------------------------------------
# main() — CLI integration
# ---------------------------------------------------------------------------

def test_main_calls_run_fuse_and_run_quantize(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fuse.py"])
    with (
        patch("fuse_module.run_fuse") as mock_fuse,
        patch("fuse_module.run_quantize") as mock_quant,
        patch("fuse_module.mlflow.set_experiment"),
        patch("fuse_module.mlflow.start_run") as mock_run_ctx,
        patch("fuse_module.mlflow.set_tag"),
        patch("fuse_module.mlflow.log_params"),
        patch("fuse_module.mlflow.log_artifact"),
    ):
        mock_run_ctx.return_value.__enter__ = lambda s: s
        mock_run_ctx.return_value.__exit__ = lambda s, *a: False
        fuse_module.main()

    mock_fuse.assert_called_once()
    mock_quant.assert_called_once()


def test_main_no_requantize_skips_run_quantize(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fuse.py", "--no-requantize"])
    with (
        patch("fuse_module.run_fuse"),
        patch("fuse_module.run_quantize") as mock_quant,
        patch("fuse_module.mlflow.set_experiment"),
        patch("fuse_module.mlflow.start_run") as mock_run_ctx,
        patch("fuse_module.mlflow.set_tag"),
        patch("fuse_module.mlflow.log_params"),
        patch("fuse_module.mlflow.log_artifact"),
    ):
        mock_run_ctx.return_value.__enter__ = lambda s: s
        mock_run_ctx.return_value.__exit__ = lambda s, *a: False
        fuse_module.main()

    mock_quant.assert_not_called()


def test_main_custom_q_bits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fuse.py", "--q-bits", "8"])
    with (
        patch("fuse_module.run_fuse"),
        patch("fuse_module.run_quantize") as mock_quant,
        patch("fuse_module.mlflow.set_experiment"),
        patch("fuse_module.mlflow.start_run") as mock_run_ctx,
        patch("fuse_module.mlflow.set_tag"),
        patch("fuse_module.mlflow.log_params"),
        patch("fuse_module.mlflow.log_artifact"),
    ):
        mock_run_ctx.return_value.__enter__ = lambda s: s
        mock_run_ctx.return_value.__exit__ = lambda s, *a: False
        fuse_module.main()

    _, kwargs = mock_quant.call_args
    # q_bits is passed as a positional-or-keyword arg; check either
    call_args = mock_quant.call_args
    assert 8 in call_args[0] or call_args[1].get("q_bits") == 8


def test_main_custom_adapter_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fuse.py", "--adapter", "./my-adapter"])
    with (
        patch("fuse_module.run_fuse") as mock_fuse,
        patch("fuse_module.run_quantize"),
        patch("fuse_module.mlflow.set_experiment"),
        patch("fuse_module.mlflow.start_run") as mock_run_ctx,
        patch("fuse_module.mlflow.set_tag"),
        patch("fuse_module.mlflow.log_params"),
        patch("fuse_module.mlflow.log_artifact"),
    ):
        mock_run_ctx.return_value.__enter__ = lambda s: s
        mock_run_ctx.return_value.__exit__ = lambda s, *a: False
        fuse_module.main()

    call_args = mock_fuse.call_args
    assert "./my-adapter" in call_args[0]


def test_main_logs_to_mlflow(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fuse.py"])
    with (
        patch("fuse_module.run_fuse"),
        patch("fuse_module.run_quantize"),
        patch("fuse_module.mlflow.set_experiment") as mock_set_exp,
        patch("fuse_module.mlflow.start_run") as mock_run_ctx,
        patch("fuse_module.mlflow.set_tag") as mock_tag,
        patch("fuse_module.mlflow.log_params") as mock_params,
        patch("fuse_module.mlflow.log_artifact"),
    ):
        mock_run_ctx.return_value.__enter__ = lambda s: s
        mock_run_ctx.return_value.__exit__ = lambda s, *a: False
        fuse_module.main()

    mock_set_exp.assert_called_once_with("mistral-finance-mlx-lora")
    mock_tag.assert_called_once_with("run_type", "fuse")
    mock_params.assert_called_once()
