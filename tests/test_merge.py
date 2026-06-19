"""
Tests for scripts/merge.py (no model weights required).
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "merge_module", Path(__file__).parent.parent / "scripts" / "merge.py"
)
_merge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_merge)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_adapter_dir(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "mistral-finetuned"
    adapter_dir.mkdir()
    config = {"lora_parameters": {"rank": 8, "scale": 10.0, "dropout": 0.05}}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
    return adapter_dir


# ---------------------------------------------------------------------------
# parse_args — defaults
# ---------------------------------------------------------------------------

def test_parse_args_default_model():
    args = _merge.parse_args([])
    assert args.model == _merge.BASE_MODEL_ID


def test_parse_args_default_adapter():
    args = _merge.parse_args([])
    assert args.adapter == _merge.DEFAULT_ADAPTER


def test_parse_args_default_output():
    args = _merge.parse_args([])
    assert args.output == _merge.DEFAULT_OUTPUT


def test_parse_args_default_no_quantize_is_false():
    args = _merge.parse_args([])
    assert args.no_quantize is False


def test_parse_args_default_bits():
    args = _merge.parse_args([])
    assert args.bits == _merge.DEFAULT_QUANTIZE_BITS


# ---------------------------------------------------------------------------
# parse_args — custom values
# ---------------------------------------------------------------------------

def test_parse_args_custom_adapter():
    args = _merge.parse_args(["--adapter", "/path/to/my-adapter"])
    assert args.adapter == "/path/to/my-adapter"


def test_parse_args_custom_output():
    args = _merge.parse_args(["--output", "/path/to/merged"])
    assert args.output == "/path/to/merged"


def test_parse_args_no_quantize_flag():
    args = _merge.parse_args(["--no-quantize"])
    assert args.no_quantize is True


def test_parse_args_bits_4():
    args = _merge.parse_args(["--bits", "4"])
    assert args.bits == 4


def test_parse_args_bits_8():
    args = _merge.parse_args(["--bits", "8"])
    assert args.bits == 8


def test_parse_args_bits_invalid_exits():
    with pytest.raises(SystemExit):
        _merge.parse_args(["--bits", "16"])


def test_parse_args_custom_model():
    args = _merge.parse_args(["--model", "mistralai/Mistral-7B-v0.1"])
    assert args.model == "mistralai/Mistral-7B-v0.1"


# ---------------------------------------------------------------------------
# validate_inputs — error paths
# ---------------------------------------------------------------------------

def test_validate_inputs_missing_adapter_dir(tmp_path):
    with pytest.raises(ValueError, match="Adapter not found"):
        _merge.validate_inputs(str(tmp_path / "nonexistent"), str(tmp_path / "out"))


def test_validate_inputs_adapter_dir_exists_but_no_config(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    with pytest.raises(ValueError, match="adapter_config.json missing"):
        _merge.validate_inputs(str(adapter_dir), str(tmp_path / "out"))


def test_validate_inputs_output_already_exists(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _merge.validate_inputs(str(adapter_dir), str(output_dir))


def test_validate_inputs_valid_paths_does_not_raise(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    _merge.validate_inputs(str(adapter_dir), str(tmp_path / "merged-output"))


def test_validate_inputs_error_message_includes_adapter_path(tmp_path):
    bad_path = str(tmp_path / "no-adapter-here")
    with pytest.raises(ValueError, match="no-adapter-here"):
        _merge.validate_inputs(bad_path, str(tmp_path / "out"))


def test_validate_inputs_error_message_includes_output_path(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    output_dir = tmp_path / "already-there"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="already-there"):
        _merge.validate_inputs(str(adapter_dir), str(output_dir))


# ---------------------------------------------------------------------------
# main() — merge and quantize paths
# ---------------------------------------------------------------------------

def test_main_calls_merge(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    output_path = str(tmp_path / "merged")

    with (
        patch.object(_merge, "merge") as mock_merge,
        patch.object(_merge, "quantize"),
    ):
        _merge.main(["--adapter", str(adapter_dir), "--output", output_path])

    mock_merge.assert_called_once_with(_merge.BASE_MODEL_ID, str(adapter_dir), output_path)


def test_main_calls_quantize_by_default(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    output_path = str(tmp_path / "merged")

    with (
        patch.object(_merge, "merge"),
        patch.object(_merge, "quantize") as mock_quantize,
    ):
        _merge.main(["--adapter", str(adapter_dir), "--output", output_path])

    mock_quantize.assert_called_once_with(output_path, _merge.DEFAULT_QUANTIZE_BITS)


def test_main_no_quantize_skips_quantize(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    output_path = str(tmp_path / "merged")

    with (
        patch.object(_merge, "merge"),
        patch.object(_merge, "quantize") as mock_quantize,
    ):
        _merge.main([
            "--adapter", str(adapter_dir),
            "--output", output_path,
            "--no-quantize",
        ])

    mock_quantize.assert_not_called()


def test_main_custom_bits_passed_to_quantize(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    output_path = str(tmp_path / "merged")

    with (
        patch.object(_merge, "merge"),
        patch.object(_merge, "quantize") as mock_quantize,
    ):
        _merge.main([
            "--adapter", str(adapter_dir),
            "--output", output_path,
            "--bits", "8",
        ])

    mock_quantize.assert_called_once_with(output_path, 8)


def test_main_passes_model_id_to_merge(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    output_path = str(tmp_path / "merged")
    custom_model = "mistralai/Mistral-7B-v0.1"

    with (
        patch.object(_merge, "merge") as mock_merge,
        patch.object(_merge, "quantize"),
    ):
        _merge.main([
            "--model", custom_model,
            "--adapter", str(adapter_dir),
            "--output", output_path,
        ])

    mock_merge.assert_called_once_with(custom_model, str(adapter_dir), output_path)


def test_main_raises_if_adapter_missing(tmp_path):
    with pytest.raises(ValueError, match="Adapter not found"):
        _merge.main([
            "--adapter", str(tmp_path / "nonexistent"),
            "--output", str(tmp_path / "merged"),
        ])


def test_main_raises_if_output_exists(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    output_dir = tmp_path / "merged"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        _merge.main([
            "--adapter", str(adapter_dir),
            "--output", str(output_dir),
        ])


# ---------------------------------------------------------------------------
# merge() — verifies it calls mlx_lm.fuse with correct arguments
# ---------------------------------------------------------------------------

def test_merge_calls_fuse_with_correct_args():
    mock_fuse = patch("mlx_lm.fuse").start()
    import mlx_lm
    mlx_lm.fuse = mock_fuse

    _merge.merge("some-model", "./adapter", "./output")

    mock_fuse.assert_called_once_with(
        model="some-model",
        adapter_path="./adapter",
        save_path="./output",
        de_quantize=False,
    )
    patch.stopall()


# ---------------------------------------------------------------------------
# quantize() — verifies it calls mlx_lm.convert with correct arguments
# ---------------------------------------------------------------------------

def test_quantize_calls_convert_with_correct_args():
    mock_convert = patch("mlx_lm.convert").start()
    import mlx_lm
    mlx_lm.convert = mock_convert

    _merge.quantize("./output", 4)

    mock_convert.assert_called_once_with(
        hf_path="./output",
        mlx_path="./output",
        quantize=True,
        q_bits=4,
    )
    patch.stopall()


def test_quantize_passes_bits_argument():
    mock_convert = patch("mlx_lm.convert").start()
    import mlx_lm
    mlx_lm.convert = mock_convert

    _merge.quantize("./output", 8)

    call_kwargs = mock_convert.call_args.kwargs
    assert call_kwargs["q_bits"] == 8
    patch.stopall()
