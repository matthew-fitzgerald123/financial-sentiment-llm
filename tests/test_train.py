"""
Tests for train/train.py helper functions (no GPU or model weights required).
"""
import json
import textwrap

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
