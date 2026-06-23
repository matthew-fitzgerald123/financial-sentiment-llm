"""
Unit tests for data/prepare.py pure helper functions.
No network access, GPU, or model weights required.
"""
import json
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_spec = importlib.util.spec_from_file_location(
    "prepare", Path(__file__).parent.parent / "data" / "prepare.py"
)
_prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prepare)


# ---------------------------------------------------------------------------
# LABEL_MAP
# ---------------------------------------------------------------------------

def test_label_map_negative():
    assert _prepare.LABEL_MAP[0] == "negative"


def test_label_map_neutral():
    assert _prepare.LABEL_MAP[1] == "neutral"


def test_label_map_positive():
    assert _prepare.LABEL_MAP[2] == "positive"


def test_label_map_covers_all_integer_keys():
    assert set(_prepare.LABEL_MAP.keys()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------

_TEXT = "Revenue grew 5% year-over-year."


@pytest.mark.parametrize("label_int,label_str,flavor", [
    (0, "negative", "unfavorable"),
    (1, "neutral", "neutral"),
    (2, "positive", "favorable"),
])
def test_format_prompt_label_and_flavor(label_int, label_str, flavor):
    result = _prepare.format_prompt({"text": _TEXT, "labels": label_int})
    assert f"Sentiment: {label_str}" in result["text"]
    assert flavor in result["text"]


def test_format_prompt_returns_dict_with_text_key():
    result = _prepare.format_prompt({"text": _TEXT, "labels": 2})
    assert isinstance(result, dict)
    assert "text" in result


def test_format_prompt_output_text_is_string():
    result = _prepare.format_prompt({"text": _TEXT, "labels": 0})
    assert isinstance(result["text"], str)


def test_format_prompt_preserves_input_text():
    result = _prepare.format_prompt({"text": _TEXT, "labels": 2})
    assert _TEXT in result["text"]


def test_format_prompt_has_inst_markers():
    result = _prepare.format_prompt({"text": _TEXT, "labels": 0})
    assert "[INST]" in result["text"]
    assert "[/INST]" in result["text"]


def test_format_prompt_has_bos_eos_tokens():
    result = _prepare.format_prompt({"text": _TEXT, "labels": 1})
    assert "<s>" in result["text"]
    assert "</s>" in result["text"]


def test_format_prompt_contains_classify_instruction():
    result = _prepare.format_prompt({"text": _TEXT, "labels": 2})
    assert "Classify the sentiment" in result["text"]


def test_format_prompt_negative_uses_unfavorable_not_favorable():
    result = _prepare.format_prompt({"text": _TEXT, "labels": 0})
    text = result["text"]
    assert "unfavorable" in text
    # "favorable" should only appear as a substring of "unfavorable"
    assert text.count("favorable") == text.count("unfavorable")


def test_format_prompt_neutral_uses_neutral_flavor():
    result = _prepare.format_prompt({"text": _TEXT, "labels": 1})
    text = result["text"]
    assert "neutral" in text
    assert "unfavorable" not in text
    # strip the label word to check the flavor word separately
    assert text.replace("Sentiment: neutral", "").replace("neutral financial conditions", "").count("favorable") == 0


def test_format_prompt_positive_uses_favorable():
    result = _prepare.format_prompt({"text": _TEXT, "labels": 2})
    text = result["text"]
    assert "favorable" in text
    assert "unfavorable" not in text


def test_format_prompt_inst_block_contains_input_text():
    """The [INST]...[/INST] block must wrap the input text, not the label."""
    result = _prepare.format_prompt({"text": _TEXT, "labels": 2})
    text = result["text"]
    inst_block = text.split("[INST]")[1].split("[/INST]")[0]
    assert _TEXT in inst_block


def test_format_prompt_response_after_inst_block_contains_label():
    """The model response (after [/INST]) must contain the structured output."""
    result = _prepare.format_prompt({"text": _TEXT, "labels": 0})
    text = result["text"]
    response = text.split("[/INST]")[1]
    assert "Sentiment: negative" in response


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------

def _fake_split(n):
    return [{"text": f"formatted example {i}"} for i in range(n)]


def test_write_jsonl_creates_file(tmp_path):
    out = tmp_path / "out.jsonl"
    _prepare.write_jsonl(_fake_split(3), out)
    assert out.exists()


def test_write_jsonl_correct_line_count(tmp_path):
    out = tmp_path / "out.jsonl"
    _prepare.write_jsonl(_fake_split(5), out)
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 5


def test_write_jsonl_each_line_is_valid_json(tmp_path):
    out = tmp_path / "out.jsonl"
    _prepare.write_jsonl(_fake_split(3), out)
    for line in out.read_text().splitlines():
        if line.strip():
            json.loads(line)


def test_write_jsonl_each_entry_has_text_key(tmp_path):
    out = tmp_path / "out.jsonl"
    _prepare.write_jsonl(_fake_split(4), out)
    for line in out.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            assert "text" in obj


def test_write_jsonl_text_values_match_split(tmp_path):
    split = _fake_split(3)
    out = tmp_path / "out.jsonl"
    _prepare.write_jsonl(split, out)
    loaded = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    for i, obj in enumerate(loaded):
        assert obj["text"] == split[i]["text"]


def test_write_jsonl_empty_split_creates_empty_file(tmp_path):
    out = tmp_path / "out.jsonl"
    _prepare.write_jsonl([], out)
    assert out.exists()
    assert out.read_text().strip() == ""


def test_write_jsonl_overwrites_existing_file(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text("old content\n")
    _prepare.write_jsonl(_fake_split(2), out)
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "old content" not in out.read_text()


def test_write_jsonl_single_example(tmp_path):
    out = tmp_path / "out.jsonl"
    _prepare.write_jsonl([{"text": "only one"}], out)
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "only one"


# ---------------------------------------------------------------------------
# main() — orchestration (network and file I/O are mocked)
# ---------------------------------------------------------------------------

_FORMATTED_TEXT = (
    "<s>[INST] Classify the sentiment of the following financial statement "
    "and briefly explain your reasoning:\n\n"
    "\"Revenue grew 5% year-over-year.\" [/INST]\n"
    "Sentiment: positive. This statement reflects favorable financial conditions. </s>"
)

_TRAIN_SPLIT = [{"text": _FORMATTED_TEXT} for _ in range(4)]
_VALID_SPLIT = [{"text": _FORMATTED_TEXT} for _ in range(2)]


def _make_mock_dataset():
    """Return a mock HuggingFace DatasetDict that supports split access and .map()."""
    train_mock = MagicMock()
    train_mock.map.return_value = _TRAIN_SPLIT

    test_mock = MagicMock()
    test_mock.map.return_value = _VALID_SPLIT

    mock_raw = MagicMock()
    mock_raw.__getitem__.side_effect = lambda key: train_mock if key == "train" else test_mock
    return mock_raw, train_mock, test_mock


def _run_main():
    """Run prepare.main() with load_dataset and write_jsonl mocked."""
    mock_raw, train_mock, test_mock = _make_mock_dataset()
    with (
        patch.object(_prepare, "load_dataset", return_value=mock_raw) as mock_load,
        patch.object(_prepare, "write_jsonl") as mock_write,
    ):
        _prepare.main()
    return mock_load, mock_write, train_mock, test_mock


def test_main_calls_load_dataset():
    """main() must call load_dataset with the correct HuggingFace dataset name."""
    mock_load, _, _, _ = _run_main()
    mock_load.assert_called_once_with("nickmuchi/financial-classification")


def test_main_calls_write_jsonl_twice():
    """main() must call write_jsonl exactly twice — once for train, once for valid."""
    _, mock_write, _, _ = _run_main()
    assert mock_write.call_count == 2


def test_main_writes_train_jsonl():
    """main() must write the train split to a path ending in train.jsonl."""
    _, mock_write, _, _ = _run_main()
    paths = [str(call.args[1]) for call in mock_write.call_args_list]
    assert any("train.jsonl" in p for p in paths)


def test_main_writes_valid_jsonl():
    """main() must write the valid split to a path ending in valid.jsonl."""
    _, mock_write, _, _ = _run_main()
    paths = [str(call.args[1]) for call in mock_write.call_args_list]
    assert any("valid.jsonl" in p for p in paths)


def test_main_maps_format_prompt_on_train():
    """main() must call .map(format_prompt) on the train split."""
    _, _, train_mock, _ = _run_main()
    train_mock.map.assert_called_once_with(_prepare.format_prompt)


def test_main_maps_format_prompt_on_test():
    """main() must call .map(format_prompt) on the test split."""
    _, _, _, test_mock = _run_main()
    test_mock.map.assert_called_once_with(_prepare.format_prompt)


def test_main_train_path_in_data_dir():
    """train.jsonl path must be under DATA_DIR."""
    _, mock_write, _, _ = _run_main()
    train_path = str(mock_write.call_args_list[0].args[1])
    assert str(_prepare.DATA_DIR) in train_path


def test_main_valid_path_in_data_dir():
    """valid.jsonl path must be under DATA_DIR."""
    _, mock_write, _, _ = _run_main()
    valid_path = str(mock_write.call_args_list[1].args[1])
    assert str(_prepare.DATA_DIR) in valid_path


def test_main_train_write_receives_train_split():
    """The first write_jsonl call must receive the mapped train split."""
    mock_raw, train_mock, test_mock = _make_mock_dataset()
    with (
        patch.object(_prepare, "load_dataset", return_value=mock_raw),
        patch.object(_prepare, "write_jsonl") as mock_write,
    ):
        _prepare.main()
    assert mock_write.call_args_list[0].args[0] is _TRAIN_SPLIT


def test_main_valid_write_receives_test_split():
    """The second write_jsonl call must receive the mapped test split."""
    mock_raw, train_mock, test_mock = _make_mock_dataset()
    with (
        patch.object(_prepare, "load_dataset", return_value=mock_raw),
        patch.object(_prepare, "write_jsonl") as mock_write,
    ):
        _prepare.main()
    assert mock_write.call_args_list[1].args[0] is _VALID_SPLIT


def test_main_accesses_train_split():
    """main() must access raw['train'] for the training split."""
    mock_raw, train_mock, test_mock = _make_mock_dataset()
    with (
        patch.object(_prepare, "load_dataset", return_value=mock_raw),
        patch.object(_prepare, "write_jsonl"),
    ):
        _prepare.main()
    accessed_keys = [c.args[0] for c in mock_raw.__getitem__.call_args_list]
    assert "train" in accessed_keys


def test_main_uses_test_split_for_valid():
    """main() must use raw['test'] (not 'train') to produce valid.jsonl."""
    mock_raw, train_mock, test_mock = _make_mock_dataset()
    with (
        patch.object(_prepare, "load_dataset", return_value=mock_raw),
        patch.object(_prepare, "write_jsonl"),
    ):
        _prepare.main()
    accessed_keys = [c.args[0] for c in mock_raw.__getitem__.call_args_list]
    assert "test" in accessed_keys
