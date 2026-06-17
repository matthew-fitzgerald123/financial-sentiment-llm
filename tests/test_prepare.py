"""
Unit tests for data/prepare.py pure helper functions.
No network access, GPU, or model weights required.
"""
import json
import importlib.util
from pathlib import Path

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
