"""
Tests for data/prepare.py — label mapping and JSONL serialization.
No network access or GPU required.
"""
import json
import importlib.util
from pathlib import Path

import pytest

# Load prepare.py without triggering main() or any dataset download.
_spec = importlib.util.spec_from_file_location(
    "prepare_module", Path(__file__).parent.parent / "data" / "prepare.py"
)
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------

def _make_example(label_int: int, text: str = "Revenue rose 12%.") -> dict:
    return {"labels": label_int, "text": text}


@pytest.mark.parametrize(
    "label_int, expected_label, expected_sentiment_word",
    [
        (0, "negative", "unfavorable"),
        (1, "neutral",  "neutral"),
        (2, "positive", "favorable"),
    ],
)
def test_format_prompt_label_mapping(label_int, expected_label, expected_sentiment_word):
    result = prepare.format_prompt(_make_example(label_int))
    text = result["text"]
    assert f"Sentiment: {expected_label}." in text
    assert expected_sentiment_word in text


def test_format_prompt_contains_instruction_markers():
    result = prepare.format_prompt(_make_example(2))
    text = result["text"]
    assert "[INST]" in text
    assert "[/INST]" in text


def test_format_prompt_embeds_original_text():
    input_text = "Operating margins expanded by 300 basis points."
    result = prepare.format_prompt({"labels": 2, "text": input_text})
    assert input_text in result["text"]


def test_format_prompt_starts_with_bos_token():
    result = prepare.format_prompt(_make_example(1))
    assert result["text"].startswith("<s>")


def test_format_prompt_ends_with_eos_token():
    result = prepare.format_prompt(_make_example(1))
    assert result["text"].strip().endswith("</s>")


def test_format_prompt_returns_dict_with_text_key():
    result = prepare.format_prompt(_make_example(0))
    assert isinstance(result, dict)
    assert "text" in result


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------

def _formatted_examples(count: int, label_int: int = 2) -> list[dict]:
    raw = [{"labels": label_int, "text": f"Example {i}."} for i in range(count)]
    return [prepare.format_prompt(ex) for ex in raw]


def test_write_jsonl_creates_file(tmp_path):
    path = tmp_path / "out.jsonl"
    prepare.write_jsonl(_formatted_examples(3), path)
    assert path.exists()


def test_write_jsonl_line_count(tmp_path):
    path = tmp_path / "out.jsonl"
    prepare.write_jsonl(_formatted_examples(5), path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 5


def test_write_jsonl_each_line_is_valid_json(tmp_path):
    path = tmp_path / "out.jsonl"
    prepare.write_jsonl(_formatted_examples(4), path)
    for line in path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            assert "text" in obj


def test_write_jsonl_text_field_is_instruction_format(tmp_path):
    path = tmp_path / "out.jsonl"
    prepare.write_jsonl(_formatted_examples(2, label_int=0), path)
    first = json.loads(path.read_text().splitlines()[0])
    assert "[INST]" in first["text"]
    assert "Sentiment: negative." in first["text"]


def test_write_jsonl_empty_split(tmp_path):
    path = tmp_path / "empty.jsonl"
    prepare.write_jsonl([], path)
    assert path.exists()
    assert path.read_text().strip() == ""


# ---------------------------------------------------------------------------
# LABEL_MAP coverage
# ---------------------------------------------------------------------------

def test_label_map_covers_all_dataset_classes():
    assert set(prepare.LABEL_MAP.keys()) == {0, 1, 2}
    assert set(prepare.LABEL_MAP.values()) == {"negative", "neutral", "positive"}
