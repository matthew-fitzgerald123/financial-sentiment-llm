"""
Tests for eval scoring helpers (no model loading required).
"""
import json
import tempfile
from pathlib import Path

import pytest
from rouge_score import rouge_scorer

# Import the pure helper functions from eval.py
import importlib.util, sys

spec = importlib.util.spec_from_file_location(
    "eval_module", Path(__file__).parent.parent / "eval" / "eval.py"
)
eval_module = importlib.util.module_from_spec(spec)


def _load_eval():
    """Load eval.py without executing its main() or model loading at import time."""
    # eval.py has no top-level side effects beyond imports, so this is safe
    spec.loader.exec_module(eval_module)


_load_eval()


SAMPLE_TEXT = (
    "<s>[INST] Classify the sentiment: 'Revenue declined 8%.' [/INST]"
    "Sentiment: negative. This statement reflects unfavorable financial conditions.</s>"
)


def test_build_question_extracts_instruction():
    q = eval_module.build_question(SAMPLE_TEXT)
    assert "Revenue declined 8%" in q
    assert "[INST]" not in q
    assert "[/INST]" not in q


def test_build_ground_truth_extracts_response():
    gt = eval_module.build_ground_truth(SAMPLE_TEXT)
    assert gt.startswith("Sentiment: negative")
    assert "</s>" not in gt
    assert "[/INST]" not in gt


def test_rouge_perfect_match():
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    text = "Sentiment: positive. This statement reflects favorable financial conditions."
    scores = scorer.score(text, text)
    assert scores["rougeL"].fmeasure == pytest.approx(1.0)


def test_rouge_zero_mismatch():
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score("apple banana cherry", "xyz foo bar")
    assert scores["rougeL"].fmeasure == pytest.approx(0.0)


def test_load_examples(tmp_path):
    jsonl = tmp_path / "valid.jsonl"
    lines = [json.dumps({"text": SAMPLE_TEXT}) for _ in range(5)]
    jsonl.write_text("\n".join(lines))

    examples = eval_module.load_examples(str(jsonl), 3)
    assert len(examples) == 3
    assert "text" in examples[0]


# --- label_accuracy tests ---

_POS = "Sentiment: positive. This statement reflects favorable financial conditions."
_NEG = "Sentiment: negative. This statement reflects unfavorable financial conditions."
_NEU = "Sentiment: neutral. This is a neutral statement."


def _make_result(ground_truth, finetuned):
    return {"ground_truth": ground_truth, "finetuned": finetuned}


def test_label_accuracy_perfect():
    results = [
        _make_result(_POS, _POS),
        _make_result(_NEG, _NEG),
        _make_result(_NEU, _NEU),
    ]
    assert eval_module.label_accuracy(results) == pytest.approx(1.0)


def test_label_accuracy_zero():
    results = [
        _make_result(_POS, _NEG),
        _make_result(_NEG, _NEU),
    ]
    assert eval_module.label_accuracy(results) == pytest.approx(0.0)


def test_label_accuracy_partial():
    results = [
        _make_result(_POS, _POS),
        _make_result(_NEG, _POS),
        _make_result(_NEU, _NEU),
        _make_result(_POS, _NEG),
    ]
    assert eval_module.label_accuracy(results) == pytest.approx(0.5)


def test_label_accuracy_empty():
    assert eval_module.label_accuracy([]) == pytest.approx(0.0)


def test_label_accuracy_unknown_label():
    results = [_make_result("No structured output here.", "Neither does this.")]
    # both parse to 'unknown', which is equal — counts as correct
    assert eval_module.label_accuracy(results) == pytest.approx(1.0)


def test_label_accuracy_case_insensitive():
    results = [_make_result("SENTIMENT: POSITIVE.", "Sentiment: positive. Explanation.")]
    assert eval_module.label_accuracy(results) == pytest.approx(1.0)
