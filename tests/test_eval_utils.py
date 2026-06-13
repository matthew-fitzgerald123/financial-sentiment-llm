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


# ---------------------------------------------------------------------------
# compute_averages
# ---------------------------------------------------------------------------

def _make_result(base_r1, base_rL, ft_r1, ft_rL):
    return {
        "base_rouge1": base_r1,
        "base_rougeL": base_rL,
        "ft_rouge1": ft_r1,
        "ft_rougeL": ft_rL,
    }


def test_compute_averages_single_result():
    results = [_make_result(0.5, 0.4, 0.9, 0.95)]
    avgs = eval_module.compute_averages(results)
    assert avgs["base_avg_rouge1"] == pytest.approx(0.5)
    assert avgs["base_avg_rougeL"] == pytest.approx(0.4)
    assert avgs["ft_avg_rouge1"] == pytest.approx(0.9)
    assert avgs["ft_avg_rougeL"] == pytest.approx(0.95)


def test_compute_averages_multiple_results():
    results = [
        _make_result(0.2, 0.1, 0.8, 0.9),
        _make_result(0.4, 0.3, 1.0, 1.0),
    ]
    avgs = eval_module.compute_averages(results)
    assert avgs["base_avg_rouge1"] == pytest.approx(0.3)
    assert avgs["base_avg_rougeL"] == pytest.approx(0.2)
    assert avgs["ft_avg_rouge1"] == pytest.approx(0.9)
    assert avgs["ft_avg_rougeL"] == pytest.approx(0.95)


def test_compute_averages_gate_passed_above_threshold():
    results = [_make_result(0.1, 0.1, 0.9, 0.9)]
    avgs = eval_module.compute_averages(results)
    assert avgs["ft_rougeL_gate_passed"] is True


def test_compute_averages_gate_failed_below_threshold():
    results = [_make_result(0.1, 0.1, 0.5, 0.5)]
    avgs = eval_module.compute_averages(results)
    assert avgs["ft_rougeL_gate_passed"] is False


def test_compute_averages_gate_exactly_at_threshold():
    results = [_make_result(0.1, 0.1, 0.85, 0.85)]
    avgs = eval_module.compute_averages(results)
    assert avgs["ft_rougeL_gate_passed"] is True


def test_compute_averages_returns_expected_keys():
    results = [_make_result(0.5, 0.5, 0.9, 0.9)]
    avgs = eval_module.compute_averages(results)
    expected_keys = {
        "base_avg_rouge1", "base_avg_rougeL",
        "ft_avg_rouge1", "ft_avg_rougeL",
        "ft_rougeL_gate_passed",
    }
    assert set(avgs.keys()) == expected_keys
