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


# --- _parse_label tests ---

def test_parse_label_positive():
    assert eval_module._parse_label("Sentiment: positive. Some explanation.") == "positive"


def test_parse_label_negative():
    assert eval_module._parse_label("Sentiment: negative. Some explanation.") == "negative"


def test_parse_label_neutral():
    assert eval_module._parse_label("Sentiment: neutral. Some explanation.") == "neutral"


def test_parse_label_case_insensitive():
    assert eval_module._parse_label("SENTIMENT: POSITIVE.") == "positive"


def test_parse_label_unknown():
    assert eval_module._parse_label("No structured output here.") == "unknown"


def test_parse_label_empty():
    assert eval_module._parse_label("") == "unknown"


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


def test_label_accuracy_unknown_both():
    results = [_make_result("No structured output here.", "Neither does this.")]
    # both parse to 'unknown', which is equal — counts as correct
    assert eval_module.label_accuracy(results) == pytest.approx(1.0)


def test_label_accuracy_case_insensitive():
    results = [_make_result("SENTIMENT: POSITIVE.", "Sentiment: positive. Explanation.")]
    assert eval_module.label_accuracy(results) == pytest.approx(1.0)


# --- save_summary tests ---

def test_save_summary_writes_json(tmp_path):
    summary = {
        "n_examples": 10,
        "data_path": "./data/valid.jsonl",
        "base_rouge1": 0.113,
        "base_rougeL": 0.094,
        "ft_rouge1": 0.970,
        "ft_rougeL": 0.970,
        "label_accuracy_base": 0.80,
        "label_accuracy_finetuned": 0.95,
    }
    out = tmp_path / "eval" / "summary.json"
    eval_module.save_summary(str(out), summary)

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == summary


def test_save_summary_creates_parent_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "summary.json"
    eval_module.save_summary(str(nested), {"ft_rougeL": 0.97})
    assert nested.exists()


def test_save_summary_required_keys(tmp_path):
    """summary.json must contain the metric and gate-passed keys."""
    summary = {
        "n_examples": 50,
        "data_path": "./data/valid.jsonl",
        "base_rouge1": 0.113,
        "base_rougeL": 0.094,
        "ft_rouge1": 0.970,
        "ft_rougeL": 0.970,
        "ft_rougeL_gate_passed": True,
        "label_accuracy_base": 0.82,
        "label_accuracy_finetuned": 0.95,
        "label_accuracy_gate_passed": True,
    }
    out = tmp_path / "summary.json"
    eval_module.save_summary(str(out), summary)

    loaded = json.loads(out.read_text())
    for key in (
        "ft_rougeL",
        "label_accuracy_finetuned",
        "ft_rougeL_gate_passed",
        "label_accuracy_gate_passed",
    ):
        assert key in loaded, f"Required summary key '{key}' missing"


# --- gate-passed flag tests ---

def _make_summary(ft_rougeL, label_acc):
    return {
        "n_examples": 10,
        "data_path": "./data/valid.jsonl",
        "base_rouge1": 0.1,
        "base_rougeL": 0.1,
        "ft_rouge1": ft_rougeL,
        "ft_rougeL": ft_rougeL,
        "ft_rougeL_gate_passed": ft_rougeL >= eval_module.ROUGE_L_GATE,
        "label_accuracy_base": 0.5,
        "label_accuracy_finetuned": label_acc,
        "label_accuracy_gate_passed": label_acc >= eval_module.LABEL_ACC_GATE,
    }


def test_gate_flags_are_booleans(tmp_path):
    out = tmp_path / "summary.json"
    eval_module.save_summary(str(out), _make_summary(0.97, 0.95))
    loaded = json.loads(out.read_text())
    assert isinstance(loaded["ft_rougeL_gate_passed"], bool)
    assert isinstance(loaded["label_accuracy_gate_passed"], bool)


def test_gate_flags_pass_when_above_thresholds(tmp_path):
    out = tmp_path / "summary.json"
    eval_module.save_summary(str(out), _make_summary(0.90, 0.90))
    loaded = json.loads(out.read_text())
    assert loaded["ft_rougeL_gate_passed"] is True
    assert loaded["label_accuracy_gate_passed"] is True


def test_gate_flags_fail_when_below_rougeL_threshold(tmp_path):
    out = tmp_path / "summary.json"
    eval_module.save_summary(str(out), _make_summary(0.80, 0.90))
    loaded = json.loads(out.read_text())
    assert loaded["ft_rougeL_gate_passed"] is False
    assert loaded["label_accuracy_gate_passed"] is True


def test_gate_flags_fail_when_below_label_acc_threshold(tmp_path):
    out = tmp_path / "summary.json"
    eval_module.save_summary(str(out), _make_summary(0.90, 0.70))
    loaded = json.loads(out.read_text())
    assert loaded["ft_rougeL_gate_passed"] is True
    assert loaded["label_accuracy_gate_passed"] is False


def test_gate_flags_both_fail_when_both_below_threshold(tmp_path):
    out = tmp_path / "summary.json"
    eval_module.save_summary(str(out), _make_summary(0.70, 0.60))
    loaded = json.loads(out.read_text())
    assert loaded["ft_rougeL_gate_passed"] is False
    assert loaded["label_accuracy_gate_passed"] is False


def test_gate_flags_pass_exactly_at_thresholds(tmp_path):
    out = tmp_path / "summary.json"
    eval_module.save_summary(
        str(out),
        _make_summary(eval_module.ROUGE_L_GATE, eval_module.LABEL_ACC_GATE),
    )
    loaded = json.loads(out.read_text())
    assert loaded["ft_rougeL_gate_passed"] is True
    assert loaded["label_accuracy_gate_passed"] is True


def test_label_acc_gate_constant_is_0_80():
    assert eval_module.LABEL_ACC_GATE == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# compute_averages
# ---------------------------------------------------------------------------

def _make_rouge_result(base_r1, base_rL, ft_r1, ft_rL):
    return {
        "base_rouge1": base_r1,
        "base_rougeL": base_rL,
        "ft_rouge1": ft_r1,
        "ft_rougeL": ft_rL,
    }


def test_compute_averages_single_result():
    results = [_make_rouge_result(0.5, 0.4, 0.9, 0.95)]
    avgs = eval_module.compute_averages(results)
    assert avgs["base_avg_rouge1"] == pytest.approx(0.5)
    assert avgs["base_avg_rougeL"] == pytest.approx(0.4)
    assert avgs["ft_avg_rouge1"] == pytest.approx(0.9)
    assert avgs["ft_avg_rougeL"] == pytest.approx(0.95)


def test_compute_averages_multiple_results():
    results = [
        _make_rouge_result(0.2, 0.1, 0.8, 0.9),
        _make_rouge_result(0.4, 0.3, 1.0, 1.0),
    ]
    avgs = eval_module.compute_averages(results)
    assert avgs["base_avg_rouge1"] == pytest.approx(0.3)
    assert avgs["base_avg_rougeL"] == pytest.approx(0.2)
    assert avgs["ft_avg_rouge1"] == pytest.approx(0.9)
    assert avgs["ft_avg_rougeL"] == pytest.approx(0.95)


def test_compute_averages_gate_passed_above_threshold():
    results = [_make_rouge_result(0.1, 0.1, 0.9, 0.9)]
    avgs = eval_module.compute_averages(results)
    assert avgs["ft_rougeL_gate_passed"] is True


def test_compute_averages_gate_failed_below_threshold():
    results = [_make_rouge_result(0.1, 0.1, 0.5, 0.5)]
    avgs = eval_module.compute_averages(results)
    assert avgs["ft_rougeL_gate_passed"] is False


def test_compute_averages_gate_exactly_at_threshold():
    results = [_make_rouge_result(0.1, 0.1, 0.85, 0.85)]
    avgs = eval_module.compute_averages(results)
    assert avgs["ft_rougeL_gate_passed"] is True


def test_compute_averages_returns_expected_keys():
    results = [_make_rouge_result(0.5, 0.5, 0.9, 0.9)]
    avgs = eval_module.compute_averages(results)
    expected_keys = {
        "base_avg_rouge1", "base_avg_rougeL",
        "ft_avg_rouge1", "ft_avg_rougeL",
        "ft_rougeL_gate_passed",
    }
    assert set(avgs.keys()) == expected_keys


# --- check_gate tests ---

def _make_results(ft_rougeL_values):
    return [{"ft_rougeL": v} for v in ft_rougeL_values]


def test_check_gate_passes_above_threshold():
    avg, passed = eval_module.check_gate(_make_results([0.90, 0.95, 0.92]))
    assert passed is True
    assert avg == pytest.approx((0.90 + 0.95 + 0.92) / 3)


def test_check_gate_fails_below_threshold():
    avg, passed = eval_module.check_gate(_make_results([0.70, 0.75, 0.80]))
    assert passed is False
    assert avg < 0.85


def test_check_gate_passes_exactly_at_threshold():
    avg, passed = eval_module.check_gate(_make_results([0.85, 0.85, 0.85]))
    assert passed is True
    assert avg == pytest.approx(0.85)


def test_check_gate_single_result_fail():
    avg, passed = eval_module.check_gate(_make_results([0.84]))
    assert passed is False


def test_check_gate_custom_threshold():
    avg, passed = eval_module.check_gate(_make_results([0.60, 0.65]), threshold=0.50)
    assert passed is True

    avg2, passed2 = eval_module.check_gate(_make_results([0.40, 0.45]), threshold=0.50)
    assert passed2 is False
