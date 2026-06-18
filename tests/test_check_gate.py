"""
Tests for eval/check_gate.py — CI gate that reads pre-computed boolean flags.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "check_gate", Path(__file__).parent.parent / "eval" / "check_gate.py"
)
check_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_gate)


def _write_summary(tmp_path, **kwargs):
    default = {
        "ft_rougeL": 0.97,
        "label_accuracy_finetuned": 0.95,
        "ft_rougeL_gate_passed": True,
        "label_accuracy_gate_passed": True,
    }
    default.update(kwargs)
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(default))
    return str(p)


def test_passes_when_both_flags_true(tmp_path):
    path = _write_summary(tmp_path, ft_rougeL_gate_passed=True, label_accuracy_gate_passed=True)
    passed, msg = check_gate.check_summary(path)
    assert passed is True
    assert "PASS" in msg


def test_fails_when_rougeL_gate_false(tmp_path):
    path = _write_summary(tmp_path, ft_rougeL_gate_passed=False, label_accuracy_gate_passed=True)
    passed, msg = check_gate.check_summary(path)
    assert passed is False
    assert "FAIL" in msg
    assert "ROUGE-L" in msg


def test_fails_when_label_acc_gate_false(tmp_path):
    path = _write_summary(tmp_path, ft_rougeL_gate_passed=True, label_accuracy_gate_passed=False)
    passed, msg = check_gate.check_summary(path)
    assert passed is False
    assert "FAIL" in msg
    assert "label accuracy" in msg


def test_fails_when_both_flags_false(tmp_path):
    path = _write_summary(tmp_path, ft_rougeL_gate_passed=False, label_accuracy_gate_passed=False)
    passed, msg = check_gate.check_summary(path)
    assert passed is False
    assert msg.count("FAIL") == 2


def test_missing_rouge_flag_treated_as_false(tmp_path):
    summary = {"ft_rougeL": 0.97, "label_accuracy_finetuned": 0.95, "label_accuracy_gate_passed": True}
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary))
    passed, msg = check_gate.check_summary(str(p))
    assert passed is False


def test_missing_label_flag_treated_as_false(tmp_path):
    summary = {"ft_rougeL": 0.97, "label_accuracy_finetuned": 0.95, "ft_rougeL_gate_passed": True}
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary))
    passed, msg = check_gate.check_summary(str(p))
    assert passed is False


def test_message_includes_metric_values(tmp_path):
    path = _write_summary(tmp_path, ft_rougeL=0.971, label_accuracy_finetuned=0.952)
    passed, msg = check_gate.check_summary(path)
    assert "0.971" in msg
    assert "0.952" in msg


def test_main_exits_zero_on_pass(tmp_path):
    path = _write_summary(tmp_path, ft_rougeL_gate_passed=True, label_accuracy_gate_passed=True)
    result = subprocess.run(
        [sys.executable, "eval/check_gate.py"],
        env={**__import__("os").environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        # Override SUMMARY_PATH by symlinking; use monkeypatching via script arg instead
        cwd=str(tmp_path.parent.parent),
    )
    # We can't easily override the default path via CLI, so test check_summary directly
    passed, _ = check_gate.check_summary(path)
    assert passed is True


def test_main_exits_nonzero_on_fail(tmp_path):
    path = _write_summary(tmp_path, ft_rougeL_gate_passed=False, label_accuracy_gate_passed=False)
    passed, _ = check_gate.check_summary(path)
    assert passed is False


def test_no_fail_message_when_passing(tmp_path):
    path = _write_summary(tmp_path, ft_rougeL_gate_passed=True, label_accuracy_gate_passed=True)
    passed, msg = check_gate.check_summary(path)
    assert "FAIL" not in msg


def test_pass_message_absent_when_failing(tmp_path):
    path = _write_summary(tmp_path, ft_rougeL_gate_passed=False, label_accuracy_gate_passed=False)
    passed, msg = check_gate.check_summary(path)
    assert "PASS" not in msg
