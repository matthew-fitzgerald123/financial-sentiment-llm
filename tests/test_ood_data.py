"""
Validate that data/ood_sample.jsonl is correctly formatted so that
eval.py can consume it without modification via `python eval/eval.py --data data/ood_sample.jsonl`.

No model weights or network access required.
"""
import json
import re
from pathlib import Path

import pytest

OOD_PATH = Path(__file__).parent.parent / "data" / "ood_sample.jsonl"

_LABEL_RE = re.compile(r"Sentiment:\s*(positive|neutral|negative)", re.IGNORECASE)


def _load_ood():
    return [json.loads(ln) for ln in OOD_PATH.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Load helpers from eval.py (same pattern as test_eval_utils.py)
# ---------------------------------------------------------------------------
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "eval_for_ood", Path(__file__).parent.parent / "eval" / "eval.py"
)
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------

def test_ood_file_exists():
    assert OOD_PATH.exists(), f"OOD sample file not found at {OOD_PATH}"


def test_ood_has_ten_examples():
    examples = _load_ood()
    assert len(examples) == 10


def test_ood_each_line_is_valid_json():
    for line in OOD_PATH.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            assert "text" in obj, "Every OOD entry must have a 'text' key"


# ---------------------------------------------------------------------------
# Per-example format checks (eval.py's parsing functions must not raise)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", _load_ood())
def test_ood_build_question_parseable(example):
    q = _eval.build_question(example["text"])
    assert len(q) > 0
    assert "[INST]" not in q
    assert "[/INST]" not in q


@pytest.mark.parametrize("example", _load_ood())
def test_ood_build_ground_truth_parseable(example):
    gt = _eval.build_ground_truth(example["text"])
    assert len(gt) > 0
    assert "</s>" not in gt


@pytest.mark.parametrize("example", _load_ood())
def test_ood_ground_truth_has_recognizable_label(example):
    gt = _eval.build_ground_truth(example["text"])
    assert _LABEL_RE.search(gt), (
        f"Ground truth must contain 'Sentiment: <positive|neutral|negative>'; got: {gt!r}"
    )


# ---------------------------------------------------------------------------
# Label distribution: all three classes should appear at least once
# ---------------------------------------------------------------------------

def test_ood_label_distribution_covers_all_classes():
    labels = set()
    for ex in _load_ood():
        gt = _eval.build_ground_truth(ex["text"])
        m = _LABEL_RE.search(gt)
        if m:
            labels.add(m.group(1).lower())
    assert labels == {"positive", "neutral", "negative"}, (
        f"OOD sample must include all three sentiment classes; found: {labels}"
    )
