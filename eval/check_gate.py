"""
Read eval/summary.json and exit non-zero if either CI gate flag is False.
Called by the CI pipeline immediately after eval/eval.py.

Thresholds are not re-defined here — they live in eval.py and are baked
into the ft_rougeL_gate_passed and label_accuracy_gate_passed flags at
eval time.
"""
import json
import sys
from pathlib import Path

SUMMARY_PATH = "./eval/summary.json"


def check_summary(path: str = SUMMARY_PATH) -> tuple:
    """Return (passed, message) where passed is True iff both gate flags are True.

    Reads ft_rougeL_gate_passed and label_accuracy_gate_passed from the
    summary written by eval.py. Missing keys are treated as False.
    """
    with open(path) as f:
        s = json.load(f)

    rouge_passed = s.get("ft_rougeL_gate_passed", False)
    label_passed = s.get("label_accuracy_gate_passed", False)

    rougeL = s.get("ft_rougeL", 0.0)
    label_acc = s.get("label_accuracy_finetuned", 0.0)

    lines = [
        f"avg fine-tuned ROUGE-L:    {rougeL:.3f}",
        f"fine-tuned label accuracy: {label_acc:.3f}",
    ]
    failed = False
    if not rouge_passed:
        lines.append("FAIL: ROUGE-L gate not passed")
        failed = True
    if not label_passed:
        lines.append("FAIL: label accuracy gate not passed")
        failed = True
    if not failed:
        lines.append("PASS")

    return not failed, "\n".join(lines)


def main():
    passed, message = check_summary()
    print(message)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
