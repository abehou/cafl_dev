import json
from types import SimpleNamespace

import pytest

from envs.eval import (
    accuracy_score,
    classification_metrics,
    evaluate_result,
    f1_score,
    normalize_for_comparison,
    precision_score,
    recall_score,
    write_evaluation,
)


def test_normalize_for_comparison_is_generic_text_normalization():
    assert normalize_for_comparison("  Hello   WORLD ") == "hello world"
    assert normalize_for_comparison(3) == 3
    assert normalize_for_comparison(True) is True


def test_evaluate_result_extracts_configured_prediction_field():
    row = {"idx": 7, "gold": "Rent Control"}
    result = SimpleNamespace(
        question="What law applies?",
        answer='{"label": " rent   CONTROL "}',
        output_dir="/tmp/item-007",
    )

    record = evaluate_result(row, result, ground_truth_field="gold", prediction_field="label")

    assert record["correct"] is True
    assert record["predicted"] == " rent   CONTROL "
    assert record["expected"] == "Rent Control"
    assert record["parsed_answer"] == {"label": " rent   CONTROL "}


def test_classification_metric_helpers_handle_multiclass_macro_scores():
    expected = ["cat", "cat", "dog", "bird"]
    predicted = ["cat", "dog", "dog", "dog"]

    assert accuracy_score(expected, predicted) == 0.5
    assert precision_score(expected, predicted) == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)
    assert recall_score(expected, predicted) == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert f1_score(expected, predicted) == pytest.approx(((2 / 3) + 0.5 + 0.0) / 3)


def test_classification_metrics_can_report_binary_positive_label_scores():
    metrics = classification_metrics(
        ["yes", "yes", "no", "no"],
        ["yes", "no", "yes", "no"],
        positive_label="yes",
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["positive_label"] == "yes"


def test_classification_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        classification_metrics(["a"], ["a", "b"])


def test_write_evaluation_writes_records_and_summary(tmp_path):
    records = [
        {"correct": True, "idx": 1, "expected": "yes", "predicted": "yes"},
        {"correct": False, "idx": 2, "expected": "yes", "predicted": "no"},
    ]

    write_evaluation(tmp_path, records)

    rows = [json.loads(line) for line in (tmp_path / "evaluation.jsonl").read_text().splitlines()]
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())

    assert rows == records
    assert summary["n"] == 2
    assert summary["n_correct"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["precision"] == pytest.approx(0.5)
    assert summary["recall"] == pytest.approx(0.25)
    assert summary["f1"] == pytest.approx(1 / 3)
