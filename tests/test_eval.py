import json
from types import SimpleNamespace

import pytest

from cafl.eval.evaluator import Evaluator
from cafl.eval.metrics import (
    accuracy_score,
    aggregate_precision_recall_f1,
    classification_metrics,
    f1_score,
    mean_score,
    normalize_for_comparison,
    precision_score,
    precision_recall_f1,
    recall_score,
    text_token_overlap,
)


def test_normalize_for_comparison_is_generic_text_normalization():
    assert normalize_for_comparison("  Hello   WORLD ") == "hello world"
    assert normalize_for_comparison(3) == 3
    assert normalize_for_comparison(True) is True


def test_evaluator_extracts_configured_prediction_field():
    row = {"idx": 7, "gold": "Rent Control"}
    result = SimpleNamespace(
        question="What law applies?",
        answer='{"label": " rent   CONTROL "}',
        output_dir="/tmp/item-007",
    )
    evaluator = Evaluator(ground_truth_field="gold", prediction_field="label")

    record = evaluator.evaluate(row, result)

    assert record["correct"] is True
    assert record["predicted"] == " rent   CONTROL "
    assert record["expected"] == "Rent Control"
    assert record["parsed_answer"] == {"label": " rent   CONTROL "}


def test_evaluator_treats_common_boolean_label_aliases_as_equivalent():
    row = {"idx": 7, "gold": "Yes"}
    result = SimpleNamespace(
        question="Is it allowed?",
        answer='{"answer": "True"}',
        output_dir=None,
    )
    evaluator = Evaluator(ground_truth_field="gold")

    record = evaluator.evaluate(row, result)

    assert record["correct"] is True
    assert record["expected"] == "Yes"
    assert record["predicted"] == "True"


def test_evaluator_can_disable_default_label_aliases():
    row = {"idx": 7, "gold": "Yes"}
    result = SimpleNamespace(
        question="Is it allowed?",
        answer='{"answer": "True"}',
        output_dir=None,
    )
    evaluator = Evaluator(ground_truth_field="gold", label_aliases={})

    record = evaluator.evaluate(row, result)

    assert record["correct"] is False


def test_evaluator_can_override_correctness_logic():
    class ContainsEvaluator(Evaluator):
        def is_correct(self, predicted, expected, **kwargs):
            return expected.casefold() in predicted.casefold()

    row = {"gold": "rent control"}
    result = SimpleNamespace(
        question="What law applies?",
        answer='{"answer": "The answer probably involves rent control protections."}',
        output_dir=None,
    )

    record = ContainsEvaluator(ground_truth_field="gold").evaluate(row, result)

    assert record["correct"] is True


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


def test_precision_recall_f1_handles_count_metrics():
    metrics = precision_recall_f1(matched=2, predicted=4, expected=3)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(4 / 7)
    assert precision_recall_f1(matched=0, predicted=0, expected=3) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_text_token_overlap_scores_word_set_overlap():
    metrics = text_token_overlap(
        "Landlord may deliver written notice to terminate.",
        "The landlord may deliver notice before termination.",
    )

    assert metrics["precision"] == pytest.approx(4 / 7)
    assert metrics["recall"] == pytest.approx(4 / 7)
    assert metrics["f1"] == pytest.approx(4 / 7)
    assert text_token_overlap("", "notice")["f1"] == 0.0


def test_mean_score_handles_empty_inputs():
    assert mean_score([1.0, 2.0, 3.0]) == 2.0
    assert mean_score([]) == 0.0


def test_aggregate_precision_recall_f1_reports_mean_and_overall_metrics():
    metrics = [
        {
            "gold_count": 1,
            "predicted_count": 1,
            "matched_count": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        {
            "gold_count": 2,
            "predicted_count": 1,
            "matched_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
    ]

    summary = aggregate_precision_recall_f1(
        metrics,
        expected_count_field="gold_count",
        prefix="evidence",
    )

    assert summary["evidence_mean_f1"] == 0.5
    assert summary["evidence_overall_precision"] == 0.5
    assert summary["evidence_overall_recall"] == pytest.approx(1 / 3)


def test_write_evaluation_writes_records_and_summary(tmp_path):
    records = [
        {"correct": True, "idx": 1, "expected": "yes", "predicted": "yes"},
        {"correct": False, "idx": 2, "expected": "yes", "predicted": "no"},
    ]

    evaluator = Evaluator(ground_truth_field="gold")

    evaluator.write(tmp_path, records)

    rows = [json.loads(line) for line in (tmp_path / "evaluation.jsonl").read_text().splitlines()]
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())

    assert rows == records
    assert summary["n"] == 2
    assert summary["n_correct"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["precision"] == pytest.approx(0.5)
    assert summary["recall"] == pytest.approx(0.25)
    assert summary["f1"] == pytest.approx(1 / 3)


def test_write_evaluation_summary_uses_label_aliases(tmp_path):
    records = [
        {"correct": True, "idx": 1, "expected": "Yes", "predicted": "True"},
        {"correct": True, "idx": 2, "expected": "No", "predicted": "False"},
    ]

    evaluator = Evaluator(ground_truth_field="gold")

    evaluator.write(tmp_path, records)

    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())

    assert summary["n_correct"] == 2
    assert summary["accuracy"] == 1.0
