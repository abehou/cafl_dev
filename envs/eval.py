"""Evaluation helpers for configured CAFL task environments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cafl.utils.schema import extract_json_object
from cafl.utils.utils import append_jsonl


def normalize_for_comparison(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value #TODO add more normalization logics if needed


def _validate_pairs(expected: list[Any], predicted: list[Any]) -> None:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length.")


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _normalized_pairs(expected: list[Any], predicted: list[Any]) -> tuple[list[Any], list[Any]]:
    return (
        [normalize_for_comparison(value) for value in expected],
        [normalize_for_comparison(value) for value in predicted],
    )


def _metric_labels(expected: list[Any], predicted: list[Any], labels: list[Any] | None = None) -> list[Any]:
    if labels is not None:
        return [normalize_for_comparison(label) for label in labels]
    return sorted({*expected, *predicted}, key=repr)


def _label_counts(expected: list[Any], predicted: list[Any], label: Any) -> dict:
    tp = sum(1 for gold, pred in zip(expected, predicted) if gold == label and pred == label)
    fp = sum(1 for gold, pred in zip(expected, predicted) if gold != label and pred == label)
    fn = sum(1 for gold, pred in zip(expected, predicted) if gold == label and pred != label)
    return {"tp": tp, "fp": fp, "fn": fn}


def _precision_recall_f1(counts: dict) -> dict:
    precision = _safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def accuracy_score(expected: list[Any], predicted: list[Any]) -> float:
    expected, predicted = _normalized_pairs(expected, predicted)
    _validate_pairs(expected, predicted)
    return _safe_divide(sum(1 for gold, pred in zip(expected, predicted) if gold == pred), len(expected))


def classification_metrics(
    expected: list[Any],
    predicted: list[Any],
    *,
    labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> dict:
    expected, predicted = _normalized_pairs(expected, predicted)
    _validate_pairs(expected, predicted)

    accuracy = accuracy_score(expected, predicted)
    if positive_label is not None:
        label = normalize_for_comparison(positive_label)
        scores = _precision_recall_f1(_label_counts(expected, predicted, label))
        return {
            "n": len(expected),
            "accuracy": accuracy,
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "average": "binary",
            "positive_label": label,
        }

    metric_labels = _metric_labels(expected, predicted, labels)
    per_label = {
        label: _precision_recall_f1(_label_counts(expected, predicted, label))
        for label in metric_labels
    }
    n_labels = len(metric_labels)
    return {
        "n": len(expected),
        "accuracy": accuracy,
        "precision": _safe_divide(sum(scores["precision"] for scores in per_label.values()), n_labels),
        "recall": _safe_divide(sum(scores["recall"] for scores in per_label.values()), n_labels),
        "f1": _safe_divide(sum(scores["f1"] for scores in per_label.values()), n_labels),
        "average": "macro",
        "labels": metric_labels,
        "per_label": per_label,
    }


def precision_score(
    expected: list[Any],
    predicted: list[Any],
    *,
    labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> float:
    return classification_metrics(expected, predicted, labels=labels, positive_label=positive_label)["precision"]


def recall_score(
    expected: list[Any],
    predicted: list[Any],
    *,
    labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> float:
    return classification_metrics(expected, predicted, labels=labels, positive_label=positive_label)["recall"]


def f1_score(
    expected: list[Any],
    predicted: list[Any],
    *,
    labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> float:
    return classification_metrics(expected, predicted, labels=labels, positive_label=positive_label)["f1"]


def evaluate_result(row: dict, result, *, ground_truth_field: str, prediction_field: str = "answer") -> dict:
    parsed = extract_json_object(result.answer)
    predicted = parsed.get(prediction_field) if isinstance(parsed, dict) else None
    expected = row.get(ground_truth_field)
    correct = normalize_for_comparison(predicted) == normalize_for_comparison(expected)
    return {
        "idx": row.get("idx"),
        "question": result.question,
        "prediction_field": prediction_field,
        "ground_truth_field": ground_truth_field,
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "parsed_answer": parsed,
        "raw_answer": result.answer,
        "output_dir": str(result.output_dir) if result.output_dir is not None else None,
    }


def write_evaluation(output_dir: Path, records: list[dict]) -> None:
    eval_path = output_dir / "evaluation.jsonl"
    for record in records:
        append_jsonl(eval_path, record)

    n = len(records)
    n_correct = sum(1 for record in records if record["correct"])
    summary = {
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
    }
    if records and all("expected" in record and "predicted" in record for record in records):
        summary.update(
            classification_metrics(
                [record["expected"] for record in records],
                [record["predicted"] for record in records],
            )
        )
        summary["n_correct"] = n_correct
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
