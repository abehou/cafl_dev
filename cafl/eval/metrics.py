"""Metric helpers for CAFL environment evaluation."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

TOKEN_RE = re.compile(r"[a-z0-9]+")
SCORE_FIELDS = ("precision", "recall", "f1")
AGGREGATE_FIELDS = (
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "overall_precision",
    "overall_recall",
    "overall_f1",
)


def normalize_for_comparison(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value  # TODO: add more normalization logic if needed.


def _validate_pairs(expected: list[Any], predicted: list[Any]) -> None:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length.")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def mean_score(values: Iterable[int | float]) -> float:
    values = list(values)
    return safe_divide(sum(values), len(values))


def precision_recall_f1(*, matched: int, predicted: int, expected: int) -> dict:
    precision = safe_divide(matched, predicted)
    recall = safe_divide(matched, expected)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def text_token_overlap(predicted: str, expected: str) -> dict:
    predicted_tokens = _token_set(predicted)
    expected_tokens = _token_set(expected)
    if not predicted_tokens or not expected_tokens:
        return _zero_scores(SCORE_FIELDS)
    return precision_recall_f1(
        matched=len(predicted_tokens & expected_tokens),
        predicted=len(predicted_tokens),
        expected=len(expected_tokens),
    )


def aggregate_precision_recall_f1(
    metrics: Iterable[Mapping[str, int | float]],
    *,
    matched_count_field: str = "matched_count",
    predicted_count_field: str = "predicted_count",
    expected_count_field: str = "expected_count",
    prefix: str | None = None,
) -> dict:
    metrics = list(metrics)
    if not metrics:
        summary = _zero_scores(AGGREGATE_FIELDS)
    else:
        summary = _aggregate_scores(
            metrics,
            matched_count_field=matched_count_field,
            predicted_count_field=predicted_count_field,
            expected_count_field=expected_count_field,
        )
    return {f"{prefix}_{key}": value for key, value in summary.items()} if prefix else summary


def accuracy_score(expected: list[Any], predicted: list[Any]) -> float:
    expected, predicted = _normalized_pairs(expected, predicted)
    return _accuracy_from_pairs(expected, predicted)


def classification_metrics(
    expected: list[Any],
    predicted: list[Any],
    *,
    labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> dict:
    expected, predicted = _normalized_pairs(expected, predicted)
    accuracy = _accuracy_from_pairs(expected, predicted)

    if positive_label is not None:
        label = normalize_for_comparison(positive_label)
        scores = _classification_scores_for_label(expected, predicted, label)
        return {
            "n": len(expected),
            "accuracy": accuracy,
            **scores,
            "average": "binary",
            "positive_label": label,
        }

    metric_labels = _metric_labels(expected, predicted, labels)
    per_label = {
        label: _classification_scores_for_label(expected, predicted, label)
        for label in metric_labels
    }
    return {
        "n": len(expected),
        "accuracy": accuracy,
        **_macro_scores(per_label.values()),
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
    return _classification_score_value("precision", expected, predicted, labels, positive_label)


def recall_score(
    expected: list[Any],
    predicted: list[Any],
    *,
    labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> float:
    return _classification_score_value("recall", expected, predicted, labels, positive_label)


def f1_score(
    expected: list[Any],
    predicted: list[Any],
    *,
    labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> float:
    return _classification_score_value("f1", expected, predicted, labels, positive_label)


def _normalized_pairs(expected: list[Any], predicted: list[Any]) -> tuple[list[Any], list[Any]]:
    _validate_pairs(expected, predicted)
    return (
        [normalize_for_comparison(value) for value in expected],
        [normalize_for_comparison(value) for value in predicted],
    )


def _accuracy_from_pairs(expected: list[Any], predicted: list[Any]) -> float:
    return safe_divide(sum(1 for gold, pred in zip(expected, predicted) if gold == pred), len(expected))


def _metric_labels(expected: list[Any], predicted: list[Any], labels: list[Any] | None = None) -> list[Any]:
    if labels is not None:
        return [normalize_for_comparison(label) for label in labels]
    return sorted({*expected, *predicted}, key=repr)


def _classification_scores_for_label(expected: list[Any], predicted: list[Any], label: Any) -> dict:
    return precision_recall_f1(**_label_match_counts(expected, predicted, label))


def _label_match_counts(expected: list[Any], predicted: list[Any], label: Any) -> dict:
    matched = sum(1 for gold, pred in zip(expected, predicted) if gold == label and pred == label)
    false_positives = sum(1 for gold, pred in zip(expected, predicted) if gold != label and pred == label)
    false_negatives = sum(1 for gold, pred in zip(expected, predicted) if gold == label and pred != label)
    return {
        "matched": matched,
        "predicted": matched + false_positives,
        "expected": matched + false_negatives,
    }


def _macro_scores(metrics: Iterable[Mapping[str, int | float]]) -> dict:
    metrics = list(metrics)
    return {
        field: mean_score(metric[field] for metric in metrics)
        for field in SCORE_FIELDS
    }


def _aggregate_scores(
    metrics: list[Mapping[str, int | float]],
    *,
    matched_count_field: str,
    predicted_count_field: str,
    expected_count_field: str,
) -> dict:
    overall = precision_recall_f1(
        matched=sum(metric[matched_count_field] for metric in metrics),
        predicted=sum(metric[predicted_count_field] for metric in metrics),
        expected=sum(metric[expected_count_field] for metric in metrics),
    )
    return {
        "mean_precision": mean_score(metric["precision"] for metric in metrics),
        "mean_recall": mean_score(metric["recall"] for metric in metrics),
        "mean_f1": mean_score(metric["f1"] for metric in metrics),
        "overall_precision": overall["precision"],
        "overall_recall": overall["recall"],
        "overall_f1": overall["f1"],
    }


def _classification_score_value(
    field: str,
    expected: list[Any],
    predicted: list[Any],
    labels: list[Any] | None,
    positive_label: Any | None,
) -> float:
    return classification_metrics(
        expected,
        predicted,
        labels=labels,
        positive_label=positive_label,
    )[field]


def _token_set(text: Any) -> set[str]:
    return set(TOKEN_RE.findall(str(normalize_for_comparison(text))))


def _zero_scores(fields: Iterable[str]) -> dict:
    return dict.fromkeys(fields, 0.0)
