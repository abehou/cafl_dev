"""Evaluator classes for configured CAFL task environments.
Can be overridden for custom evaluation logic, e.g. for structured outputs or multi-step reasoning tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cafl.utils.schema import extract_json_object
from cafl.utils.utils import append_jsonl
from envs.metrics import classification_metrics, normalize_for_comparison

DEFAULT_LABEL_ALIASES = {
    "yes": True,
    "y": True,
    "true": True,
    "t": True,
    "1": True,
    "no": False,
    "n": False,
    "false": False,
    "f": False,
    "0": False,
}


class Evaluator:
    def __init__(
        self,
        *,
        ground_truth_field: str,
        prediction_field: str = "answer",
        label_aliases: dict[str, Any] | None = None,
    ):
        self.ground_truth_field = ground_truth_field
        self.prediction_field = prediction_field
        aliases = DEFAULT_LABEL_ALIASES if label_aliases is None else label_aliases
        self.label_aliases = {
            normalize_for_comparison(key): value
            for key, value in aliases.items()
        }

    def evaluate(self, row: dict, result) -> dict:
        parsed = extract_json_object(result.answer)
        predicted = self.extract_prediction(parsed)
        expected = row.get(self.ground_truth_field)
        correct = self.is_correct(predicted, expected, row=row, result=result, parsed_answer=parsed)
        return {
            "idx": row.get("idx"),
            "question": result.question,
            "prediction_field": self.prediction_field,
            "ground_truth_field": self.ground_truth_field,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "parsed_answer": parsed,
            "raw_answer": result.answer,
            "output_dir": str(result.output_dir) if result.output_dir is not None else None,
        }

    def extract_prediction(self, parsed_answer: Any) -> Any:
        if isinstance(parsed_answer, dict):
            return parsed_answer.get(self.prediction_field)
        return None

    def is_correct(self, predicted: Any, expected: Any, **kwargs) -> bool:
        return self.normalize_label(predicted) == self.normalize_label(expected)

    def normalize_label(self, value: Any) -> Any:
        normalized = normalize_for_comparison(value)
        return self.label_aliases.get(normalized, normalized)

    def summarize(self, records: list[dict]) -> dict:
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
                    [self.normalize_label(record["expected"]) for record in records],
                    [self.normalize_label(record["predicted"]) for record in records],
                )
            )
            summary["n_correct"] = n_correct
        return summary

    def write(self, output_dir: Path, records: list[dict]) -> None:
        eval_path = output_dir / "evaluation.jsonl"
        for record in records:
            append_jsonl(eval_path, record)

        summary = self.summarize(records)
        (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
