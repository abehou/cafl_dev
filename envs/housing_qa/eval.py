"""Housing QA evaluation with answer and evidence matching metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cafl.eval.evaluator import Evaluator
from cafl.eval.metrics import (
    aggregate_precision_recall_f1,
    normalize_for_comparison,
    precision_recall_f1,
    text_token_overlap,
)
from cafl.utils.config_utils import task_prompt
from cafl.utils.utils import read_json, read_jsonl


class HousingQAEvaluator(Evaluator):
    def __init__(
        self,
        *,
        ground_truth_field: str,
        prediction_field: str = "answer",
        statutes_field: str = "statutes",
        passage_overlap_threshold: float = 0.5,
        label_aliases: dict[str, Any] | None = None,
    ):
        super().__init__(
            ground_truth_field=ground_truth_field,
            prediction_field=prediction_field,
            label_aliases=label_aliases,
        )
        self.statutes_field = statutes_field
        self.passage_overlap_threshold = passage_overlap_threshold

    def evaluate(self, row: dict, result) -> dict:
        record = super().evaluate(row, result)
        gold_statutes = _normalize_statute_records(row.get(self.statutes_field) or [])
        predicted_statutes = _normalize_statute_records(
            (record["parsed_answer"] or {}).get(self.statutes_field) or []
        )
        statute_match = _score_statute_matches(gold_statutes, predicted_statutes)
        passage_match = _score_passage_matches(
            gold_statutes,
            predicted_statutes,
            threshold=self.passage_overlap_threshold,
        )
        return {
            **record,
            "gold_statutes": gold_statutes,
            "predicted_statutes": predicted_statutes,
            "statute_match": statute_match,
            "statute_precision": statute_match["precision"],
            "statute_recall": statute_match["recall"],
            "statute_f1": statute_match["f1"],
            "passage_match": passage_match,
            "passage_precision": passage_match["precision"],
            "passage_recall": passage_match["recall"],
            "passage_f1": passage_match["f1"],
        }

    def summarize(self, records: list[dict]) -> dict:
        summary = super().summarize(records)
        summary.update(_summarize_match_metrics(records, "statute_match", "statute"))
        summary.update(_summarize_match_metrics(records, "passage_match", "passage"))
        return summary


def evaluate_run(
    run_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict:
    run_dir = Path(run_dir)
    env_dir = Path(__file__).resolve().parent
    data_dir = Path(data_dir) if data_dir is not None else env_dir / "data"
    config_path = Path(config_path) if config_path is not None else env_dir / "config.json"
    config = read_json(config_path)
    rows = read_jsonl(data_dir / config["task_file"])
    run_summary = read_json(run_dir / "summary.json")
    run_items = run_summary.get("items") or [run_summary]

    rows_by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_prompt.setdefault(task_prompt(row, config), []).append(row)

    evaluator = HousingQAEvaluator(
        ground_truth_field=config["ground_truth_field"],
        prediction_field=config.get("prediction_field", "answer"),
        statutes_field=config.get("statutes_field", "statutes"),
        passage_overlap_threshold=config.get("passage_overlap_threshold", 0.5),
        label_aliases=config.get("label_aliases"),
    )
    evaluation = []
    for item in run_items:
        prompt = item.get("question")
        matching_rows = rows_by_prompt.get(prompt) or []
        if not matching_rows:
            raise ValueError(f"Could not match run item back to task row: {prompt!r}")
        row = matching_rows.pop(0)
        result = SimpleNamespace(
            question=prompt,
            answer=item.get("answer", ""),
            output_dir=item.get("output_dir"),
        )
        evaluation.append(evaluator.evaluate(row, result))

    for path in (run_dir / "evaluation.jsonl", run_dir / "evaluation_summary.json"):
        path.unlink(missing_ok=True)
    evaluator.write(run_dir, evaluation)
    return read_json(run_dir / "evaluation_summary.json")

def _normalize_statute_records(statutes: list[dict]) -> list[dict]:
    return [
        {
            "statute_index": _first_present(statute, "statute_index", "statute_idx", "idx"),
            "citation": statute.get("citation"),
            "excerpt": statute.get("excerpt", ""),
            "key": _statute_key(statute),
        }
        for statute in statutes
        if isinstance(statute, dict)
    ]


def _score_statute_matches(gold_statutes: list[dict], predicted_statutes: list[dict]) -> dict:
    gold_keys = {statute["key"] for statute in gold_statutes if statute["key"] is not None}
    predicted_keys = {statute["key"] for statute in predicted_statutes if statute["key"] is not None}
    matched_keys = gold_keys & predicted_keys
    matched_gold = [
        index for index, statute in enumerate(gold_statutes)
        if statute["key"] in matched_keys
    ]
    matched_predicted = [
        index for index, statute in enumerate(predicted_statutes)
        if statute["key"] in matched_keys
    ]
    return {
        "gold_count": len(gold_keys),
        "predicted_count": len(predicted_keys),
        "matched_count": len(matched_keys),
        **precision_recall_f1(
            matched=len(matched_keys),
            predicted=len(predicted_keys),
            expected=len(gold_keys),
        ),
        "matched_gold": matched_gold,
        "matched_predicted": matched_predicted,
        "matched_keys": sorted(matched_keys),
    }


def _score_passage_matches(
    gold_statutes: list[dict],
    predicted_statutes: list[dict],
    *,
    threshold: float,
) -> dict:
    gold_by_key: dict[str, list[tuple[int, dict]]] = {}
    for gold_index, gold in enumerate(gold_statutes):
        key = gold["key"]
        if key is not None:
            gold_by_key.setdefault(key, []).append((gold_index, gold))

    scored_pairs = []
    best_overlaps = []
    for predicted_index, predicted in enumerate(predicted_statutes):
        key = predicted["key"]
        if key is None:
            best_overlaps.append({"predicted_index": predicted_index, "gold_index": None, "f1": 0.0})
            continue
        candidates = [
            (gold_index, text_token_overlap(predicted["excerpt"], gold["excerpt"]))
            for gold_index, gold in gold_by_key.get(key, ())
        ]
        if not candidates:
            best_overlaps.append({"predicted_index": predicted_index, "gold_index": None, "f1": 0.0})
            continue
        gold_index, scores = max(candidates, key=lambda item: item[1]["f1"])
        best_overlaps.append({
            "predicted_index": predicted_index,
            "gold_index": gold_index,
            **scores,
        })
        if scores["f1"] >= threshold:
            scored_pairs.append((scores["f1"], predicted_index, gold_index, scores))

    matches = []
    used_predicted = set()
    used_gold = set()
    for _score, predicted_index, gold_index, scores in sorted(scored_pairs, reverse=True):
        if predicted_index in used_predicted or gold_index in used_gold:
            continue
        used_predicted.add(predicted_index)
        used_gold.add(gold_index)
        matches.append({
            "predicted_index": predicted_index,
            "gold_index": gold_index,
            **scores,
        })

    return {
        "gold_count": len(gold_statutes),
        "predicted_count": len(predicted_statutes),
        "matched_count": len(matches),
        **precision_recall_f1(
            matched=len(matches),
            predicted=len(predicted_statutes),
            expected=len(gold_statutes),
        ),
        "matches": sorted(matches, key=lambda match: match["predicted_index"]),
        "best_overlaps": best_overlaps,
    }


def _summarize_match_metrics(records: list[dict], field: str, prefix: str) -> dict:
    scores = aggregate_precision_recall_f1(
        (record[field] for record in records if field in record),
        expected_count_field="gold_count",
    )
    return {
        f"{prefix}_by_question_precision": scores["mean_precision"],
        f"{prefix}_by_question_recall": scores["mean_recall"],
        f"{prefix}_by_question_f1": scores["mean_f1"],
        f"{prefix}_pooled_precision": scores["overall_precision"],
        f"{prefix}_pooled_recall": scores["overall_recall"],
        f"{prefix}_pooled_f1": scores["overall_f1"],
    }


def _statute_key(statute: dict) -> str | None:
    statute_index = _first_present(statute, "statute_index", "statute_idx", "idx")
    if statute_index is not None:
        return f"idx:{statute_index}"
    citation = statute.get("citation")
    if not citation:
        return None
    section = _citation_section(citation)
    return f"citation:{section or normalize_for_comparison(citation)}"


def _citation_section(citation: str) -> str | None:
    match = re.search(r"\b\d+[a-z]?-\d+[a-z]?-\d+(?:\.\d+)?\b", citation.casefold())
    return match.group(0) if match else None


def _first_present(statute: dict, *keys: str) -> Any:
    for key in keys:
        value = statute.get(key)
        if value is not None:
            return str(value)
    return None

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Housing QA run directory.")
    parser.add_argument("run_dir", help="Run directory containing summary.json.")
    args = parser.parse_args(argv)

    summary = evaluate_run(args.run_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
