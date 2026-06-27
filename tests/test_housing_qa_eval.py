from types import SimpleNamespace
import json

import pytest

from cafl.utils.config_utils import task_prompt
from envs.housing_qa.eval import HousingQAEvaluator, evaluate_run


def test_housing_qa_evaluator_scores_answer_and_statute_matches():
    row = {
        "idx": 1,
        "answer": "Yes",
        "statutes": [
            {
                "statute_idx": 101,
                "citation": "ALA. CODE § 35-9A-421(A)",
                "excerpt": "The landlord may deliver written notice to terminate the lease.",
            },
            {
                "statute_idx": 202,
                "citation": "ALA. CODE § 35-9A-461(C)",
                "excerpt": "Service may be made by posting and mailing the notice.",
            },
        ],
    }
    result = SimpleNamespace(
        question="Is notice required?",
        answer=(
            '{"answer": "True", "statutes": ['
            '{"statute_index": 101, "citation": "AL Code § 35-9A-421 (2021)", '
            '"excerpt": "landlord may deliver written notice to terminate the lease"}'
            "]}"
        ),
        output_dir=None,
    )

    record = HousingQAEvaluator(ground_truth_field="answer").evaluate(row, result)

    assert record["correct"] is True
    assert record["statute_precision"] == 1.0
    assert record["statute_recall"] == 0.5
    assert record["statute_f1"] == pytest.approx(2 / 3)
    assert record["passage_precision"] == 1.0
    assert record["passage_recall"] == 0.5
    assert record["passage_f1"] == pytest.approx(2 / 3)
    assert record["statute_match"]["matched_gold"] == [0]
    assert record["passage_match"]["matches"][0]["predicted_index"] == 0
    assert record["passage_match"]["matches"][0]["gold_index"] == 0


def test_housing_qa_evaluator_separates_statute_match_from_passage_match():
    row = {
        "answer": "No",
        "statutes": [
            {
                "statute_idx": 101,
                "citation": "ALA. CODE § 35-9A-421(A)",
                "excerpt": "The landlord may deliver written notice to terminate the lease.",
            }
        ],
    }
    result = SimpleNamespace(
        question="Is notice required?",
        answer=(
            '{"answer": "False", "statutes": ['
            '{"statute_index": 101, "citation": "AL Code § 35-9A-421 (2021)", '
            '"excerpt": "A tenant shall keep plumbing fixtures clean and safe"}'
            "]}"
        ),
        output_dir=None,
    )

    record = HousingQAEvaluator(ground_truth_field="answer").evaluate(row, result)

    assert record["correct"] is True
    assert record["statute_f1"] == 1.0
    assert record["passage_f1"] == 0.0
    assert record["passage_match"]["best_overlaps"][0]["f1"] == 0.0


def test_housing_qa_evaluator_summary_includes_evidence_metrics():
    records = [
        {
            "correct": True,
            "expected": "Yes",
            "predicted": "True",
            "statute_match": {
                "gold_count": 1,
                "predicted_count": 1,
                "matched_count": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
            "passage_match": {
                "gold_count": 1,
                "predicted_count": 1,
                "matched_count": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
        },
        {
            "correct": False,
            "expected": "No",
            "predicted": "True",
            "statute_match": {
                "gold_count": 2,
                "predicted_count": 1,
                "matched_count": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "passage_match": {
                "gold_count": 2,
                "predicted_count": 1,
                "matched_count": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        },
    ]

    summary = HousingQAEvaluator(ground_truth_field="answer").summarize(records)

    assert summary["accuracy"] == 0.5
    assert summary["statute_by_question_f1"] == 0.5
    assert summary["statute_pooled_precision"] == 0.5
    assert summary["statute_pooled_recall"] == pytest.approx(1 / 3)
    assert summary["passage_by_question_f1"] == 0.5


def test_evaluate_run_writes_housing_qa_records_and_summary(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {
        "task_file": "questions.jsonl",
        "corpus_dir": "corpus",
        "task_field": "question",
        "ground_truth_field": "answer",
        "output_schema": {"answer": {"type": "string"}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    row = {
        "idx": 7,
        "question": "(For the state of Alabama), Is notice required?",
        "answer": "Yes",
        "statutes": [
            {
                "statute_idx": 101,
                "citation": "ALA. CODE § 35-9A-421(A)",
                "excerpt": "The landlord may deliver written notice to terminate the lease.",
            }
        ],
    }
    (data_dir / "questions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (data_dir / "corpus").mkdir()
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    answer = {
        "answer": "True",
        "statutes": [
            {
                "statute_index": 101,
                "citation": "AL Code § 35-9A-421 (2021)",
                "excerpt": "landlord may deliver written notice to terminate the lease",
            }
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "question": task_prompt(row, config),
                        "answer": json.dumps(answer),
                        "output_dir": str(run_dir / "item-000"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = evaluate_run(run_dir, data_dir=data_dir, config_path=config_path)

    rows = [json.loads(line) for line in (run_dir / "evaluation.jsonl").read_text().splitlines()]
    written_summary = json.loads((run_dir / "evaluation_summary.json").read_text())
    assert len(rows) == 1
    assert rows[0]["idx"] == 7
    assert summary == written_summary
    assert written_summary["accuracy"] == 1.0
    assert written_summary["statute_pooled_f1"] == 1.0
    assert written_summary["passage_pooled_f1"] == 1.0
