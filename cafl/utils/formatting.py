"""Formatting helpers for CAFL output files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return repr(value)


def message_content(message: dict) -> str:
    return stringify(message.get("content", message.get("output_text", message.get("output", ""))))


def result_record(record_type: str, state: Any, content: str, *, message: dict | None = None) -> dict:
    record = {
        "type": record_type,
        "run_id": state.run_id,
        "task_id": state.task_id,
        "item_id": state.item_id,
        "content": content,
    }
    if message is not None:
        record["role"] = message.get("role")
        record["model"] = message.get("extra", {}).get("model_name")
    return record


def summary_dict(result: Any) -> dict:
    return {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "item_id": result.item_id,
        "question": result.question,
        "answer": result.answer,
        "status": result.status,
        "n_events": len(result.events),
        "n_calls": result.state.n_calls,
        "cost": result.state.cost,
        "output_dir": str(result.output_dir) if result.output_dir is not None else None,
    }


def write_summary(path: Path, result: Any) -> None:
    path.write_text(json.dumps(summary_dict(result), indent=2, ensure_ascii=False))


def write_batch_summary(path: Path, results: list[Any]) -> None:
    summary = {
        "run_id": path.parent.name,
        "status": "completed",
        "n_items": len(results),
        "items": [summary_dict(result) for result in results],
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
