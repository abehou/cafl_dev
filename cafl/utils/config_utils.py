"""Helpers for loading and validating configured task environments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from cafl.tools.retrieval import bm25_tool_instruction

REQUIRED_ENV_CONFIG_FIELDS = ("task_file", "corpus_dir", "task_field", "ground_truth_field", "output_schema")


def resolve_environment(env: str, envs_root: Path) -> tuple[Path, Path]:
    env_path = Path(env).expanduser()
    candidates: list[tuple[Path, Path]] = []
    if env_path.exists():
        candidates.extend(
            [
                (env_path / "data", env_path / "config.json"),
                (env_path, env_path / "config.json"),
                (env_path, env_path.parent / "config.json"),
                (env_path / "data", env_path / "data" / "config.json"),
            ]
        )

    env_root = envs_root / env
    candidates.extend(
        [
            (env_root / "data", env_root / "config.json"),
            (env_root / "data", env_root / "data" / "config.json"),
        ]
    )

    for data_dir, config_path in candidates:
        if data_dir.exists() and config_path.exists():
            return data_dir.resolve(), config_path.resolve()
    raise FileNotFoundError(f"Could not find config.json for environment {env!r}.")


def validate_env_config(config: dict[str, Any], data_dir: Path) -> None:
    missing = [field for field in REQUIRED_ENV_CONFIG_FIELDS if field not in config]
    errors = []
    if missing:
        errors.append(f"Missing required config fields: {', '.join(missing)}")

    task_file = data_dir / config["task_file"] if "task_file" in config else None
    if task_file is not None and not task_file.exists():
        errors.append(f"Configured task_file does not exist: {task_file}")

    corpus_dir = data_dir / config["corpus_dir"] if "corpus_dir" in config else None
    if corpus_dir is not None and not corpus_dir.exists():
        errors.append(f"Configured corpus_dir does not exist: {corpus_dir}")

    if "output_schema" in config and not isinstance(config["output_schema"], dict):
        errors.append("Config field output_schema must be a JSON object.")

    if task_file is not None and task_file.exists() and {"task_field", "ground_truth_field"} <= set(config):
        sample = first_jsonl_record(task_file)
        if sample is None:
            errors.append(f"Configured task_file is empty: {task_file}")
        else:
            for field_name in ("task_field", "ground_truth_field"):
                row_field = config[field_name]
                if row_field not in sample:
                    errors.append(f"Configured {field_name} {row_field!r} is not present in the first task row.")

    if errors:
        raise ValueError("Invalid config.json:\n- " + "\n- ".join(errors))


def resolve_memory_dir(config: dict[str, Any], config_path: Path) -> Path:
    memory_dir = Path(config.get("memory_dir", "memory"))
    if memory_dir.is_absolute():
        return memory_dir
    return config_path.parent / memory_dir


def first_jsonl_record(path: Path) -> dict | None:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    return None


def select_tasks(rows: list[dict], *, num_items: int, shuffle: bool, seed: int) -> list[dict]:
    rows = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if num_items >= 0:
        rows = rows[:num_items]
    return rows


def task_prompt(row: dict, config: dict) -> str:
    task_field = config.get("task_field", "task")
    generic_instruction = config.get("generic_instruction")
    if generic_instruction:
        return f"{generic_instruction}\n\n{row[task_field]}"
    return f"Task:\n{row[task_field]}"


def template_vars_for_env(
    config: dict[str, Any],
    data_dir: Path,
    *,
    bm25_index_path: Path | None = None,
) -> dict[str, str]:
    corpus_dir = data_dir / config["corpus_dir"] if config.get("corpus_dir") else ""
    return {
        "task_environment_instructions": config.get("agent_additional_system_prompt", ""),
        "corpus_dir": str(corpus_dir),
        "bm25_instructions": bm25_tool_instruction(bm25_index_path) if bm25_index_path is not None else "",
    }
