"""Run CAFL agent tasks in a configured task environment."""

from __future__ import annotations

import asyncio
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

ENVS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ENVS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cafl import Cafl
from cafl.config import DEFAULT_SYSTEM_TEMPLATE, CaflConfig
from cafl.logging import EventLogger
from cafl.utils.utils import get_path_time_signature, read_json, read_jsonl, safe_slug
from envs.eval import evaluate_result, write_evaluation

def resolve_data_dir(env: str) -> Path:
    env_path = Path(env).expanduser()
    candidates = []
    if env_path.exists():
        candidates.extend([env_path, env_path / "data"])
    candidates.append(ENVS_ROOT / env / "data")

    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find config.json for environment {env!r}.")


def prepare_run_dir(output_root: str | Path, env: str) -> Path:
    env_name = Path(env).name
    run_dir = Path(output_root) / f"{get_path_time_signature()}-{safe_slug(env_name)}-batch"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_system_template(config: dict, data_dir: Path) -> str:
    corpus_dir = config.get("corpus_dir")
    corpus_text = ""
    if corpus_dir:
        corpus_text = f"\nThe local corpus directory is: {data_dir / corpus_dir}\n"

    return (
        DEFAULT_SYSTEM_TEMPLATE
        + "\n\n## Task Environment Instructions\n"
        + config.get("agent_additional_system_prompt", "")
        + corpus_text
    )


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
    generic_instruction = config.get("generic_instruction", None)
    if generic_instruction:
        return f"{generic_instruction}\n\n{row[task_field]}"
    task = row[task_field]
    return f"Task:\n{task}"


def main() -> None:
    parser = ArgumentParser(description="Run CAFL agent tasks in a configured task environment.")
    parser.add_argument("--env", type=str, required=True, help="Environment name under envs/, or a path to its data dir.")
    parser.add_argument("--num_items", type=int, default=10, help="Number of items to run, or -1 for all items.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle task items with a controlled seed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--no-event-log", action="store_true")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.env)
    config = read_json(data_dir / "config.json")
    task_file = data_dir / config["task_file"]
    rows = select_tasks(read_jsonl(task_file), num_items=args.num_items, shuffle=args.shuffle, seed=args.seed)
    tasks = [task_prompt(row, config) for row in rows]
    run_dir = prepare_run_dir(args.output_root, args.env)

    agent = Cafl(
        model=args.model,
        cafl_config=CaflConfig(
            system_template=build_system_template(config, data_dir),
            output_schema=config.get("output_schema"),
        ),
        event_logger=None if args.no_event_log else EventLogger(run_dir / "events.log"),
    )
    results = asyncio.run(
        agent.run_many_async(
            tasks,
            output_root=None,
            output_dir=run_dir,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    )

    prediction_field = config.get("prediction_field", "answer")
    evaluation = [
        evaluate_result(
            row,
            result,
            ground_truth_field=config["ground_truth_field"],
            prediction_field=prediction_field,
        )
        for row, result in zip(rows, results)
    ]
    write_evaluation(run_dir, evaluation)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
