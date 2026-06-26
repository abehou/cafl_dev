"""Run CAFL agent tasks in a configured task environment."""

from __future__ import annotations

import asyncio
import sys
from argparse import ArgumentParser
from pathlib import Path

ENVS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ENVS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cafl import Cafl
from cafl.config import CaflConfig
from cafl.logging import EventLogger
from cafl.tools.retrieval import prepare_bm25_index
from cafl.utils.config_utils import (
    resolve_memory_dir,
    resolve_environment,
    select_tasks,
    task_prompt,
    template_vars_for_env,
    validate_env_config,
)
from cafl.utils.utils import get_path_time_signature, read_json, read_jsonl, safe_slug
from envs.eval import Evaluator


def prepare_run_dir(output_root: str | Path | None, env: str, *, data_dir: Path) -> Path:
    env_name = Path(env).name
    root = Path(output_root) if output_root is not None else data_dir / "runs"
    run_dir = root / f"{get_path_time_signature()}-{safe_slug(env_name)}-batch"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = ArgumentParser(description="Run CAFL agent tasks in a configured task environment.")
    parser.add_argument("--env", type=str, required=True, help="Environment name under envs/, or a path to its data dir.")
    parser.add_argument("--num_items", type=int, default=10, help="Number of items to run, or -1 for all items.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle task items with a controlled seed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--output-root", default=None, help="Run output root. Defaults to the environment data/runs folder.")
    parser.add_argument("--no-event-log", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar for batch runs.")
    parser.add_argument("--skip-build", action="store_true", help="Do not build or use a BM25 index for the configured corpus.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the BM25 index even if it already exists.")
    args = parser.parse_args()

    data_dir, config_path = resolve_environment(args.env, ENVS_ROOT)
    config = read_json(config_path)
    validate_env_config(config, data_dir)
    bm25_index_path = prepare_bm25_index(
        config,
        data_dir,
        rebuild=args.rebuild,
        skip=args.skip_build,
    )
    task_file = data_dir / config["task_file"]
    rows = select_tasks(read_jsonl(task_file), num_items=args.num_items, shuffle=args.shuffle, seed=args.seed)
    tasks = [task_prompt(row, config) for row in rows]
    run_dir = prepare_run_dir(args.output_root, args.env, data_dir=data_dir)
    template_vars = template_vars_for_env(config, data_dir, bm25_index_path=bm25_index_path)
    memory_dir = resolve_memory_dir(config, config_path)

    agent = Cafl(
        model=args.model,
        cafl_config=CaflConfig(
            output_schema=config.get("output_schema"),
            memory_dir=str(memory_dir),
            max_memory_chars=config.get("max_memory_chars", 12000),
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
            max_concurrency=args.max_concurrency,
            show_progress=not args.no_progress and bool(tasks),
            progress_desc=f"Running {Path(args.env).name}",
            **template_vars,
        )
    )

    # Can be modified to use a custom evaluator class for structured outputs or multi-step reasoning tasks.
    evaluator = Evaluator(
        ground_truth_field=config["ground_truth_field"],
        prediction_field=config.get("prediction_field", "answer"),
    )
    evaluation = [evaluator.evaluate(row, result) for row, result in zip(rows, results)]
    evaluator.write(run_dir, evaluation)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
