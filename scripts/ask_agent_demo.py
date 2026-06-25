"""Run Cafl on one question, or run 10 questions in parallel."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafl import Cafl
from cafl.logging import ConsoleEventLogger
from cafl.utils.utils import list_gemini_models


QUESTIONS = [
    "What is CAFL? Answer based on the local code",
    "Tell me the requirements needed for this repo without seeing requirements.txt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cafl on one question or 10 questions in parallel.")
    parser.add_argument("--question", default="What does this agent do?")
    parser.add_argument("--model", default="gemini-3-flash")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--parallel", action="store_true", help="Run the built-in list of 10 questions in parallel.")
    parser.add_argument("--list-models", action="store_true", help="List Gemini API models that support generateContent.")
    args = parser.parse_args()

    if args.list_models:
        for model_name in list_gemini_models():
            print(model_name)
        return

    agent = Cafl(model=args.model, event_logger=ConsoleEventLogger())
    if args.parallel:
        results = asyncio.run(
            agent.run_many_async(
                QUESTIONS,
                output_root=args.output_root,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        )
        for result in results:
            print(f"Q: {result.question}")
            print(f"A: {result.answer}")
            print(f"Saved: {result.output_dir}")
            print()
        return

    result = agent.run(args.question, output_root=args.output_root, max_tokens=args.max_tokens, timeout=args.timeout)
    print(result.answer)
    print(f"Saved: {result.output_dir}")


if __name__ == "__main__":
    main()
