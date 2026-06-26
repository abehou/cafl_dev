"""Validation and coaching checks for the CAFL run loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from .memory import format_working_memory
from .utils.formatting import normalize_action
from .utils.schema import validate_output_text


@dataclass
class ValidationResult:
    message: dict | None = None
    error: str | None = None


class RunValidator:
    def __init__(self, agent: Any):
        self.agent = agent

    def before_action(self, state: Any, action: dict) -> ValidationResult:
        return check_repeated_action(self.agent, state, action)

    def after_final_answer(self, state: Any, extract_answer: Callable[[Any], tuple[str, dict]]) -> ValidationResult:
        return check_output_schema(self.agent, state, extract_answer)


def check_repeated_action(agent: Any, state: Any, action: dict) -> ValidationResult:
    limit = agent.cafl_config.repeated_tool_call_limit
    if limit <= 0:
        return ValidationResult()

    counts = state.memory.setdefault("action_counts", {})
    signature = action_signature(action)
    count = counts.get(signature, 0) + 1
    counts[signature] = count
    if count < limit:
        return ValidationResult()

    memory_summary = format_working_memory(state.memory)
    content = (
        f"You repeated the same tool call {count} times.\n\n"
        f"Repeated tool call:\n{action!r}\n\n"
    )
    if memory_summary:
        content += f"{memory_summary}\n\n"
    content += (
        "Do not repeat this exact tool call again. Recalibrate your retrieval strategy before using another tool:\n"
        "1. What did the previous result show?\n"
        "2. Why was it insufficient or noisy?\n"
        "3. Choose exactly one next move: answer now, inspect a specific result, or run a genuinely different search."
    )
    return ValidationResult(
        message=agent.model.format_message(
            role="user",
            content=content,
            extra={
                "interrupt_type": "RepeatedToolCallCalibration",
                "repeat_count": count,
                "action": action,
            },
        ),
        error="RepeatedToolCallCalibration",
    )


def check_output_schema(agent: Any, state: Any, extract_answer: Callable[[Any], tuple[str, dict]]) -> ValidationResult:
    if agent.cafl_config.output_schema is None:
        return ValidationResult()
    if state.output_validation_failures >= agent.cafl_config.output_validation_retries:
        return ValidationResult()

    answer, _message = extract_answer(state)
    _parsed, error = validate_output_text(answer, agent.cafl_config.output_schema)
    if error is None:
        return ValidationResult()

    state.output_validation_failures += 1
    return ValidationResult(
        message=agent.model.format_message(
            role="user",
            content=(
                "Your previous final answer did not match the required JSON output schema.\n"
                f"Validation error: {error}\n"
                "Retry now. Return only a JSON object matching the schema, with no markdown fences "
                "and no extra prose."
            ),
            extra={
                "interrupt_type": "OutputSchemaValidationError",
                "validation_error": error,
                "retry": state.output_validation_failures,
            },
        ),
        error="OutputSchemaValidationError",
    )


def action_signature(action: dict) -> str:
    return json.dumps(normalize_action(action), sort_keys=True, default=str)
