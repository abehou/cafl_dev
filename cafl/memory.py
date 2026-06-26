"""Environment and per-run memory helpers for CAFL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .utils.formatting import json_preview, normalize_action

MAX_ACTIONS = 20
MAX_OBSERVATIONS = 12
MAX_OUTPUT_PREVIEW_CHARS = 500
WORKING_MEMORY_OBSERVATIONS = 5
WORKING_MEMORY_ACTIONS = 5

_ToolSummarizer = Callable[[dict, Any], dict | None]
_TOOL_SUMMARIZERS: list[_ToolSummarizer] = []


def register_tool_summarizer(summarizer: _ToolSummarizer) -> _ToolSummarizer:
    _TOOL_SUMMARIZERS.append(summarizer)
    return summarizer


def load_environment_memory(memory_dir: Path | str | None, *, max_chars: int) -> str:
    if memory_dir is None:
        return ""
    memory_path = Path(memory_dir) / "MEMORY.md"
    if not memory_path.exists():
        return ""
    return memory_path.read_text(encoding="utf-8")[:max_chars]


def record_tool_observation(memory: dict, action: dict, output: Any) -> None:
    action_record = normalize_action(action)
    _append_limited(memory.setdefault("actions", []), action_record, MAX_ACTIONS)
    summary = _summarize_tool_observation(action_record, output)
    if summary is not None:
        _append_limited(memory.setdefault("tool_observations", []), summary, MAX_OBSERVATIONS)


def _summarize_tool_observation(action: dict, output: Any) -> dict | None:
    for summarizer in _TOOL_SUMMARIZERS:
        summary = summarizer(action, output)
        if summary is not None:
            return summary
    return _generic_tool_summary(action, output)


def _generic_tool_summary(action: dict, output: Any) -> dict:
    return {
        "tool": "shell" if "command" in action or "cmd" in action else "tool",
        "action": action,
        "summary": json_preview(output, max_chars=MAX_OUTPUT_PREVIEW_CHARS),
    }


def _append_limited(items: list, item: Any, limit: int) -> None:
    items.append(item)
    del items[:-limit]


def format_working_memory(memory: dict) -> str:
    observations = memory.get("tool_observations") or []
    actions = memory.get("actions") or []
    repeated_actions = _repeated_action_summaries(memory)

    if not observations and not actions and not repeated_actions:
        return ""

    lines = [
        "## Working Memory",
        "Use this compact memory of this run before choosing the next action.",
    ]

    if observations:
        lines.append("")
        lines.append("Recent tool outcomes:")
        for observation in observations[-WORKING_MEMORY_OBSERVATIONS:]:
            lines.append(f"- {_format_observation(observation)}")

    if repeated_actions:
        lines.append("")
        lines.append("Repeated action patterns:")
        for repeated_action in repeated_actions[:3]:
            lines.append(f"- {repeated_action}")

    if actions:
        lines.append("")
        lines.append("Recent actions tried:")
        for action in actions[-WORKING_MEMORY_ACTIONS:]:
            lines.append(f"- {_format_action(action)}")

    lines.append("")
    lines.append("If the memory already contains enough evidence, answer instead of searching again.")
    return "\n".join(lines)


def _format_observation(observation: dict) -> str:
    tool = observation.get("tool", "tool")
    summary = observation.get("summary") or observation.get("output_preview") or ""
    detail_parts = []
    for key in ("query", "filters", "doc_id", "top_results", "relevance"):
        if key in observation and observation[key]:
            detail_parts.append(f"{key}={observation[key]}")
    details = f" ({'; '.join(detail_parts)})" if detail_parts else ""
    return f"{tool}{details}: {summary}"


def _format_action(action: dict) -> str:
    if "command" in action:
        return str(action["command"])
    if "cmd" in action:
        return str(action["cmd"])
    return json.dumps(action, sort_keys=True, default=str)


def _repeated_action_summaries(memory: dict) -> list[str]:
    counts = memory.get("action_counts") or {}
    summaries = []
    for signature, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        if count <= 1:
            continue
        summaries.append(f"{count}x {signature[:240]}")
    return summaries
