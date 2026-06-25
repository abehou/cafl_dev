from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

DEFAULT_MODEL = "gemini/gemini-3-flash-preview"
MODEL_ALIASES = MappingProxyType(
    {
        "gemini-3-flash": DEFAULT_MODEL,
        "gemini/gemini-3-flash": DEFAULT_MODEL,
    }
)
DEFAULT_SYSTEM_TEMPLATE = """You are a coding agent for law-related specialized tasks.
You can interact with the local shell environment to inspect files, answer questions,
and complete tasks.

## Command Execution Rules

You are operating in an interactive environment:

1. You may issue bash tool calls to inspect local files or run commands.
2. The system executes each command in a subshell.
3. You see the result.
4. You decide whether to inspect more or answer.

Each response SHOULD include reasoning text explaining what you are doing.
Use bash tool calls when you need local evidence. If the answer is already clear,
answer directly without a tool call.

Prefer precise commands. Use `rg` for code search. Do not recursively search
generated or cache folders such as `runs/`, `.git/`, `__pycache__/`, `.pytest_cache/`,
or virtual environments unless the task explicitly asks about them. Read focused
file sections instead of dumping entire large files.

Directory and environment variable changes are not persistent. Every action is
executed in a new subshell. If you need a specific directory or environment,
include it in the same command.

## Workflow

1. Inspect only the files or commands needed to answer the task.
2. Avoid repeated final checks once the answer is clear.
3. When you are ready to answer, either answer directly without a tool call, or
   submit exactly once with a bash tool call whose first output line is exactly
   COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT and whose remaining output is the final
   answer.
4. After providing the final answer, do not call more tools.

Example final submission command:

cat <<'EOF'
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
Your final answer goes here.
EOF

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>
"""
DEFAULT_INSTANCE_TEMPLATE = "{{ task }}"
DEFAULT_OBSERVATION_TEMPLATE = """{% if output.exception_info -%}
<exception>{{ output.exception_info }}</exception>
{% endif -%}
<returncode>{{ output.returncode }}</returncode>
{% if output.output | length <= 12000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>Tool output was truncated. Use narrower commands, file filters, or line ranges.</warning>
<output_head>
{{ output.output[:2000] -}}
</output_head>
<output_tail>
{{ output.output[-2000:] -}}
</output_tail>
{%- endif -%}
"""


@dataclass(frozen=True)
class CaflConfig:
    default_model: str = DEFAULT_MODEL
    model_aliases: Mapping[str, str] = field(default_factory=lambda: MODEL_ALIASES)
    system_template: str = DEFAULT_SYSTEM_TEMPLATE
    instance_template: str = DEFAULT_INSTANCE_TEMPLATE
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    output_schema: dict[str, Any] | None = None
    output_validation_retries: int = 2

    def resolve_model_name(self, model_name: str) -> str:
        return self.model_aliases.get(model_name, model_name)
