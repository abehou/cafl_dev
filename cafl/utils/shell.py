"""Small shell command parsing helpers."""

from __future__ import annotations

import contextlib
import shlex


def split_shell_command(command: str) -> list[str] | None:
    with contextlib.suppress(ValueError):
        return shlex.split(command)
    return None


def shell_arg(parts: list[str], name: str) -> str | None:
    with contextlib.suppress(ValueError):
        index = parts.index(name)
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def shell_args(parts: list[str], name: str) -> list[str]:
    return [parts[index + 1] for index, part in enumerate(parts[:-1]) if part == name]
