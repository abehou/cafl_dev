"""Event logging helpers for CAFL runs."""

from __future__ import annotations

import sys
from threading import Lock
from typing import TextIO

from .backend import MiniEvent


DEFAULT_MAX_EVENT_CHARS = 2000


class ConsoleEventLogger:
    def __init__(self, *, stream: TextIO | None = None, max_chars: int = DEFAULT_MAX_EVENT_CHARS):
        self.stream = stream if stream is not None else sys.stdout
        self.max_chars = max_chars
        self._lock = Lock()

    def __call__(self, event: MiniEvent) -> None:
        label = event.item_id or event.run_id
        content = event.content.strip()
        if not content:
            content = "<empty>"
        if len(content) > self.max_chars:
            omitted = len(content) - self.max_chars
            content = f"{content[:self.max_chars]}\n... <truncated {omitted} chars>"
        with self._lock:
            print(f"[{label} #{event.index} {event.role}/{event.status}] {content}", file=self.stream, flush=True)
