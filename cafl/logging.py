"""Event logging helpers for CAFL runs."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from .backend import MiniEvent


DEFAULT_MAX_EVENT_CHARS = 500

class EventLogger:
    def __init__(self, path: Path | str, *, max_chars: int = DEFAULT_MAX_EVENT_CHARS):
        self.path = Path(path)
        self.max_chars = max_chars
        self._lock = Lock()

    def __call__(self, event: MiniEvent) -> None:
        label = event.item_id or event.run_id
        content = " ".join((event.content or "").split()) or "<empty>"
        if len(content) > self.max_chars:
            content = f"{content[:self.max_chars]} ... <truncated {len(content) - self.max_chars} chars>"
        line = f"[{label} #{event.index} {event.role}/{event.status}] {content}"
        if event.error:
            line += f" | error={event.error}"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
