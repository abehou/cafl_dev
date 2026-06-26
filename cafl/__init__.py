"""CAFL agent source code."""

__all__ = ["Cafl", "CaflConfig", "EventLogger"]


def __getattr__(name: str):
    if name == "Cafl":
        from .backend import Cafl

        return Cafl
    if name == "CaflConfig":
        from .config import CaflConfig

        return CaflConfig
    if name == "EventLogger":
        from .logging import EventLogger

        return EventLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
