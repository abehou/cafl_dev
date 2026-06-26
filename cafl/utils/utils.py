import datetime
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen


def get_time_signature() -> str:
    """Get a time signature string in the format YYYY/MM/DD_HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S")


def get_path_time_signature() -> str:
    """Get a filesystem-friendly timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_tool_output_json(output: Any) -> Any:
    if isinstance(output, dict) and "output" in output:
        output = output["output"]
    if not isinstance(output, str):
        return output
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_slug(text: str, *, max_length: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (slug or "item")[:max_length].strip("-") or "item"


def list_gemini_models() -> list[str]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to list Gemini API models.")

    with urlopen(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}") as response:
        data = json.loads(response.read().decode("utf-8"))
    return [
        model["name"].replace("models/", "gemini/", 1)
        for model in data.get("models", [])
        if "generateContent" in model.get("supportedGenerationMethods", [])
    ]
