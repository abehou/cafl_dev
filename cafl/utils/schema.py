"""Output schema parsing and validation helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import ValidationError, validate


JSON_SCHEMA_KEYS = {
    "$schema",
    "$defs",
    "additionalProperties",
    "allOf",
    "anyOf",
    "const",
    "enum",
    "items",
    "oneOf",
    "properties",
    "required",
    "type",
}


def extract_json_object(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def schema_spec_to_json_schema(schema: Any) -> dict:
    if _is_json_schema(schema):
        return schema
    return _infer_json_schema(schema)


def format_output_schema_instruction(schema: Any) -> str:
    return (
        "## Output Schema\n"
        "Return the final answer as a JSON object matching this schema. "
        "Do not include markdown fences or extra prose outside the JSON object.\n"
        f"{json.dumps(schema, indent=2, ensure_ascii=False)}"
    )


def validate_output_text(text: str, schema: Any) -> tuple[dict | None, str | None]:
    parsed = extract_json_object(text)
    if parsed is None:
        return None, "Final answer is not a JSON object."
    try:
        validate(instance=parsed, schema=schema_spec_to_json_schema(schema))
    except ValidationError as e:
        return parsed, e.message
    return parsed, None


def _is_json_schema(schema: Any) -> bool:
    return isinstance(schema, dict) and bool(JSON_SCHEMA_KEYS & set(schema))


def _infer_json_schema(schema: Any) -> dict:
    if isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {key: _infer_json_schema(value) for key, value in schema.items()},
            "required": list(schema.keys()),
            "additionalProperties": True,
        }
    if isinstance(schema, list):
        item_schema = _infer_json_schema(schema[0]) if schema else {}
        return {"type": "array", "items": item_schema}
    if isinstance(schema, str):
        return {"type": _infer_type_from_description(schema)}
    if isinstance(schema, bool):
        return {"type": "boolean"}
    if isinstance(schema, int):
        return {"type": "integer"}
    if isinstance(schema, float):
        return {"type": "number"}
    return {}


def _infer_type_from_description(description: str) -> str:
    text = description.lower()
    if "string" in text:
        return "string"
    if "integer" in text or "int" in text:
        return "integer"
    if "number" in text or "float" in text:
        return "number"
    if "boolean" in text or "bool" in text:
        return "boolean"
    if "array" in text or "list" in text:
        return "array"
    if "object" in text or "dict" in text:
        return "object"
    return "string"
