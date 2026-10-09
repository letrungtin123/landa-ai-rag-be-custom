"""The text and list bounds of a strict server model, stated in the prompt (QC run 8de1c76b, Q1).

The provider schema carries no length or count bounds (``runtime.idm_provider_response_model``), so a
model that never saw them wrote 33 criteria facts (bound 24) and 8 media points (bound 6) and the whole
W4 answer was rejected. The bounds are read from the same JSON schema the server validates with, so the
prompt can never drift from the contract.
"""

from __future__ import annotations

import functools
from typing import Any, Final

from pydantic import BaseModel

_MAX_DEPTH: Final = 8
# Lists of identifiers (fact keys, block ids): their item form is stated where they are introduced.
_ID_LIST_SUFFIXES: Final = ("_keys", "_ids")


def _resolve(node: dict[str, Any], definitions: dict[str, Any]) -> dict[str, Any]:
    for _ in range(_MAX_DEPTH):
        if "$ref" in node:
            node = definitions.get(str(node["$ref"]).rsplit("/", 1)[-1], {})
        elif isinstance(node.get("allOf"), list) and len(node["allOf"]) == 1:
            node = node["allOf"][0]
        elif isinstance(node.get("anyOf"), list):
            node = next((branch for branch in node["anyOf"] if branch.get("type") != "null"), {})
        else:
            return node
    return node


def _span(low: Any, high: Any, unit: str) -> str:
    return f"{low}-{high} {unit}" if isinstance(low, int) and low > 0 else f"at most {high} {unit}"


def _text_bound(node: dict[str, Any]) -> str | None:
    if node.get("type") != "string" or not isinstance(node.get("maxLength"), int):
        return None
    if any(key in node for key in ("pattern", "enum", "const")):
        return None  # identifiers and enums: their form is stated where they are introduced
    return _span(node.get("minLength"), node["maxLength"], "chars")


def _walk(node: dict[str, Any], path: str, definitions: dict[str, Any], lines: list[str], depth: int) -> None:
    properties = node.get("properties")
    if not isinstance(properties, dict) or depth > _MAX_DEPTH:
        return
    parts: list[str] = []
    nested: list[tuple[dict[str, Any], str]] = []
    for name, raw in properties.items():
        field = _resolve(raw, definitions)
        text = _text_bound(field)
        if text is not None:
            parts.append(f"{name} {text}")
        elif field.get("type") == "array":
            items = _resolve(field.get("items") or {}, definitions)
            if isinstance(field.get("maxItems"), int):
                each = None if name.endswith(_ID_LIST_SUFFIXES) else _text_bound(items)
                parts.append(f"{name} {_span(field.get('minItems'), field['maxItems'], 'items')}"
                             + (f" (each {each})" if each else ""))
            if items.get("type") == "object":
                nested.append((items, f"{path}.{name}[]" if path else f"{name}[]"))
        elif field.get("type") == "object":
            nested.append((field, f"{path}.{name}" if path else name))
    if parts:
        lines.append(f"- {path or 'answer'}: " + "; ".join(parts))
    for child, child_path in nested:
        _walk(child, child_path, definitions, lines, depth + 1)


@functools.cache
def answer_limits(model: type[BaseModel]) -> str:
    """Every text and list bound of ``model``, one line per object path (identifiers left out)."""

    schema = model.model_json_schema()
    lines: list[str] = []
    _walk(schema, "", schema.get("$defs", {}), lines, 0)
    return "\n".join(lines)


__all__ = ["answer_limits"]
