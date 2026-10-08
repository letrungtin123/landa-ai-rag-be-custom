"""Bound-tolerant reading of provider answers (QC course 234653, defect D3).

The wire schema sent to the provider carries no length/count/number bounds (they are
not part of the documented provider schema subset), while the server models validate
strictly. One summary of 301 characters used to reject a whole W1-reduce/W2/W4 answer.
Before strict validation this module walks the answer along the server model's JSON
schema and, for author-facing text only:

* cuts a free-text string that exceeds its ``maxLength`` by at most a small margin
  (``single_line`` to the bound) — a far longer string still fails and is repaired;
* drops list items beyond ``maxItems`` for annotation lists (SME questions, issues,
  gaps, prerequisites, merges/conflicts, support items, judge findings) whose shorter
  form is still a correct answer;
* clamps estimate numbers (screens, minutes) into their range.

Identifiers, enums, patterns, structural lists and every other field stay strict: they
decide ownership and accounting, so a wrong value must fail and be repaired.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Final

from pydantic import BaseModel

from app.idm.text import single_line

TRIMMED_STRING_CODE: Final = "IDM_RESPONSE_FIELD_TRIMMED"
TRIMMED_LIST_CODE: Final = "IDM_RESPONSE_LIST_TRIMMED"
CLAMPED_NUMBER_CODE: Final = "IDM_RESPONSE_NUMBER_CLAMPED"

# A string is cut only when it exceeds the bound by at most this share of the bound (and
# always by up to ``_MIN_MARGIN_CHARS``); a longer string means the field was misused.
_MARGIN_SHARE: Final = 0.5
_MIN_MARGIN_CHARS: Final = 50
_SAFE_LIST_FIELDS: Final = frozenset({
    "sme_questions", "issues", "gaps", "prerequisites", "merges", "conflicts", "support_items", "findings",
})
# Lists of plain strings under these names are statements the lesson serves, not references.
_SAFE_STRING_LIST_FIELDS: Final = frozenset({"learning_objectives"})
_CLAMP_FIELDS: Final = frozenset({"est_screens", "est_minutes"})
_ID_SUFFIXES: Final = ("_id", "_ids", "_key", "_keys")
_SCHEMA_CACHE: dict[type[BaseModel], dict[str, Any]] = {}


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    cached = _SCHEMA_CACHE.get(model)
    if cached is None:
        cached = model.model_json_schema()
        if model.__module__ == "app.idm.contracts":
            _SCHEMA_CACHE[model] = cached
    return cached


class _Walker:
    def __init__(self, definitions: dict[str, Any]) -> None:
        self.definitions = definitions
        self.codes: Counter[str] = Counter()

    def resolve(self, node: dict[str, Any], value: Any) -> dict[str, Any]:
        """Follow ``$ref``/``allOf`` and pick the ``anyOf`` branch that matches ``value``'s JSON type."""

        for _ in range(8):
            if "$ref" in node:
                node = self.definitions.get(str(node["$ref"]).rsplit("/", 1)[-1], {})
            elif isinstance(node.get("allOf"), list) and len(node["allOf"]) == 1:
                node = node["allOf"][0]
            elif isinstance(node.get("anyOf"), list):
                node = next((branch for branch in node["anyOf"] if _matches(branch, value, self.definitions)), {})
            else:
                return node
        return node

    def walk(self, node: dict[str, Any], value: Any, name: str) -> Any:
        node = self.resolve(node, value)
        if isinstance(value, dict) and isinstance(node.get("properties"), dict):
            properties: dict[str, Any] = node["properties"]
            return {key: self.walk(properties[key], item, key) if key in properties else item
                    for key, item in value.items()}
        if isinstance(value, list) and isinstance(node.get("items"), dict):
            items = [self.walk(node["items"], item, name) for item in value]
            limit = node.get("maxItems")
            string_items = node["items"].get("type") == "string"
            if (isinstance(limit, int) and len(items) > limit
                    and (name in _SAFE_LIST_FIELDS or (string_items and name in _SAFE_STRING_LIST_FIELDS))):
                self.codes[TRIMMED_LIST_CODE] += 1
                items = items[:limit]
            return items
        if isinstance(value, str):
            return self.string(node, value, name)
        if type(value) is int and name in _CLAMP_FIELDS:
            low, high = node.get("minimum"), node.get("maximum")
            clamped = max(low, value) if isinstance(low, int) else value
            clamped = min(high, clamped) if isinstance(high, int) else clamped
            if clamped != value:
                self.codes[CLAMPED_NUMBER_CODE] += 1
            return clamped
        return value

    def string(self, node: dict[str, Any], value: str, name: str) -> str:
        limit = node.get("maxLength")
        if (not isinstance(limit, int) or "pattern" in node or "enum" in node or "const" in node
                or name.endswith(_ID_SUFFIXES) or name == "local_id"):
            return value
        if len(value.strip()) <= limit:
            return value
        text = " ".join(value.split())
        if len(text) > limit + max(_MIN_MARGIN_CHARS, int(limit * _MARGIN_SHARE)):
            return value  # far too long: strict validation reports it and the repair fixes it
        self.codes[TRIMMED_STRING_CODE] += 1
        return text if len(text) <= limit else single_line(text, limit)


def _matches(branch: dict[str, Any], value: Any, definitions: dict[str, Any]) -> bool:
    if "$ref" in branch:
        branch = definitions.get(str(branch["$ref"]).rsplit("/", 1)[-1], {})
    expected = branch.get("type")
    if expected is None:
        return True
    checks = {"null": value is None, "string": isinstance(value, str), "array": isinstance(value, list),
              "object": isinstance(value, dict), "integer": type(value) is int, "boolean": isinstance(value, bool),
              "number": type(value) in {int, float}}
    return bool(checks.get(str(expected), False))


def coerce_provider_answer(model: type[BaseModel], value: Any) -> tuple[Any, Counter[str]]:
    """Return ``value`` adjusted to the soft bounds of ``model`` and the adjustment counts."""

    schema = _schema(model)
    walker = _Walker(schema.get("$defs", {}))
    return walker.walk(schema, value, ""), walker.codes
