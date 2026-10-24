"""JSON parsing of provider (model) answers for the lesson-author writers: one home for both acceptance rules.

The two rules differ on purpose and keep their exact behaviour:

* ``parse_provider_json_object``: the whole answer, or else the widest ``{...}`` span in it, must be one JSON
  object (proposal and repair answers).
* ``parse_first_provider_json_value``: the first complete JSON object or array, ignoring prose before it and
  everything after it (staged skeleton and unit batches).

Both fail with the same 502 ``ServiceError`` the routes have always returned.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.errors import ServiceError


def provider_json_error(label: str, *, shape: str = "") -> ServiceError:
    return ServiceError(status_code=502, detail=f"AI không trả JSON {label}{shape} hợp lệ.")


def object_span(text: str) -> str | None:
    """What ``re.search(r"\\{.*\\}", text, re.DOTALL)`` matches: from the first "{" to the last "}" after it.

    The regex tried every "{" and ran to the end of the text from each one when no "}" came after it (quadratic on an
    answer of "{"); the first "{" either has a "}" after it, and the greedy ``.*`` stops at the last one, or none does.
    """

    first, last = text.find("{"), text.rfind("}")
    return text[first:last + 1] if 0 <= first < last else None


def parse_provider_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        span = object_span(text)
        if span is None:
            # Implicitly chained to the decode error, exactly as before.
            raise provider_json_error(label)  # noqa: B904
        try:
            parsed = json.loads(span)
        except json.JSONDecodeError as exc:
            raise provider_json_error(label) from exc
    if not isinstance(parsed, dict):
        raise provider_json_error(label, shape=" dạng object")
    return parsed


def parse_first_provider_json_value(text: str, label: str) -> Any:
    """Parse the first complete JSON value without accepting trailing prose."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise provider_json_error(label)
