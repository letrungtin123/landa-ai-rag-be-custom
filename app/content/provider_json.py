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
import re
from typing import Any

from app.core.errors import ServiceError

_OBJECT_SPAN = re.compile(r"\{.*\}", flags=re.DOTALL)


def provider_json_error(label: str, *, shape: str = "") -> ServiceError:
    return ServiceError(status_code=502, detail=f"AI không trả JSON {label}{shape} hợp lệ.")


def parse_provider_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _OBJECT_SPAN.search(text)
        if not match:
            # Implicitly chained to the decode error, exactly as before.
            raise provider_json_error(label)  # noqa: B904
        try:
            parsed = json.loads(match.group(0))
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
