"""The shape of a model's JSON answer: balanced ``{...}`` spans and content-free diagnostics.

Moved verbatim out of ``app.lesson_author_blueprint`` (refactor wave 3b): the provider calls
(``app.infra.gemini.calls``) describe every structured answer they could not parse with it, and the
legacy blueprint parser recovers its candidate objects with the same scanner. The description holds
lengths, positions and counts only, never answer text.
"""

from __future__ import annotations

from typing import Any


def find_balanced_json_objects(value: str) -> list[str]:
    """The outermost ``{...}`` spans of ``value`` in order; braces inside JSON strings do not count."""

    objects: list[str] = []
    stack: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(value):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append(index)
        elif char == "}":
            if not stack:
                continue
            start = stack.pop()
            if not stack:
                objects.append(value[start : index + 1])
    return objects


def describe_lesson_author_blueprint_response(text: str) -> dict[str, Any]:
    candidate = text.strip().lstrip("\ufeff")
    balanced_objects = find_balanced_json_objects(candidate)
    return {
        "response_chars": len(text),
        "first_non_whitespace": candidate[:1] or None,
        "last_non_whitespace": candidate[-1:] or None,
        "has_code_fence": candidate.startswith(chr(96) * 3),
        "first_object_index": candidate.find("{"),
        "last_object_index": candidate.rfind("}"),
        "balanced_object_count": len(balanced_objects),
        "balanced_object_lengths": [len(item) for item in balanced_objects[-3:]],
    }
