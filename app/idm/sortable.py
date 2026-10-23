"""Sortable items that do not show the order the learner is asked for (run 2f8781d6, A3).

The learner UI shuffles the items of an la_sortable slot; the run's items still read "Tầng 1: …", "Tầng 2: …", so the
prefixes gave the answer away. A leading ordinal label ("Tầng N:", "Bước N:", "Giai đoạn N -", "N.", "(N)", "Step N:")
is removed when what is left still makes a valid item: at least the shared validator's 14 characters of a step
(``SORTABLE_STEP_INCOMPLETE``, a rule Node shares; never changed here), distinct items, a capital first letter;
otherwise the items stay as written. Applied to the writer's items and to the source-locked fallback's ("Bước N: …").

Review 5 (M4): only a pure ordinal prefix is removed: every item opens with the same label (the same word, or the same
bare number form) whose number a separator ends (":", ".", ")", "-"), and the numbers run on one by one ("Bước 9" …
"Bước 12" too). A number that is content stays: "Mốc 30 ngày: …", "Tier 2 engineers …", "Phần 1 của buổi gặp: …", or
one actor item "Cấp 2 phê duyệt …".
"""

from __future__ import annotations

import re
from typing import Any, Final

ORDER_LABELS_STRIPPED: Final = "w5_sortable_order_labels_stripped"
_ORDER_LABEL_RE: Final = re.compile(
    r"^\s*(?:(?P<word>tầng|bước|giai\s+đoạn|cấp|bậc|phần|mốc|step|stage|phase|level|layer|tier|part)\s*(?P<named>\d{1,2})"
    r"\s*[:.)\-–—]|\(\s*(?P<bracketed>\d{1,2})\s*\)\s*[:.\-–—]?|(?P<bare>\d{1,2})\s*(?P<mark>[.)])(?!\d))\s*(?=\S)",  # noqa: RUF001
    re.IGNORECASE)
# The shared validator's shortest sortable step (validation.py, SORTABLE_STEP_INCOMPLETE).
_MIN_ITEM_CHARS: Final = 14


def _text(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    value = item.get("text") if isinstance(item, dict) else None
    return value if isinstance(value, str) else None


def _label(match: re.Match[str]) -> tuple[str, int]:
    """(the form of an ordinal label: its word, "(N)" or the mark after a bare number; its number)."""

    if match.group("word"):
        return " ".join(match.group("word").casefold().split()), int(match.group("named"))
    if match.group("bracketed"):
        return "(N)", int(match.group("bracketed"))
    return f"N{match.group('mark')}", int(match.group("bare"))


def without_order_labels(component: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """``component`` (an la_sortable payload) with the leading ordinal label of its items removed, and how many items
    changed; ``(component, 0)`` when no item has one or a stripped item would not be a valid item."""

    items = component.get("items")
    if not isinstance(items, list) or not items:
        return component, 0
    texts = [text for item in items if (text := _text(item)) is not None]
    if len(texts) != len(items):
        return component, 0
    matches = [_ORDER_LABEL_RE.match(text) for text in texts]
    labels = [_label(match) for match in matches if match is not None]
    numbers = sorted(number for _form, number in labels)
    # Review 5 (M4): one label form on every item, its numbers one by one.
    if (len(labels) != len(texts) or len({form for form, _number in labels}) != 1
            or numbers != list(range(numbers[0], numbers[0] + len(numbers)))):
        return component, 0
    stripped = [" ".join(text[match.end():].split()) for text, match in zip(texts, matches, strict=True) if match]
    stripped = [text[:1].upper() + text[1:] for text in stripped]
    count = len(stripped)
    if (any(len(" ".join(text.split())) < _MIN_ITEM_CHARS for text in stripped)
            or len({text.casefold() for text in stripped}) != len(stripped)):
        return component, 0
    renamed = [{**item, "text": text} if isinstance(item, dict) else text for item, text in zip(items, stripped,
                                                                                                strict=True)]
    return {**component, "items": renamed}, count


def strip_order_labels(component: dict[str, Any] | None) -> dict[str, Any] | None:
    """``without_order_labels`` of an la_sortable slot of a source-locked inventory (any other slot as it is)."""

    if component is None or component.get("type") != "la_sortable":
        return component
    return without_order_labels(component)[0]


__all__ = ["ORDER_LABELS_STRIPPED", "strip_order_labels", "without_order_labels"]
