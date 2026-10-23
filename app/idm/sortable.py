"""Sortable items that do not show the order the learner is asked for (run 2f8781d6, A3).

The learner UI shuffles the items of an la_sortable slot; the run's items still read "Tầng 1: …", "Tầng 2: …", so the
prefixes gave the answer away. A leading ordinal label ("Tầng N:", "Bước N:", "Giai đoạn N -", "N.", "(N)", "Step N:")
is removed from every item that has one, when what is left still makes a valid item: at least the shared validator's
14 characters of a step (``SORTABLE_STEP_INCOMPLETE``, a rule Node shares; never changed here), distinct items, a
capital first letter; otherwise the items stay as written. Applied to the writer's items and to the source-locked
fallback's ("Bước N: …").
"""

from __future__ import annotations

import re
from typing import Any, Final

ORDER_LABELS_STRIPPED: Final = "w5_sortable_order_labels_stripped"
_ORDER_LABEL_RE: Final = re.compile(
    r"^\s*(?:(?:tầng|bước|giai\s+đoạn|cấp|bậc|phần|mốc|step|stage|phase|level|layer|tier|part)\s*\d{1,2}"
    r"|\(\s*\d{1,2}\s*\)|\d{1,2}\s*[.)](?!\d))\s*[:.)\-–—]?\s*(?=\S)",  # noqa: RUF001 - the dashes are separators
    re.IGNORECASE)
# The shared validator's shortest sortable step (validation.py, SORTABLE_STEP_INCOMPLETE).
_MIN_ITEM_CHARS: Final = 14


def _text(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    value = item.get("text") if isinstance(item, dict) else None
    return value if isinstance(value, str) else None


def without_order_labels(component: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """``component`` (an la_sortable payload) with the leading ordinal label of its items removed, and how many items
    changed; ``(component, 0)`` when no item has one or a stripped item would not be a valid item."""

    items = component.get("items")
    if not isinstance(items, list) or not items:
        return component, 0
    texts = [text for item in items if (text := _text(item)) is not None]
    if len(texts) != len(items):
        return component, 0
    stripped: list[str] = []
    count = 0
    for text in texts:
        match = _ORDER_LABEL_RE.match(text)
        rest = " ".join(text[match.end():].split()) if match else text
        if match:
            count += 1
            rest = rest[:1].upper() + rest[1:]
        stripped.append(rest)
    if (not count or any(len(" ".join(text.split())) < _MIN_ITEM_CHARS for text in stripped)
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
