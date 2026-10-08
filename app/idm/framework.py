"""Enumerated frameworks a title or an objective promises (QC course 364564, N5).

Chapter 2 of the QC course served the objective "… 5 chuyển dịch …" but never showed the five
shifts together; its first unit, titled "Tổng quan 5 chuyển dịch", taught something else. A title
or objective that names "N <items>" ("5 chuyển dịch", "6 trụ cột", "ba giai đoạn", "8 Quick-Win")
promises an overview of all N items. These helpers find such a promise, the numbered items the
facts give for it ("Shift 1: …", "Bước 3: …", "Cấp 2 | …") and whether an html slot lists them.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.idm.text import idm_fold

_MIN_ITEMS: Final = 2
_MAX_ITEMS: Final = 12
# Number words with their diacritics: folded, "sáu" (six) would read as "sau" (after).
_NUMBER_WORDS: Final[Mapping[str, int]] = {
    "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
# Nouns that name the parts of a framework (lower case, NFC, with diacritics).
_FRAMEWORK_NOUNS: Final = (
    "chuyển dịch", "trụ cột", "giai đoạn", "bước", "nguyên tắc", "năng lực", "yếu tố", "thành phần", "cấp độ",
    "cấp", "nấc", "tầng", "lực đẩy", "trục", "khía cạnh", "phẩm chất", "chặng", "nhóm", "loại", "kỹ năng",
    "thói quen", "hành động", "quick-win", "quick win", "trọng tâm", "chiến lược", "mảnh ghép", "lớp",
    "shifts", "shift", "pillars", "pillar", "stages", "stage", "phases", "phase", "steps", "step", "principles",
    "principle", "capabilities", "capability", "competencies", "competency", "elements", "element",
    "components", "component", "levels", "level", "dimensions", "dimension", "factors", "factor", "drivers",
    "driver", "habits", "habit", "rules", "rule", "layers", "layer", "quick wins", "quick win", "keys", "types",
    "groups", "areas",
)
_PROMISE_RE: Final = re.compile(
    r"(?<!\w)(\d{1,2}|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\s+("
    + "|".join(re.escape(noun) for noun in sorted(_FRAMEWORK_NOUNS, key=len, reverse=True)) + r")(?![\w-])")
# A numbered item at the start of a fact: "Shift 1: …", "Bước 3 - …", "Cấp 2 | …" (after a table "Row N:").
_ITEM_RE: Final = re.compile(
    r"^\s*(?:row\s+\d+\s*:\s*)?[-•*]?\s*(?P<stem>[^\W\d_]+(?:[ \t][^\W\d_]+){0,2}?)[ \t]+(?P<number>\d{1,2})(?![\w.,])"
    r"\s*[:.)|\-–—]?\s*(?P<rest>.*)$", re.IGNORECASE)  # noqa: RUF001 - dashes
_NAME_END_RE: Final = re.compile(r"[,.;:|()\[\]]")
_MAX_NAME_WORDS: Final = 6
_MIN_NAME_CHARS: Final = 3
_LIST_FIELDS: Final = {"bullets": "items", "steps": "items", "table": "rows"}


@dataclass(frozen=True)
class FrameworkPromise:
    """``count`` items of a framework named ``noun`` (as written, lower case), with the numbered items the
    facts give for it (folded labels such as "shift 1" and names), possibly none."""

    count: int
    noun: str
    phrase: str
    items: tuple[tuple[str, str], ...] = ()


def framework_promise(texts: Iterable[str]) -> tuple[int, str, str] | None:
    """The first "N <items>" a text names: (N, noun, the phrase as written)."""

    for text in texts:
        normalized = unicodedata.normalize("NFC", text)
        for match in _PROMISE_RE.finditer(normalized.lower()):
            number = match.group(1)
            count = int(number) if number.isdigit() else _NUMBER_WORDS[number]
            if _MIN_ITEMS <= count <= _MAX_ITEMS:
                return count, match.group(2), normalized[match.start():match.end()]
    return None


def numbered_items(fact_texts: Iterable[str], count: int, noun: str) -> tuple[tuple[str, str], ...]:
    """The numbered items the facts give for a framework of ``count`` items: (folded label, folded name).

    The stem (the word before the number) with at least ``count`` distinct numbers wins; a stem that
    shares a word with ``noun`` first, then the first one in fact order.
    """

    stems: dict[str, dict[int, str]] = {}
    for text in fact_texts:
        match = _ITEM_RE.match(unicodedata.normalize("NFC", text))
        if match is None:
            continue
        stem = idm_fold(match.group("stem"))
        name = _NAME_END_RE.split(match.group("rest"), maxsplit=1)[0]
        stems.setdefault(stem, {}).setdefault(int(match.group("number")),
                                              " ".join(idm_fold(name).split()[:_MAX_NAME_WORDS]))
    noun_words = set(idm_fold(noun).split())
    candidates = [stem for stem, numbers in stems.items() if len(numbers) >= count]
    if not candidates:
        return ()
    chosen = next((stem for stem in candidates if noun_words & set(stem.split())), candidates[0])
    return tuple((f"{chosen} {number}", name) for number, name in sorted(stems[chosen].items()))


def build_promise(titles: Sequence[str], fact_texts: Sequence[str]) -> FrameworkPromise | None:
    found = framework_promise(titles)
    if found is None:
        return None
    count, noun, phrase = found
    return FrameworkPromise(count, noun, phrase, numbered_items(fact_texts, count, noun))


def _contains(folded: str, phrase: str) -> bool:
    return bool(phrase) and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", folded) is not None


def _longest_list(components: Sequence[dict[str, Any]]) -> int:
    longest = 0
    for component in components:
        semantic = component.get("semantic_content")
        sections = semantic.get("sections") if isinstance(semantic, dict) else None
        for section in sections if isinstance(sections, list) else []:
            for block in section.get("blocks") or [] if isinstance(section, dict) else []:
                field = _LIST_FIELDS.get(str(block.get("kind"))) if isinstance(block, dict) else None
                entries = block.get(field) if field else None
                if isinstance(entries, list):
                    longest = max(longest, len(entries))
    return longest


def framework_coverage(promise: FrameworkPromise, html_components: Sequence[dict[str, Any]],
                       visible_text: str) -> int:
    """How many promised items the html shows: the numbered items it names (by label or name) when the
    facts number them, else the entries of its longest list or table."""

    if promise.items:
        folded = idm_fold(visible_text)
        return sum(_contains(folded, label) or (len(name) >= _MIN_NAME_CHARS and _contains(folded, name))
                   for label, name in promise.items)
    return _longest_list(html_components)


def framework_listed(promise: FrameworkPromise, html_components: Sequence[dict[str, Any]], visible_text: str) -> bool:
    """The html lists every item the title promises (see ``framework_coverage``)."""

    return framework_coverage(promise, html_components, visible_text) >= promise.count


__all__ = [
    "FrameworkPromise", "build_promise", "framework_coverage", "framework_listed", "framework_promise",
    "numbered_items",
]
