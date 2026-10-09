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
# The "Row N:" label the PDF reader puts before every table row is never part of an item (QC run ab8d67e1, R1: with
# the label optional the pattern backtracked and took "Row" itself as the stem, so 7 table rows became "4 lực đẩy").
ROW_LABEL_RE: Final = re.compile(r"^\s*row\s+\d{1,3}\s*:\s*", re.IGNORECASE)
# A numbered item at the start of a fact (its row label removed): "Shift 1: …", "Bước 3 - …", "Cấp 2 | …".
_ITEM_RE: Final = re.compile(
    r"^\s*[-•*]?\s*(?P<stem>[^\W\d_]+(?:[ \t][^\W\d_]+){0,2}?)[ \t]+(?P<number>\d{1,2})(?![\w.,])"
    r"\s*[:.)|\-–—]?\s*(?P<rest>.*)$", re.IGNORECASE)  # noqa: RUF001 - dashes
# Stems that number something other than the items of a framework (folded): table rows, pages, chapters.
_NON_ITEM_STEMS: Final = frozenset({"row", "rows", "page", "trang", "chuong", "chapter", "phan", "part", "muc",
                                    "section", "slide", "hinh", "figure", "bang", "table"})
# Stems that name the items of a framework noun (folded): "SHIFT 1" names one of "5 chuyển dịch", "Bậc 01" one of
# "6 nấc thang". A noun outside every group is named only by itself. A stem must name the promised noun: the first
# numbered stem of the chapter is never a fallback (R1: "Row" was taken for "4 lực đẩy").
_NOUN_STEMS: Final[tuple[frozenset[str], ...]] = (
    frozenset({"chuyen dich", "dich chuyen", "shift", "shifts"}),
    frozenset({"tru cot", "pillar", "pillars"}),
    frozenset({"giai doan", "chang", "dot", "phase", "phases", "stage", "stages"}),
    frozenset({"buoc", "step", "steps"}),
    frozenset({"nguyen tac", "quy tac", "principle", "principles", "rule", "rules"}),
    frozenset({"nang luc", "capability", "capabilities", "competency", "competencies"}),
    frozenset({"yeu to", "thanh phan", "manh ghep", "element", "elements", "component", "components", "factor",
               "factors"}),
    frozenset({"cap do", "cap", "muc do", "nac", "nac thang", "bac", "tang", "lop", "level", "levels", "layer",
               "layers"}),
    frozenset({"luc day", "driver", "drivers", "force", "forces"}),
    frozenset({"truc", "khia canh", "dimension", "dimensions", "axis"}),
    frozenset({"nhom", "group", "groups"}),
    frozenset({"loai", "type", "types"}),
    frozenset({"ky nang", "skill", "skills"}),
    frozenset({"thoi quen", "habit", "habits"}),
    frozenset({"hanh dong", "action", "actions"}),
    frozenset({"chien luoc", "strategy", "strategies"}),
)
_NAME_END_RE: Final = re.compile(r"[,.;:|()\[\]]")
# The display label of an item stops at a dash clause ("THINK BIG — Từ Tư Duy …" -> "THINK BIG").
_DISPLAY_END_RE: Final = re.compile(r"\s+[—–-]\s+")  # noqa: RUF001 - dashes
_MAX_NAME_WORDS: Final = 6
# Words of an item name in its display label (a list item or a support brief), after the folded matching name.
_MAX_DISPLAY_WORDS: Final = 20
_MIN_NAME_CHARS: Final = 3
_LIST_FIELDS: Final = {"bullets": "items", "steps": "items", "table": "rows"}


@dataclass(frozen=True)
class FrameworkPromise:
    """``count`` items of a framework named ``noun`` (as written, lower case), with the numbered items the
    facts give for it (folded labels such as "shift 1" and names), possibly none, and their display form
    ("SHIFT 1: THINK BIG")."""

    count: int
    noun: str
    phrase: str
    items: tuple[tuple[str, str], ...] = ()
    labels: tuple[str, ...] = ()


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


def without_row_label(text: str) -> str:
    """``text`` without the "Row N:" label the PDF reader puts before a table row."""

    return ROW_LABEL_RE.sub("", text, count=1)


def stem_names_noun(stem: str, noun: str) -> bool:
    """The folded ``stem`` ("shift", "bac", "giai doan") names the items of the framework ``noun``."""

    folded = idm_fold(noun)
    aliases = next((group for group in _NOUN_STEMS if folded in group), frozenset({folded}))
    return stem not in _NON_ITEM_STEMS and any(stem == alias or stem.endswith(" " + alias) for alias in aliases)


def _stems(texts: Iterable[str]) -> dict[str, dict[int, tuple[str, str]]]:
    """stem -> number -> (folded name, display label) of the numbered items of one block."""

    stems: dict[str, dict[int, tuple[str, str]]] = {}
    for text in texts:
        match = _ITEM_RE.match(without_row_label(unicodedata.normalize("NFC", text)))
        if match is None:
            continue
        stem = idm_fold(match.group("stem"))
        name = _DISPLAY_END_RE.split(_NAME_END_RE.split(match.group("rest"), maxsplit=1)[0], maxsplit=1)[0]
        words = name.split()
        number = int(match.group("number"))
        display = (f"{' '.join(match.group('stem').split())} {number}"
                   + (f": {' '.join(words[:_MAX_DISPLAY_WORDS])}" if words else ""))
        # The folded name stops at the dash clause too: "Think Big" names "SHIFT 1: THINK BIG — Từ Tư Duy …".
        stems.setdefault(stem, {}).setdefault(number, (idm_fold(" ".join(words[:_MAX_NAME_WORDS])), display))
    return stems


def _numbered(fact_groups: Iterable[Sequence[str]], count: int, noun: str) -> list[tuple[str, str, str]]:
    """(folded label, folded name, display label) of the promised items, by number; empty when unsure.

    ``fact_groups`` are the facts of one source block (or one support item) each. The items are numbered exactly
    1..N under a stem that names ``noun``, either inside ONE group (a table or a list) or one item per group (a
    framework taught block by block: "SHIFT 1" heads one block, "SHIFT 2" the next). Never items gathered from
    several tables (QC run ab8d67e1, R1: rows 1-6 of the 5-axes table and row 8 of the Ladder were taken for
    "4 lực đẩy"), never more or fewer than promised, never a stem that names something else.
    """

    wanted = set(range(1, count + 1))
    # stem -> number -> item, kept only while every group contributes a single item of that stem.
    headings: dict[str, dict[int, tuple[str, str]]] = {}
    split: set[str] = set()
    for texts in fact_groups:
        for stem, numbers in _stems(texts).items():
            if not stem_names_noun(stem, noun):
                continue
            if set(numbers) == wanted:
                return [(f"{stem} {number}", name, display) for number, (name, display) in sorted(numbers.items())]
            if len(numbers) > 1 or set(numbers) & set(headings.get(stem, {})):
                split.add(stem)
            headings.setdefault(stem, {}).update(numbers)
    for stem, numbers in headings.items():
        if stem not in split and set(numbers) == wanted:
            return [(f"{stem} {number}", name, display) for number, (name, display) in sorted(numbers.items())]
    return []


def numbered_items(fact_groups: Iterable[Sequence[str]], count: int, noun: str) -> tuple[tuple[str, str], ...]:
    """The numbered items the facts give for a framework of ``count`` items: (folded label, folded name)."""

    return tuple((label, name) for label, name, _display in _numbered(fact_groups, count, noun))


def build_promise(titles: Sequence[str], fact_groups: Sequence[Sequence[str]]) -> FrameworkPromise | None:
    """The "N <items>" the titles promise, with the items one group of facts numbers for it (or none)."""

    found = framework_promise(titles)
    if found is None:
        return None
    count, noun, phrase = found
    items = _numbered(fact_groups, count, noun)
    return FrameworkPromise(count, noun, phrase, tuple((label, name) for label, name, _ in items),
                            tuple(display for _label, _name, display in items))


# --- the items of a framework a unit cannot see (QC run 8de1c76b, Q3) ---------------------------------------
# The orientation unit "Bản đồ 5 chuyển dịch" owned only the introduction block; the names of the five shifts were
# in the blocks of the next lessons, so the writer could not list them. W4 hands them to the unit's teaching html as
# a server-written support item ("Liệt kê đủ 5 chuyển dịch: SHIFT 1: THINK BIG; …"), which reaches the writer
# through the unit brief and lets W5 check and, as a last resort, insert the list.
FRAMEWORK_SUPPORT_LEAD: Final[Mapping[str, str]] = {"vi": "Liệt kê đủ", "en": "List all"}
_SUPPORT_ITEM_SEPARATOR: Final = "; "
_SUPPORT_BRIEF_MAX_CHARS: Final = 300
_SHORT_NAME_WORDS: Final = 3


def framework_support_brief(phrase: str, labels: Sequence[str], locale: str) -> str | None:
    """The support-item brief that names every item, within the 300-character bound (names shortened, then
    labels only); ``None`` when even the bare labels do not fit."""

    lead = FRAMEWORK_SUPPORT_LEAD["vi" if locale == "vi" else "en"]
    shortened = [label.split(": ", 1)[0] + (": " + " ".join(label.split(": ", 1)[1].split()[:_SHORT_NAME_WORDS])
                                            if ": " in label else "") for label in labels]
    bare = [label.split(": ", 1)[0] for label in labels]
    for variant in (labels, shortened, bare):
        brief = f"{lead} {phrase}: " + _SUPPORT_ITEM_SEPARATOR.join(variant)
        if len(brief) <= _SUPPORT_BRIEF_MAX_CHARS:
            return brief
    return None


def framework_support_lines(briefs: Iterable[str]) -> list[str]:
    """The item lines of server-written framework support briefs (other briefs are ignored)."""

    lines: list[str] = []
    for brief in briefs:
        lead = next((value for value in FRAMEWORK_SUPPORT_LEAD.values() if brief.startswith(value + " ")), None)
        if lead is None:
            continue
        _phrase, separator, items = brief[len(lead) + 1:].partition(": ")
        if separator:
            lines.extend(item.strip() for item in items.split(_SUPPORT_ITEM_SEPARATOR) if item.strip())
    return lines


def unit_framework_brief(titles: Sequence[str], unit_fact_groups: Sequence[Sequence[str]],
                         shard_fact_groups: Sequence[Sequence[str]], locale: str) -> str | None:
    """The framework support brief a unit needs: its titles promise "N <items>", its own facts do not number
    them, and exactly N items under a stem that names the promised noun are numbered in one block of its shard.
    ``None`` whenever that is not certain: the writer then lists the items from the unit's own facts."""

    found = framework_promise(titles)
    if found is None:
        return None
    count, noun, phrase = found
    if _numbered(unit_fact_groups, count, noun):
        return None
    items = _numbered(shard_fact_groups, count, noun)
    if len(items) != count:
        return None
    return framework_support_brief(phrase, [display for _label, _name, display in items][:_MAX_ITEMS], locale)


def has_list_of(component: dict[str, Any], count: int) -> bool:
    """The html slot already has a list, step list or table of exactly ``count`` entries."""

    semantic = component.get("semantic_content")
    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    for section in sections if isinstance(sections, list) else []:
        for block in section.get("blocks") or [] if isinstance(section, dict) else []:
            field = _LIST_FIELDS.get(str(block.get("kind"))) if isinstance(block, dict) else None
            entries = block.get(field) if field else None
            if isinstance(entries, list) and len(entries) == count:
                return True
    return False


def with_framework_list(component: dict[str, Any], promise: FrameworkPromise, locale: str) -> dict[str, Any] | None:
    """``component`` (an html slot written as semantic content) with one more section listing every promised
    item by its display label, right after its first section; ``None`` when there is nothing to insert.

    Only exactly the promised items are inserted, and never into an html that already lists that many entries
    (QC run ab8d67e1, R1: the writer had listed the 4 forces; 7 "Row N" lines were inserted under them): when
    unsure, the slot is left for the author with a review note.
    """

    semantic = component.get("semantic_content")
    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    if (not isinstance(semantic, dict) or not isinstance(sections, list) or not sections
            or len(promise.labels) != promise.count or has_list_of(component, promise.count)):
        return None
    heading = (f"{promise.phrase[:1].upper()}{promise.phrase[1:]} gồm những gì?" if locale == "vi"
               else f"What are the {promise.phrase}?")
    section = {"heading": heading, "learning_block_ids": [],
               "blocks": [{"kind": "bullets", "text": None, "items": list(promise.labels), "rows": []}]}
    return {**component, "semantic_content": {**semantic, "sections": [sections[0], section, *sections[1:]]}}


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
    "FRAMEWORK_SUPPORT_LEAD", "ROW_LABEL_RE", "FrameworkPromise", "build_promise", "framework_coverage",
    "framework_listed", "framework_promise", "framework_support_brief", "framework_support_lines", "has_list_of",
    "numbered_items", "stem_names_noun", "unit_framework_brief", "with_framework_list", "without_row_label",
]
