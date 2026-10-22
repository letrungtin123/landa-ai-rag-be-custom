"""A worksheet gives the input data its task refers to (run 5496155c, QC 2026-10-22).

Three of seven worksheets of the run asked the learner to work on data they never showed: "Bạn hãy đọc danh sách nhân
sự tham gia dự án …" without the list, "Hãy bóc tách nội dung phỏng vấn …" without the interview, "Từ danh sách 5 yêu
cầu phức tạp …" without the requests. The learner cannot do such a worksheet.

``missing_input`` finds a reference to given data in the worksheet's task and text (accent-folded, fixed patterns: a
list read or used, an interview's content, a counted set of requests, cases or people read or sorted, data "below")
and looks for that data in the worksheet: a list or table block (at least as many entries as the count it names), or
for an interview or a case a paragraph of some length, that is not the worksheet's only form (its template), not part
of the worked example and not the final self-check list. The W5 writer repairs a worksheet without its data once
(``IDM_W5_WORKSHEET_INPUT_MISSING``), then the slot is kept for the author's review with a plain note; the server
never writes the data itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Final

from app.idm.policy import (
    IDM_WORKSHEET_REUSE_MIN_PAIRS,
    IDM_WORKSHEET_REUSE_MIN_WORDS,
    IDM_WORKSHEET_REUSE_PAIR_SHARE,
)
from app.idm.text import content_words, idm_fold

WORKSHEET_INPUT_CODE: Final = "IDM_W5_WORKSHEET_INPUT_MISSING"
# Paid verification of 3ed380d (R2): the worked example solves the task's own input entries, so it gives the answer.
EXAMPLE_REUSE_CODE: Final = "IDM_W5_WORKSHEET_EXAMPLE_REUSES_INPUT"
_NUMBERS: Final = {"hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
                   "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_COUNT: Final = r"(\d{1,2}|" + "|".join(_NUMBERS) + r")"
_ITEMS_VI: Final = (r"(?:yeu cau|tinh huong|de xuat|y kien|phan hoi|ho so|nhan su|khach hang|sang kien|khieu nai"
                    r"|truong hop|ung vien|nha cung cap|du an|cau tra loi)")
_ITEMS_EN: Final = (r"(?:requests?|cases?|complaints?|profiles?|items?|candidates?|suppliers?|customers?|ideas?"
                    r"|initiatives?|answers?)")
# (kind, accent-folded pattern); a count, when the pattern has one, is its first group.
_REFERENCES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("items", re.compile(rf"\b(?:doc|tu|cho|voi|xem|phan loai|danh gia|ra soat|sap xep|xep|dua tren|dua vao)\s+"
                         rf"(?:(?:mo ta|thong tin|danh sach|noi dung)\s+)?(?:cac\s+|nhung\s+)?{_COUNT}\s+"
                         rf"{_ITEMS_VI}\b")),
    ("items", re.compile(rf"\bthe\s+(?:following\s+)?{_COUNT}\s+{_ITEMS_EN}\b")),
    ("list", re.compile(r"\b(?:doc|tu|dua tren|dua vao|theo|xem|ra soat|voi|cho)\s+(?:cac\s+|nhung\s+)?danh sach\b")),
    ("list", re.compile(r"\b(?:read|use|from|review|classify|sort)\s+(?:the\s+)?list\b")),
    ("interview", re.compile(r"\b(?:noi dung|bien ban|doan|ket qua|ghi chep|loi|cau tra loi|ban ghi)\s+(?:cua\s+)?"
                             r"(?:buoi\s+|cuoc\s+)?phong van\b")),
    ("interview", re.compile(r"\b(?:the|this)\s+interview\b|\binterview\s+(?:notes|transcript|answers)\b")),
    ("data", re.compile(r"\b(?:tinh huong|du lieu|so lieu|mo ta|ho so)\s+(?:sau|duoi day|ben duoi|da cho|"
                        r"duoc cung cap)\b")),
    ("data", re.compile(rf"\bthe\s+following\s+{_ITEMS_EN}\b|\b(?:case|data)\s+below\b")),
)
# The worked example: a section whose heading names an example or a model entry.
_EXAMPLE_WORDS: Final = ("vi du", "example", "mau hoan thanh", "bai mau", "tham khao", "mau doi chieu", "ho so mau",
                         "minh hoa", "sample", "hoan chinh", "worked")
# The input data section the writer is asked for ("Dữ liệu đầu vào" / "Input data", R1-R2). Its heading wins over the
# example words ("Dữ liệu minh họa" holds the task's five requests); a template the learner fills ("Bảng thu thập dữ
# liệu khách hàng", guiding questions) is no input (review of dfb6db2, items 7 and 12).
_INPUT_HEADING_RE: Final = re.compile(r"\b(?:du lieu|dau vao|input|data)\b")
_TEMPLATE_HEADING_RE: Final = re.compile(r"\b(?:thu thap|can dien|bieu mau|mau dien|template|fill)\b")
# A named entry ("Ý kiến 1: …", "Yêu cầu 2 - …") or a labelled one ("Thế mạnh cốt lõi: …", a label of a few words).
_NAMED_ENTRY_RE: Final = re.compile(
    r"^\s*(?P<name>[^\W\d_]+(?:\s+[^\W\d_]+){0,3}\s+\d{1,2})\s*[:.)\-\N{EN DASH}]\s*(?P<text>.*)$", re.DOTALL)
_LABEL_MAX_WORDS: Final = 10
_LIST_KINDS: Final = frozenset({"bullets", "steps", "table"})
_TEXT_KINDS: Final = frozenset({"paragraph", "task", "warning"})
_MIN_LIST_ENTRIES: Final = 2
# An interview or a case written as text: at least this many words.
_MIN_TEXT_WORDS: Final = 25


@dataclass(frozen=True)
class InputReference:
    """Data a worksheet's task refers to: its kind (list, items, interview, data) and the count it names, if any."""

    kind: str
    count: int | None = None


def _blocks(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    blocks = section.get("blocks")
    return [block for block in blocks if isinstance(block, dict)] if isinstance(blocks, list) else []


def _sections(component: Mapping[str, Any]) -> list[dict[str, Any]]:
    semantic = component.get("semantic_content")
    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    return [section for section in sections if isinstance(section, dict)] if isinstance(sections, list) else []


def _entries(block: Mapping[str, Any]) -> int:
    values = block.get("rows") if block.get("kind") == "table" else block.get("items")
    return len(values) if isinstance(values, list) else 0


def _count(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else _NUMBERS.get(value)


def input_reference(texts: Iterable[str]) -> InputReference | None:
    """The first reference to given data in ``texts`` (accent-folded), or ``None``."""

    for text in texts:
        folded = idm_fold(text)
        for kind, pattern in _REFERENCES:
            if match := pattern.search(folded):
                return InputReference(kind, _count(match.group(1)) if pattern.groups else None)
    return None


def _questions_table(section: Mapping[str, Any]) -> bool:
    """A table whose values are mostly guiding questions: the template the learner fills, never given data."""

    rows = [row for block in _blocks(section) if block.get("kind") == "table"
            for row in block.get("rows") or [] if isinstance(row, dict)]
    return bool(rows) and 2 * sum(str(row.get("value") or "").strip().endswith("?") for row in rows) >= len(rows)


def is_input_section(section: Mapping[str, Any]) -> bool:
    """A section that gives the task's input data: its heading names the data (even with an example word), it is not
    a template the learner fills and not the task itself."""

    heading = idm_fold(str(section.get("heading") or ""))
    return (_INPUT_HEADING_RE.search(heading) is not None and _TEMPLATE_HEADING_RE.search(heading) is None
            and not any(block.get("kind") == "task" for block in _blocks(section)) and not _questions_table(section))


def is_example_section(section: Mapping[str, Any]) -> bool:
    """The worked example: a heading that names an example or a model entry, and no input data heading."""

    heading = idm_fold(str(section.get("heading") or ""))
    return any(word in heading for word in _EXAMPLE_WORDS) and not is_input_section(section)


_is_example = is_example_section


def _entry_texts(section: Mapping[str, Any]) -> list[str]:
    """The entries of a section: list items, table rows ("label: value") and texts."""

    entries: list[str] = []
    for block in _blocks(section):
        kind = block.get("kind")
        if kind == "table":
            entries.extend(f"{row.get('label') or ''}: {row.get('value') or ''}" for row in block.get("rows") or []
                           if isinstance(row, dict))
        elif kind in {"bullets", "steps"}:
            entries.extend(str(item) for item in block.get("items") or [])
        elif kind in _TEXT_KINDS:
            entries.append(str(block.get("text") or ""))
    return [entry for entry in entries if entry.strip(" :")]


def _name_and_body(entry: str) -> tuple[str | None, str]:
    if match := _NAMED_ENTRY_RE.match(entry):
        return idm_fold(match.group("name")), match.group("text")
    label, colon, rest = entry.partition(":")
    return None, rest if colon and rest.strip() and len(label.split()) <= _LABEL_MAX_WORDS else entry


def _pairs(text: str) -> set[tuple[str, str]]:
    return set(pairwise(re.findall(r"\w+", idm_fold(text))))


def _reused(entry: str, examples: Sequence[str]) -> bool:
    name, body = _name_and_body(entry)
    words, pairs = set(content_words(body)), _pairs(body)
    for example in examples:
        if (name is not None and re.search(rf"\b{re.escape(name)}\b", idm_fold(example))
                and len(words & set(content_words(example))) >= IDM_WORKSHEET_REUSE_MIN_WORDS):
            return True
        if (len(pairs) >= IDM_WORKSHEET_REUSE_MIN_PAIRS
                and len(pairs & _pairs(example)) / len(pairs) >= IDM_WORKSHEET_REUSE_PAIR_SHARE):
            return True
    return False


def reused_entries(component: Mapping[str, Any]) -> list[int]:
    """The 1-based positions of the input data entries the worked example works on (R2: the example then gives the
    learner the answer of the task), or ``[]``."""

    sections = _sections(component)
    inputs = [entry for section in sections if is_input_section(section) for entry in _entry_texts(section)]
    examples = [entry for section in sections if _is_example(section) for entry in _entry_texts(section)]
    if not inputs or not examples:
        return []
    return [position for position, entry in enumerate(inputs, 1) if _reused(entry, examples)]


def reuse_hint(positions: Sequence[int]) -> str:
    """The repair line's server words: which input entries reappear in the worked example (positions only)."""

    listed = ", ".join(str(position) for position in positions)
    return f" (input entries {listed} reappear in the worked example)" if positions else ""


def missing_input(component: Mapping[str, Any]) -> InputReference | None:
    """The data reference of a worksheet html slot whose data the worksheet does not give, or ``None``."""

    sections = _sections(component)
    blocks = [(position, block) for position, section in enumerate(sections) for block in _blocks(section)]
    self_check = next((block for _position, block in reversed(blocks) if block.get("kind") == "bullets"), None)
    examples = {position for position, section in enumerate(sections) if _is_example(section)}
    texts = [(position, block) for position, block in blocks
             if block.get("kind") in _TEXT_KINDS and position not in examples]
    reference = input_reference(str(block.get("text") or "") for _position, block in texts)
    if reference is None:
        return None
    candidates = [block for position, block in blocks if position not in examples and block is not self_check]
    forms = [block for block in candidates if block.get("kind") in {"table", "steps"}]
    need = max(_MIN_LIST_ENTRIES, reference.count or 0)
    # A list or table with the entries is the data unless it is the worksheet's only form (its template).
    if any(block.get("kind") in _LIST_KINDS and _entries(block) >= need
           and (block.get("kind") == "bullets" or len(forms) > 1) for block in candidates):
        return None
    if reference.kind in {"interview", "data"} and any(
            block.get("kind") in _TEXT_KINDS and input_reference([str(block.get("text") or "")]) is None
            and len(str(block.get("text") or "").split()) >= _MIN_TEXT_WORDS for block in candidates):
        return None
    return reference


def input_hint(reference: InputReference | None) -> str:
    """The repair line's server words for ``reference`` (never the writer's text); empty without one."""

    if reference is None:
        return ""
    what = {"list": "a list", "items": "the items", "interview": "an interview", "data": "the case or data"}
    count = f" ({reference.count} entries)" if reference.count else ""
    return f" (the task refers to {what.get(reference.kind, 'data')}{count} the worksheet does not give)"


def references_of(components: Sequence[Mapping[str, Any]]) -> list[InputReference | None]:
    """``missing_input`` of each component (``None`` for one that gives its data or refers to none)."""

    return [missing_input(component) for component in components]


__all__ = ["EXAMPLE_REUSE_CODE", "WORKSHEET_INPUT_CODE", "InputReference", "input_hint", "input_reference",
           "is_example_section", "is_input_section", "missing_input", "references_of", "reuse_hint",
           "reused_entries"]
