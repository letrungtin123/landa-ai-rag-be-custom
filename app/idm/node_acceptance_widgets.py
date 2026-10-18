"""Workspace widget rules of the Node acceptance mirror (``app.idm.node_acceptance``): FAQ, sortable,
crossword and diagram components.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any, Final

from app.idm.node_acceptance_problems import _MIN_CHOICES, PAYLOAD_INVALID, REFERENCE_INVALID, plain_text_valid
from app.idm.node_acceptance_text import (
    _COMBINING_RE,
    _JS_WS_RUN,
    _list,
    _nullish,
    _record,
    _Rejected,
    js_slice,
    js_trim,
    read_string,
)


def _faq_items(component: dict[str, Any]) -> list[tuple[int, str, str]]:
    """``normalizeFaqComponent`` items as (slot position, question, answer)."""

    raw_items = _list(component.get("items"))
    items = []
    for position, raw in enumerate(raw_items):
        item = _record(raw)
        question = read_string(_nullish(item.get("question"), item.get("q")), "", 500)
        answer = read_string(_nullish(item.get("answer"), item.get("a"), item.get("content")), "", 2_000)
        if question and answer:
            items.append((position, question, answer))
    return items[:8]


def _unique_key(value: str) -> str:
    return js_trim(unicodedata.normalize("NFKC", value)).lower()


def _later_duplicates(keys: Sequence[tuple[int, str]]) -> list[int]:
    seen: set[str] = set()
    duplicates: list[int] = []
    for position, key in keys:
        if key in seen:
            duplicates.append(position)
        seen.add(key)
    return duplicates


def _faq_findings(component: dict[str, Any]) -> None:
    items = _faq_items(component)
    if len(items) < _MIN_CHOICES:
        raise _Rejected("FAQ_ITEM_COUNT")
    bad = [position for position, question, answer in items
           if not plain_text_valid(question) or not plain_text_valid(answer)]
    if bad:
        raise _Rejected(PAYLOAD_INVALID, "FAQ_TEXT_FORBIDDEN_CHARACTER", bad)
    duplicates = _later_duplicates([(position, _unique_key(question)) for position, question, _answer in items])
    if duplicates:
        raise _Rejected(REFERENCE_INVALID, "FAQ_DUPLICATE_QUESTION", duplicates)


_SORTABLE_DEFAULT_QUESTION: Final = "Sap xep cac muc theo dung thu tu."
_MIN_SORTABLE_ITEMS: Final = 3
_MAX_SORTABLE_ITEMS: Final = 10


def _sortable_findings(component: dict[str, Any]) -> None:
    raw_items: list[Any] = next((_list(component[key]) for key in ("items", "ordered_items", "steps")
                                 if isinstance(component.get(key), list)), [])
    items = []
    for raw in raw_items:
        record = _record(raw)
        text = (read_string(raw, "", 500) if isinstance(raw, str)
                else read_string(_nullish(record.get("text"), record.get("label"), record.get("title")), "", 500))
        if text:
            items.append(text)
    items = items[:_MAX_SORTABLE_ITEMS]
    if len(items) < _MIN_SORTABLE_ITEMS:
        raise _Rejected("SORTABLE_ITEM_COUNT")
    question = read_string(_nullish(component.get("question_text"), component.get("question"),
                                    component.get("prompt")), _SORTABLE_DEFAULT_QUESTION, 500)
    if not plain_text_valid(question) or any(not plain_text_valid(item) for item in items):
        raise _Rejected(PAYLOAD_INVALID, "SORTABLE_TEXT_FORBIDDEN_CHARACTER")
    if _later_duplicates([(index, _unique_key(item)) for index, item in enumerate(items)]):
        raise _Rejected(REFERENCE_INVALID, "SORTABLE_DUPLICATE_ITEM")


_CROSSWORD_ANSWER_RE: Final = re.compile(r"^[A-Z0-9]{2,24}\Z")
_MIN_CROSSWORD_WORDS: Final = 3
_MAX_CROSSWORD_WORDS: Final = 10
_MAX_CROSSWORD_ANSWER: Final = 24


def _crossword_answer(value: Any) -> str:
    text = re.sub("\u0111", "D", read_string(value, "", 80), flags=re.IGNORECASE)
    text = _COMBINING_RE.sub("", unicodedata.normalize("NFD", text)).upper()
    return re.sub(r"[^A-Z0-9]", "", text)[:_MAX_CROSSWORD_ANSWER]


def _crossword_findings(component: dict[str, Any]) -> None:
    raw_words = _list(component.get("words"))
    words = []
    for index, raw in enumerate(raw_words):
        word = _record(raw)
        answer = _crossword_answer(_nullish(word.get("answer"), word.get("term"), word.get("text")))
        clue = read_string(_nullish(word.get("clue"), word.get("definition"), word.get("hint")), "", 500)
        hint = read_string(word.get("hint"), "", 500)
        if len(answer) >= _MIN_CHOICES and clue:
            words.append((index, answer, clue, hint))
    words = words[:_MAX_CROSSWORD_WORDS]
    if len(words) < _MIN_CROSSWORD_WORDS:
        raise _Rejected("CROSSWORD_WORD_COUNT")
    if any(not _CROSSWORD_ANSWER_RE.search(answer) or not plain_text_valid(clue) or re.search("[<>]", hint)
           for _row, answer, clue, hint in words):
        raise _Rejected(PAYLOAD_INVALID, "CROSSWORD_TEXT_FORBIDDEN_CHARACTER")
    # The renderer keeps the source row of every word; a dropped word leaves a gap Node rejects.
    if any(row != position for position, (row, _answer, _clue, _hint) in enumerate(words)) or _later_duplicates(
            [(position, _unique_key(answer)) for position, (_row, answer, _clue, _hint) in enumerate(words)]):
        raise _Rejected(REFERENCE_INVALID, "CROSSWORD_WORD_LAYOUT")


_DIAGRAM_ESCAPED_NEWLINE: Final = re.compile(r"(?:\\r\\n|\\n|\\r|(?<![^\W_])/n(?![^\W_]))", re.IGNORECASE)
_DIAGRAM_CONTROL: Final = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u2028\u2029]")
_MIN_DIAGRAM_NODES: Final = 2
_MAX_DIAGRAM_NODES: Final = 12


def _diagram_text(value: Any, maximum: int) -> str:
    """``normalizeDiagramDisplayText``."""

    if not isinstance(value, str):
        return ""
    text = _DIAGRAM_CONTROL.sub(" ", _DIAGRAM_ESCAPED_NEWLINE.sub(" ", value))
    return js_slice(js_trim(_JS_WS_RUN.sub(" ", text)), maximum)


def _diagram_findings(component: dict[str, Any]) -> None:
    """Node lays the diagram out itself; only its text rules are mirrored."""

    raw_nodes = _list(component.get("nodes"))
    texts: list[str] = []
    labels = 0
    for raw in raw_nodes:
        node = _record(raw)
        label = _diagram_text(read_string(raw, "", 120) if isinstance(raw, str) else read_string(
            _nullish(node.get("label"), node.get("title"), node.get("name")), "", 120), 120)
        if not label:
            continue
        labels += 1
        if labels > _MAX_DIAGRAM_NODES:
            break
        tooltip = read_string(_nullish(node.get("tooltip"), node.get("description"), node.get("summary")), label, 500)
        texts.extend([label, _diagram_text(tooltip, 500)])
    if labels < _MIN_DIAGRAM_NODES:
        raise _Rejected("DIAGRAM_NODE_COUNT")
    texts.append(_diagram_text(read_string(_nullish(component.get("name"), component.get("title")), "Main Diagram",
                                           120), 120))
    edges = _list(component.get("edges"))
    texts.extend(label for edge in edges if (label := _diagram_text(read_string(_record(edge).get("label"), "", 120),
                                                                     120)))
    if any(re.search("[<>]", text) for text in texts):
        raise _Rejected(PAYLOAD_INVALID, "DIAGRAM_TEXT_FORBIDDEN_CHARACTER")
