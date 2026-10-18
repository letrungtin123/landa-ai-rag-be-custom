"""Line rules of the "keep the document as it is" mode (QLT-6 ``preserve_source``).

The author uploads a finished course document. AI only splits it into chapters, lessons and units; every word the
learner sees is the document's own text, in order. This module holds the pure line rules every preserve stage shares:

* the learner-visible form of a source line and its tokens, the unit of the exact-text check. The Node mirror is
  ``lesson-author-idm-preserve.logic.ts`` and both use the ECMAScript white-space set, so a unit Python accepts is
  accepted by Node and the other way round (``tests/test_idm_preserve_text.py`` and the backend cross-language test
  pin the parity). A table row of the source snapshot (``Row N: a | b | c``) is shown as its cells: the ``Row N:``
  label and the `` | `` separators are parser markup, not words of the document;
* heading candidates (``Chương 1``, ``Bài 2``, ``1.1``, ``II.``, short ALL-CAPS lines, ...);
* page furniture: page numbers, and a short line repeated at a page edge on several pages (a running header or
  footer). Nothing else is ever dropped.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from app.contracts.orchestration_v2.provider import SourceSnapshotFactV2
from app.idm.policy import (
    IDM_HEADING_ALL_CAPS_RATIO,
    IDM_PRESERVE_FURNITURE_MAX_CHARS,
    IDM_PRESERVE_FURNITURE_MIN_PAGES,
    IDM_PRESERVE_TITLE_MAX_CHARS,
    IDM_PRESERVE_TITLE_MAX_WORDS,
    IDM_PRESERVE_TITLE_MIN_CHARS,
)
from app.idm.text import idm_fold

# ECMAScript WhiteSpace + LineTerminator (what JavaScript ``\s`` and ``String.prototype.trim`` use): TAB to CR,
# SPACE, NO-BREAK SPACE, OGHAM SPACE MARK, EN QUAD to HAIR SPACE, LINE/PARAGRAPH SEPARATOR, NARROW NO-BREAK SPACE,
# MEDIUM MATHEMATICAL SPACE, IDEOGRAPHIC SPACE and the byte-order mark.
_WS_CODES: Final = (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029,
                    0x202F, 0x205F, 0x3000, 0xFEFF)
WS_CHARS: Final = "".join(chr(code) for code in _WS_CODES)
WS_CLASS: Final = re.escape(WS_CHARS)
_WS_RUN: Final = re.compile(f"[{WS_CLASS}]+")
# The source snapshot encodes a table row as "Row <n>: cell | cell | ..." (``app.instructional_quality``).
TABLE_ROW_RE: Final = re.compile(f"Row[{WS_CLASS}]+([0-9]+)[{WS_CLASS}]*:[{WS_CLASS}]*([\\s\\S]+)", re.IGNORECASE)
_CELL_SPLIT_RE: Final = re.compile(f"[{WS_CLASS}]+\\|[{WS_CLASS}]+")
# A literal pipe inside a cell is stored as a broken bar by the table extractor.
_ESCAPED_PIPE: Final = chr(0xA6)
_TAG_RE: Final = re.compile(r"<[^>]+>")
_ENTITY_RE: Final = re.compile(r"&(?:#([0-9]{1,7})|#[xX]([0-9a-fA-F]{1,6})|([A-Za-z]{2,8}));")
# Exactly the entities the renderer, the Node normaliser and the course html sanitizer emit.
_NAMED_ENTITIES: Final = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": chr(0xA0)}
_MAX_CODE_POINT: Final = 0x10FFFF
_SURROGATES: Final = range(0xD800, 0xE000)
# Below SPACE: C0 control characters, which the workspace editor rejects in titles.
_FIRST_PRINTABLE: Final = 0x20

HeadingLevel = Literal[1, 2, 3]
# Folded chapter / lesson / section markers followed by a number ("Modun 1", "Bài 3", "Chương II", "Section 2").
_KEYWORD_LEVELS: Final[tuple[tuple[HeadingLevel, re.Pattern[str]], ...]] = (
    (1, re.compile(r"^(?:chuong|chapter|phan|part|module|modun|modul|hoc phan|tuan|week)\s*(?:[0-9]{1,3}|[ivxlc]{1,6})"
                   r"(?![0-9a-z])")),
    (2, re.compile(r"^(?:bai hoc|bai|lesson|unit)\s*(?:[0-9]{1,3}(?:\.[0-9]{1,3})*|[ivxlc]{1,6})(?![0-9a-z])")),
    (3, re.compile(r"^(?:muc|section|tiet)\s*(?:[0-9]{1,3}(?:\.[0-9]{1,3})*|[ivxlc]{1,6})(?![0-9a-z])")),
)
# "1.2 Title", "1.2.3. Title" (depth 2-3); "1. Title" and "II. Title" (depth 1, also the shape of a list item).
_DECIMAL_RE: Final = re.compile(r"^([0-9]{1,2}(?:\.[0-9]{1,2}){1,2})\.?\s+\S")
_SINGLE_NUMBER_RE: Final = re.compile(r"^(?:[0-9]{1,2}|[ivxlc]{1,6})[.)]\s+\S")
_ELLIPSIS: Final = chr(0x2026)
_DASHES: Final = "-" + chr(0x2013) + chr(0x2014)
_TERMINAL_RE: Final = re.compile(f"[.!?;:,{_ELLIPSIS}]$")
_TITLE_END_RE: Final = re.compile(f"[.;:,{_ELLIPSIS}]$")
_KEYWORD_HEADING_MAX_CHARS: Final = 160
_KEYWORD_HEADING_MAX_WORDS: Final = 20
_HEADING_MAX_CHARS: Final = 120
_HEADING_MAX_WORDS: Final = 15
_CAPS_MIN_LETTERS: Final = 4
# "Trang 12", "Page 3 of 10", "12/40", "- 7 -", "7": a page number on its own line (at most three digits, so a year
# such as "2025" alone on a line stays).
_PAGE_NUMBER_RE: Final = re.compile(
    r"^(?:(?:trang|page|tr\.?|p\.?)\s*[0-9]{1,4}(?:\s*(?:/|of|tren)\s*[0-9]{1,4})?"
    f"|[{_DASHES}]?\\s*[0-9]{{1,3}}\\s*[{_DASHES}]?|[0-9]{{1,3}}\\s*/\\s*[0-9]{{1,4}})$")
# A running header or footer sits in the first or last lines of its page.
_PAGE_EDGE_LINES: Final = 2

NoiseReason = Literal["page_number", "repeated_page_line"]


def normalize_line(text: str) -> str:
    """The line with white-space runs collapsed to one space and no edge white space (titles are kept so)."""

    return _WS_RUN.sub(" ", text).strip(WS_CHARS)


def tokens(text: str) -> list[str]:
    """Words of a text: the parts between ECMAScript white-space runs."""

    return [part for part in _WS_RUN.split(text) if part]


def table_cells(text: str) -> list[str] | None:
    """The cells of a source table-row line (``Row N: a | b``), or ``None`` for any other line."""

    match = TABLE_ROW_RE.fullmatch(text)
    if match is None:
        return None
    cells = [cell.strip(WS_CHARS).replace(_ESCAPED_PIPE, "|") for cell in _CELL_SPLIT_RE.split(match.group(2))]
    return [cell for cell in cells if cell]


def visible_text(text: str) -> str:
    """What a learner reads of one source line: a table row shows its cells, any other line itself."""

    cells = table_cells(text)
    return text if cells is None else " ".join(cells)


def expected_tokens(fact_texts: Sequence[str]) -> list[str]:
    """The words a kept unit must show, in order: the visible words of its lines."""

    return [token for text in fact_texts for token in tokens(visible_text(text))]


def _decode_entity(match: re.Match[str]) -> str:
    decimal, hexadecimal, name = match.groups()
    if name is not None:
        return _NAMED_ENTITIES.get(name, match.group(0))
    code = int(decimal) if decimal is not None else int(hexadecimal, 16)
    if code > _MAX_CODE_POINT or code in _SURROGATES:
        return match.group(0)
    return chr(code)


def html_tokens(html: str) -> list[str]:
    """Words a learner reads in an html fragment: tags are separators, entities are decoded once."""

    return tokens(_ENTITY_RE.sub(_decode_entity, _TAG_RE.sub(" ", html)))


# --- headings ----------------------------------------------------------------------------------------------
def keyword_level(text: str) -> HeadingLevel | None:
    """Level of a heading introduced by a chapter (1), lesson (2) or section (3) marker with its number."""

    line = normalize_line(text)
    if len(line) > _KEYWORD_HEADING_MAX_CHARS or len(line.split()) > _KEYWORD_HEADING_MAX_WORDS:
        return None
    if table_cells(line) is not None or _TERMINAL_RE.search(line):
        return None
    folded = idm_fold(line)
    for level, pattern in _KEYWORD_LEVELS:
        if pattern.match(folded):
            return level
    return None


def _heading_shaped(line: str) -> bool:
    return (0 < len(line) <= _HEADING_MAX_CHARS and len(line.split()) <= _HEADING_MAX_WORDS
            and table_cells(line) is None and not _TERMINAL_RE.search(line))


def numbering_depth(text: str) -> int | None:
    """Depth of a numbered heading: 2-3 for "1.2"/"1.2.3", 1 for "1."/"II." (which may also be a list item)."""

    line = normalize_line(text)
    if not _heading_shaped(line):
        return None
    folded = idm_fold(line)
    decimal = _DECIMAL_RE.match(folded)
    if decimal is not None:
        return decimal.group(1).count(".") + 1
    return 1 if _SINGLE_NUMBER_RE.match(folded) else None


def all_caps(text: str) -> bool:
    """A short ALL-CAPS line ("CHANGE MINDSET", "PHẦN MỞ ĐẦU")."""

    line = normalize_line(text)
    letters = [char for char in line if char.isalpha()]
    return (_heading_shaped(line) and len(letters) >= _CAPS_MIN_LETTERS
            and sum(char.isupper() for char in letters) / len(letters) >= IDM_HEADING_ALL_CAPS_RATIO)


def heading_hint(texts: Sequence[str], index: int) -> str:
    """The heading mark shown to the segmentation model: H1/H2/H3 (marker or numbering), H (heading-like), "-"."""

    text = texts[index]
    level = keyword_level(text)
    if level is not None:
        return f"H{level}"
    depth = numbering_depth(text)
    if depth is not None and depth > 1:
        return f"H{min(depth, 3)}"
    line = normalize_line(text)
    if all_caps(text) or (depth == 1 and len(line.split()) <= _HEADING_MAX_WORDS):
        return "H"
    following = normalize_line(texts[index + 1]) if index + 1 < len(texts) else ""
    if _heading_shaped(line) and len(line.split()) <= _HEADING_MAX_WORDS // 2 and len(following) > 2 * len(line):
        return "H"
    return "-"


def is_render_heading(text: str) -> bool:
    """A line the renderer shows as a section heading (text unchanged): a marker, "1.2" numbering or ALL CAPS."""

    depth = numbering_depth(text)
    return keyword_level(text) is not None or (depth is not None and depth > 1) or all_caps(text)


def title_eligible(text: str) -> bool:
    """A line that can be a chapter, lesson or unit title verbatim: editor-safe, heading-sized and not a sentence or a
    lead-in (a heading may end with "?" but not with ".", ":", ";", "," or an ellipsis)."""

    line = normalize_line(text)
    return (IDM_PRESERVE_TITLE_MIN_CHARS <= len(line) <= IDM_PRESERVE_TITLE_MAX_CHARS
            and len(line.split()) <= IDM_PRESERVE_TITLE_MAX_WORDS and table_cells(line) is None
            and not _TITLE_END_RE.search(line)
            and "<" not in line and ">" not in line and not any(ord(char) < _FIRST_PRINTABLE for char in line))


# --- page furniture ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Furniture:
    """Facts removed before segmentation, with their reason, and the counts the author note reports."""

    reasons: dict[str, NoiseReason]
    page_numbers: int
    repeated_lines: int


def _page_edges(facts: Sequence[SourceSnapshotFactV2]) -> set[int]:
    """Indexes of the first and last lines of each run of facts on the same page."""

    edges: set[int] = set()
    start = 0
    for index in range(1, len(facts) + 1):
        if index == len(facts) or facts[index].source_page != facts[start].source_page:
            run = range(start, index)
            edges.update(run[:_PAGE_EDGE_LINES])
            edges.update(run[-_PAGE_EDGE_LINES:])
            start = index
    return edges


def page_furniture(facts: Sequence[SourceSnapshotFactV2]) -> Furniture:
    """Page numbers, and running header/footer lines after their first occurrence.

    A repeated line counts only where it sits at a page edge, on at least three different pages, is short, is not a
    table row and does not end with ":" (a label such as "Lưu ý:" introduces content). Its first occurrence stays:
    a running header often repeats the real heading.
    """

    reasons: dict[str, NoiseReason] = {}
    for fact in facts:
        if _PAGE_NUMBER_RE.match(idm_fold(fact.fact_text)):
            reasons[fact.fact_key] = "page_number"
    edges = _page_edges(facts)
    pages: defaultdict[str, set[int]] = defaultdict(set)
    positions: defaultdict[str, list[int]] = defaultdict(list)
    for index, fact in enumerate(facts):
        line = normalize_line(fact.fact_text)
        if (index not in edges or fact.fact_key in reasons or fact.source_page is None
                or len(line) > IDM_PRESERVE_FURNITURE_MAX_CHARS or table_cells(line) is not None
                or line.endswith(":")):
            continue
        pages[line].add(fact.source_page)
        positions[line].append(index)
    for line, seen in pages.items():
        if len(seen) < IDM_PRESERVE_FURNITURE_MIN_PAGES:
            continue
        first = min(index for index, fact in enumerate(facts) if normalize_line(fact.fact_text) == line)
        for index in positions[line]:
            if index != first:
                reasons[facts[index].fact_key] = "repeated_page_line"
    values = list(reasons.values())
    return Furniture(reasons, values.count("page_number"), values.count("repeated_page_line"))
