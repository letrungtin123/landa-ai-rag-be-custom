"""Source lines of an IDM fallback slot: repeated row labels, cut labels, missing spaces, glued words (run 2f8781d6).

QC of run 2f8781d6 (N1, heavy): the html fallback of "Bộ hồ sơ mẫu 3 vai trò" (pages 24-26, three profiles that share
the row labels "Về Chức năng (Functional Job)", "Về Cảm xúc (Emotional Pain)" …) put 8 source rows under the wrong
layer, because

* the shared cleaning drops every fact that repeats an earlier one, so the labels of the third profile were gone and
  its rows followed the second profile's last label (``source_facts``: a repeated row label is kept when a new section
  title came since its first use);
* the PDF reader had already dropped a repeated label line inside page 24, leaving the end of a cut label ("Pain): *
  Sợ bị coi là …"), which the renderer removed with its label (``OrphanLabels``: the end is joined to the one label
  start the slot cut the same way, "Về Ý nghĩa / Xã hội (Social/Meaningful" + "Pain)");
* a label kept twice must stay unique (``HTML_DUPLICATE_BLOCK`` counts every paragraph, list item and cell of 12
  characters or more, Python and Node alike), so a repeated label is prefixed with the short name of its section
  ("Người sử dụng trực tiếp" + ``LABEL_JOINER`` + "Về Cảm xúc (Emotional Job)", ``prefix_repeated_labels``).

The fallback text also restores the space the PDF lost after a full stop ("nội bộ.Phân loại" -> "nội bộ. Phân loại",
only between a lowercase letter and a capital followed by a lowercase letter: never "TP.HCM", "3.5", "v.v." or a URL)
and splits a glued word ("họphải") only when the slot's own facts write its parts as words and no other split of it
is possible (``tidy_facts``); the author note lists the glued words left.

Deterministic and meaning-preserving: no word is added except a repeated label's section name, which the source
states as that section's title.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final

from app.content.instructional_quality import TABLE_ROW_RE, clean_source_facts
from app.core.text import casefold_strip_marks
from app.idm.acronyms import keeps_capitals
from app.idm.text import glued_words

# The end of a label whose start the PDF reader dropped as a repeated line ("Pain): Sợ bị coi là …").
ORPHAN_LABEL_END_RE: Final = re.compile(r"^[^\W\d_]+\)[ \t]*:[ \t]*(?=\S)")
# A list marker before a table cell's label ("- Về Chức năng (Functional Pain)").
_LEAD_MARKER_RE: Final = re.compile(r"^(?:[-*•–][ \t]*)+(?=\S)")  # noqa: RUF001 - the en dash is a marker
# A section title written in capitals ("1.3 HỒ SƠ KHÁCH HÀNG B2B (VD3: NGƯỜI SỬ DỤNG TRỰC TIẾP)").
_SECTION_CAPITALS_SHARE: Final = 0.6
_MIN_SECTION_LETTERS: Final = 4
# Its short name: the bracket after a code ("(VD3: Người sử dụng trực tiếp)"), else the bracket, else the title without
# its number; a longer name is no prefix (the repeated label is then dropped, as before).
_CODED_BRACKET_RE: Final = re.compile(r"\(\s*[^\W_]{1,12}\s*:\s*([^()]{3,60}?)\s*\)\s*$")
_BRACKET_RE: Final = re.compile(r"\(\s*([^()]{3,60}?)\s*\)\s*$")
_NUMBERING_RE: Final = re.compile(r"^[\d.]+\s+")
_SHORT_NAME_MAX_CHARS: Final = 60
_SHORT_NAME_MAX_WORDS: Final = 8
LABEL_JOINER: Final = " – "  # noqa: RUF001 - the en dash joins the section name and the label
# A full stop the PDF glued to the next sentence: a lowercase letter (at most two closing marks may follow it, as in
# 'cần").Nâng'), the stop, then a capital and a lowercase letter.
_STOP_RE: Final = re.compile(r"\.(?=\S\S)")
_CLOSING: Final = ")\"'" + chr(0x201D) + chr(0x2019)
_MAX_CLOSING: Final = 2
# A web or e-mail address keeps its dots.
_ADDRESS_RE: Final = re.compile(r"://|^www\.|@", re.IGNORECASE)
# The word before the stop: two letters at least, not after another stop ("v.v.", "e.g." keep theirs).
_LAST_WORD_RE: Final = re.compile(r"(?<![.\w])[^\W\d_]{2,}$")
_WORD_RE: Final = re.compile(r"[^\W\d_]+")
# A glued word holds a handful of syllables at most; the split search stops after this many ways.
_MAX_SPLIT_WAYS: Final = 2

LabelTest = Callable[[str], bool]


def _fold(text: str) -> str:
    return casefold_strip_marks(text)


def _row_split(line: str) -> tuple[str, str]:
    """(the "Row N: " prefix or "", the rest) of a line."""

    match = TABLE_ROW_RE.match(line)
    if match is None:
        return "", line
    return line[:match.start(2)], match.group(2)


def label_text(line: str, is_label: LabelTest) -> str | None:
    """The label a line is (without its "Row N:" prefix and list marker), or ``None`` for a line of content, a
    table row of several cells or a section title."""

    _prefix, rest = _row_split(line)
    text = _LEAD_MARKER_RE.sub("", rest).strip()
    if not text or " | " in text or not is_label(text) or is_section(text, is_label):
        return None
    return text


def _capitals_share(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    return sum(char.isupper() for char in letters) / len(letters) if len(letters) >= _MIN_SECTION_LETTERS else 0.0


def is_section(line: str, is_label: LabelTest) -> bool:
    """A section title: a label line, no table row, written in capitals."""

    return (not TABLE_ROW_RE.match(line) and " | " not in line and is_label(line)
            and _capitals_share(line) >= _SECTION_CAPITALS_SHARE)


def short_name(title: str) -> str | None:
    """The short name of a section title, in sentence case ("1.3 HỒ SƠ KHÁCH HÀNG B2B (VD3: NGƯỜI SỬ DỤNG TRỰC TIẾP)"
    -> "Người sử dụng trực tiếp"), or ``None`` when it is too long to lead a label."""

    text = " ".join(title.split())
    coded, bracket = _CODED_BRACKET_RE.search(text), _BRACKET_RE.search(text)
    # A coded bracket names the case ("VD3: …"); else the title itself ("2.3 LỘ TRÌNH … (WORKBOOK)": not "Workbook").
    main = _NUMBERING_RE.sub("", text[:bracket.start()] if bracket else text).strip(" .:;-")
    candidates = [coded.group(1) if coded else "", main, bracket.group(1) if bracket else ""]
    name = next((value.strip(" .:;-") for value in candidates if value.strip(" .:;-")
                 and len(value) <= _SHORT_NAME_MAX_CHARS and len(value.split()) <= _SHORT_NAME_MAX_WORDS), "")
    if not name:
        return None
    if _capitals_share(name) >= _SECTION_CAPITALS_SHARE:
        words = [word if keeps_capitals(word) else word.lower() for word in name.split()]
        name = " ".join(words)
    return name[:1].upper() + name[1:]


def source_facts(values: Iterable[Any], is_label: LabelTest, *, keep_labels: bool) -> list[str]:
    """The cleaned facts of a fallback slot in source order (``clean_source_facts`` one by one), each said once, then
    tidied (``tidy_facts``). With ``keep_labels`` a row label repeated after a new section title is kept (its rows
    follow it); any other repeat goes, as the shared cleaning does."""

    cleaned = [text for value in values for text in clean_source_facts([value], preserve_table_numeric=True)]
    facts: list[str] = []
    first_section: dict[str, str | None] = {}
    section: str | None = None
    for text in tidy_facts(cleaned):
        key = _fold(text)
        if is_section(text, is_label):
            section = key if short_name(text) else None
        if key not in first_section:
            first_section[key] = section
            facts.append(text)
        elif (keep_labels and section is not None and first_section[key] != section
              and label_text(text, is_label) is not None):
            facts.append(text)
    return facts


class OrphanLabels:
    """The label starts of a slot the PDF cut before their last word ("Về Ý nghĩa / Xã hội (Social/Meaningful" +
    "Job): …"); an orphan label end ("Pain): …") is the same start cut the same way when the slot has exactly one."""

    def __init__(self) -> None:
        self.starts: dict[str, str] = {}

    def note(self, start: str, part: str) -> None:
        """``part`` closes the bracket of ``start`` as a cut label does: remember the start."""

        if ORPHAN_LABEL_END_RE.match(part) and start.count("(") > start.count(")"):
            self.starts.setdefault(_fold(start), start)

    def restore(self, orphan: str) -> str:
        """``orphan`` with its label start again when it is certain, else without its cut word (as before)."""

        if ORPHAN_LABEL_END_RE.match(orphan) and len(self.starts) == 1:
            return f"{next(iter(self.starts.values()))} {orphan}"
        return ORPHAN_LABEL_END_RE.sub("", orphan)


def prefix_repeated_labels(lines: Sequence[str], is_label: LabelTest) -> list[str]:
    """``lines`` with every label used again under another section title led by that section's short name, so no label
    repeats ("Row 2: Người sử dụng trực tiếp" + ``LABEL_JOINER`` + "Về Chức năng (Functional Job)"); a label that
    would still repeat (two sections of one short name) is left out, as the shared cleaning did."""

    result: list[str] = []
    first: dict[str, str | None] = {}
    shown: set[str] = set()
    section: tuple[str, str | None] | None = None
    for source in lines:
        if is_section(source, is_label):
            section = (_fold(source), short_name(source))
        label = label_text(source, is_label)
        if label is None:
            result.append(source)
            continue
        key = _fold(label)
        where = section[0] if section else None
        line = source
        if key in first and first[key] != where and section is not None and section[1]:
            label = f"{section[1]}{LABEL_JOINER}{label}"
            if _fold(label) in shown:
                continue
            shown.add(_fold(label))
            line = f"{_row_split(source)[0]}{label}"
        first.setdefault(key, where)
        result.append(line)
    return result


def restore_stops(text: str) -> str:
    """``text`` with a space after each full stop the PDF glued to the next sentence."""

    def keep(match: re.Match[str]) -> str:
        start = match.start()
        before = text[:start].rstrip(_CLOSING)
        after = text[match.end():match.end() + 2]
        token = text[text.rfind(" ", 0, start) + 1:].split(" ", 1)[0]
        word = _LAST_WORD_RE.search(before)
        glued = (word is not None and word.group(0)[-1].islower() and start - len(before) <= _MAX_CLOSING
                 and after[0].isalpha() and after[0].isupper() and after[1].isalpha() and after[1].islower()
                 and not _ADDRESS_RE.search(token))
        return ". " if glued else "."

    return _STOP_RE.sub(keep, text)


def _splits(word: str, vocabulary: frozenset[str]) -> list[list[str]]:
    """The ways ``word`` (lower case) is written as two vocabulary words or more, at most ``_MAX_SPLIT_WAYS``."""

    found: list[list[str]] = []

    def walk(start: int, parts: list[str]) -> None:
        if len(found) >= _MAX_SPLIT_WAYS:
            return
        if start == len(word):
            if len(parts) > 1:
                found.append(list(parts))
            return
        for end in range(start + 1, len(word) + 1):
            if word[start:end] in vocabulary:
                walk(end, [*parts, word[start:end]])

    walk(0, [])
    return found


def split_glued(text: str, vocabulary: frozenset[str]) -> str:
    """``text`` with each glued word split where its only split into words of ``vocabulary`` is."""

    for word in glued_words(text):
        lower = word.lower()
        ways = _splits(lower, vocabulary) if len(lower) == len(word) else []
        if len(ways) != 1:
            continue
        pieces, offset = [], 0
        for part in ways[0]:
            pieces.append(word[offset:offset + len(part)])
            offset += len(part)
        text = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", " ".join(pieces), text)
    return text


def tidy_facts(facts: Sequence[str]) -> list[str]:
    """``facts`` with the spaces after full stops restored and the glued words split that the facts themselves show
    how to split (their standalone words, ``split_glued``)."""

    spaced = [restore_stops(unicodedata.normalize("NFC", fact)) for fact in facts]
    glued = {word.lower() for fact in spaced for word in glued_words(fact)}
    vocabulary = frozenset(word.lower() for fact in spaced for word in _WORD_RE.findall(fact)
                           if word.lower() not in glued)
    return [split_glued(fact, vocabulary) for fact in spaced]


__all__ = ["LABEL_JOINER", "ORPHAN_LABEL_END_RE", "OrphanLabels", "is_section", "label_text", "prefix_repeated_labels",
           "restore_stops", "short_name", "source_facts", "split_glued", "tidy_facts"]
