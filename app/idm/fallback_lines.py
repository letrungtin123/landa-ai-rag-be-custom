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

Review 5: a step section ("BƯỚC 2: THỰC HIỆN") leads its labels by its words without the step marker, so a prefixed
label never reads as a numbered step (M3); a cut label gets its start back only when the slot writes that label in full
elsewhere (L2); every label a later section repeats is led by its section name, also when an earlier section wrote it as
the lead of a row ("Về Chức năng (Functional Job): …"), and a prefixed label is measured as a label without that name
(``label_core``, L3); a stop inside a name keeps its place (L5; review 6, L1: "Node.Js", "System.Console", "TP.HCM",
"Tp.Hồ Chí Minh" stay, "Việt Nam.Bạn", "tăng mạnh.Do đó", "journey.It" are mended).

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
# Review 5 (M3): the step marker of a section title ("BƯỚC 2:", "Step 3 -", "Giai đoạn 1.").
_STEP_MARKER_RE: Final = re.compile(r"^((?:bước|buoc|step|giai đoạn|phase|phần|part)\s*\d{1,3})\s*[:.)\-–—]?\s*",  # noqa: RUF001
                                    re.IGNORECASE)
# The lead of a row ("Về Chức năng (Functional Job): Sàng lọc …"): the text before its first colon.
_LEAD_RE: Final = re.compile(r"^([^:]{2,90}?)\s*:\s+\S")
# Review 6 (M2): a label the section name may lead has letters of its own ("Mục tiêu"; never "85%").
_MIN_LABEL_LETTERS: Final = 4
_SHORT_NAME_MAX_CHARS: Final = 60
_SHORT_NAME_MAX_WORDS: Final = 8
LABEL_JOINER: Final = " – "  # noqa: RUF001 - the en dash joins the section name and the label
# A full stop the PDF glued to the next sentence: a lowercase letter (at most two closing marks may follow it, as in
# 'cần").Nâng'), the stop, then a capital and a lowercase letter.
_STOP_RE: Final = re.compile(r"\.(?=\S\S)")
_CLOSING: Final = ")\"'" + chr(0x201D) + chr(0x2019)
_MAX_CLOSING: Final = 2
# Review 6 (L1): the abbreviations a dot ends inside a name ("Tp.Hồ Chí Minh", "Ths.Nguyễn", "Mr.Smith", "Co.Ltd").
_DOTTED_ABBREVIATIONS: Final = frozenset({"tp", "ths", "ts", "gs", "pgs", "bs", "ks", "mr", "mrs", "ms", "dr", "st",
                                          "co", "inc", "ltd", "corp", "jsc", "vd", "no"})
# A web or e-mail address keeps its dots.
_ADDRESS_RE: Final = re.compile(r"://|^www\.|@", re.IGNORECASE)
# The word before the stop: two letters at least (one Vietnamese letter with its mark: "Lưu ý.Đây"), not after another
# stop ("v.v.", "e.g." keep theirs).
_LAST_WORD_RE: Final = re.compile(r"(?<![.\w])(?:[^\W\d_]{2,}|[^\W\d_a-zA-Z])$")
_WORD_RE: Final = re.compile(r"[^\W\d_]+")
# The lead of an FAQ slot's title.
_FAQ_LEAD_RE: Final = re.compile(r"^\s*(?:hỏi\s*(?:-|–|&)?\s*đáp|câu hỏi thường gặp|faq|q\s*&\s*a)\s*[:\-–]\s*",  # noqa: RUF001
                                 re.IGNORECASE)
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
    # Review 5 (M3): a step's words without its marker ("Thực hiện"), or the marker alone ("Bước 2") with no words.
    step = _STEP_MARKER_RE.match(main)
    main = (main[step.end():].strip(" .:;-") or step.group(1)) if step else main
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


def _plain_label(text: str) -> str:
    """``text`` without its "Row N:" prefix, list marker and emphasis stars, folded."""

    return _fold(_LEAD_MARKER_RE.sub("", _row_split(text)[1].replace("*", " ")).strip())


def written_labels(facts: Iterable[str]) -> frozenset[str]:
    """The labels the facts of a slot write in full (a line or the lead of a row, folded), for ``OrphanLabels``."""

    found: set[str] = set()
    for fact in facts:
        for cell in _row_split(fact)[1].split(" | "):
            plain = _plain_label(cell)
            found.add(plain)
            if (lead := _LEAD_RE.match(plain)) is not None:
                found.add(lead.group(1).strip())
    return frozenset(found)


class OrphanLabels:
    """The label starts of a slot the PDF cut before their last word ("Về Ý nghĩa / Xã hội (Social/Meaningful" +
    "Job): …"); an orphan label end ("Pain): …") is the same start cut the same way when the slot has exactly one and
    writes that label in full elsewhere (``known``, review 5: never an invented label)."""

    def __init__(self, known: frozenset[str] = frozenset()) -> None:
        self.starts: dict[str, str] = {}
        self.known = known

    def note(self, start: str, part: str) -> None:
        """``part`` closes the bracket of ``start`` as a cut label does: remember the start."""

        if ORPHAN_LABEL_END_RE.match(part) and start.count("(") > start.count(")"):
            self.starts.setdefault(_fold(start), start)

    def restore(self, orphan: str) -> str:
        """``orphan`` with its label start again when it is certain, else without its cut word (as before)."""

        end = ORPHAN_LABEL_END_RE.match(orphan)
        if end is not None and len(self.starts) == 1:
            start = next(iter(self.starts.values()))
            if _plain_label(f"{start} {end.group(0).split(')', 1)[0]})") in self.known:
                return f"{start} {orphan}"
        return ORPHAN_LABEL_END_RE.sub("", orphan)


def _label_keys(line: str, is_label: LabelTest) -> tuple[str | None, list[str]]:
    """(the label a line is, its keys: the folded label and the lead of a row it opens, "Về Chức năng (Functional
    Job)" of "Về Chức năng (Functional Job): Sàng lọc …")."""

    label = label_text(line, is_label)
    lead = _LEAD_RE.match(_plain_label(line))
    keys = [key for key in (_fold(label) if label is not None else "",
                            lead.group(1).strip() if lead and is_label(lead.group(1)) else "") if key]
    return label, keys


def _sections(lines: Sequence[str], is_label: LabelTest) -> list[tuple[str, str | None] | None]:
    """The section (folded title, short name) each line is under, ``None`` before the first one."""

    found: list[tuple[str, str | None] | None] = []
    section: tuple[str, str | None] | None = None
    for line in lines:
        if is_section(line, is_label):
            section = (_fold(line), short_name(line))
        found.append(section)
    return found


class PrefixedLabel(str):
    """A label ``prefix_repeated_labels`` led with its section's name; ``core`` is the label alone. Review 6 (M1):
    only such a line is measured as a label without that name, never a source line that holds an en dash."""

    core: str

    def __new__(cls, text: str, core: str) -> PrefixedLabel:
        made = super().__new__(cls, text)
        made.core = core
        return made


def _joins_title(lines: Sequence[str], position: int, is_label: LabelTest) -> bool:
    """Review 6 (L7): the label at ``position`` comes right after its section title and before a line of content, so
    the renderer joins the two into one heading ("Bước 2: thực hiện - Mục tiêu của bước này"), unique by the title."""

    line = lines[position]
    previous = lines[position - 1] if position else ""
    following = lines[position + 1] if position + 1 < len(lines) else ""
    return (bool(previous) and is_section(previous, is_label) and is_label(line) and bool(following)
            and not is_label(following) and not TABLE_ROW_RE.match(following)
            and all(text.count("(") == text.count(")") for text in (previous, line)))


def prefix_repeated_labels(lines: Sequence[str], is_label: LabelTest) -> list[str]:
    """``lines`` with every label an earlier section already used (as a label or as the lead of a row, "Về Chức năng
    (Functional Job): Sàng lọc …") led by its section's short name, so no label repeats ("Row 2: Người sử dụng trực
    tiếp" + ``LABEL_JOINER`` + "Về Chức năng (Functional Job)"); a label that would still repeat (two sections of one
    short name) is left out, as the shared cleaning did. Review 6: never a line no other section uses, a figure or a
    data line ("85%", "Ngân sách tối đa 2 tỷ đồng", M2), nor a label joined with its section title (L7)."""

    sections = _sections(lines, is_label)
    first: dict[str, str | None] = {}
    result: list[str] = []
    shown: set[str] = set()
    for position, (source, section) in enumerate(zip(lines, sections, strict=True)):
        where = section[0] if section else None
        label, keys = _label_keys(source, is_label)
        repeated = label is not None and where is not None and any(key in first and first[key] != where
                                                                   for key in keys)
        for key in keys:
            first.setdefault(key, where)
        if (label is None or not repeated or section is None or not section[1]
                or sum(char.isalpha() for char in label) < _MIN_LABEL_LETTERS
                or _joins_title(lines, position, is_label)):
            result.append(source)
            continue
        prefixed = f"{section[1]}{LABEL_JOINER}{label}"
        if _fold(prefixed) not in shown:
            shown.add(_fold(prefixed))
            result.append(PrefixedLabel(f"{_row_split(source)[0]}{prefixed}", label))
    return result


def label_core(line: str) -> str:
    """The label a line is measured as: a ``PrefixedLabel`` without its section name ("Người sử dụng trực tiếp" +
    ``LABEL_JOINER`` + "Về Ý nghĩa / Xã hội (Social/Meaningful Job)" is a label of 9 words, review 5 L3); any other
    line as it is (review 6, M1)."""

    return line.core if isinstance(line, PrefixedLabel) else line


def restore_stops(text: str) -> str:
    """``text`` with a space after each full stop the PDF glued to the next sentence."""

    def keep(match: re.Match[str]) -> str:
        start = match.start()
        before = text[:start].rstrip(_CLOSING)
        after = text[match.end():match.end() + 2]
        token = text[text.rfind(" ", 0, start) + 1:].split(" ", 1)[0]
        word = _LAST_WORD_RE.search(before)
        following = _WORD_RE.match(text, match.end())
        # Review 6 (L1): a lower-case letter before the stop, a capital and a lower-case letter after it, never a name's
        # dot: a dotted abbreviation ("Tp.", "Mr.") or a code or namespace of two English words with a capital before
        # the dot ("Node.Js", "System.Console"; "TP.HCM" has no lower-case letter before it).
        name = (word is not None and following is not None and word.group(0).isascii()
                and following.group(0).isascii() and not word.group(0).islower())
        glued = (word is not None and word.group(0)[-1].islower() and start - len(before) <= _MAX_CLOSING
                 and after[0].isalpha() and after[0].isupper() and after[1].isalpha() and after[1].islower()
                 and word.group(0).lower() not in _DOTTED_ABBREVIATIONS and not name
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


def faq_topic(title: str) -> str:
    """An FAQ slot's title without its "Hỏi đáp:" / "FAQ:" lead, for the question of a fallback FAQ item (run
    2f8781d6: "Người học cần ghi nhớ gì về 'Hỏi đáp: Thực hành …'?" quoted the slot's own title)."""

    return _FAQ_LEAD_RE.sub("", title).strip() or title


__all__ = ["LABEL_JOINER", "ORPHAN_LABEL_END_RE", "OrphanLabels", "PrefixedLabel", "faq_topic", "is_section",
           "label_core", "label_text", "prefix_repeated_labels", "restore_stops", "short_name", "source_facts",
           "split_glued", "tidy_facts", "written_labels"]
