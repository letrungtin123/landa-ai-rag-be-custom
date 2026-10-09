"""PDF artefacts in learner text written by W5 (QC run ab8d67e1, R5).

The PDF reader keeps marks of the page in the facts the writer copies from: a "Row N:" label before every table
row ("Row 1: 01 Năng Lực Sẽ Trở Thành Best-in" in the commitment table of c5.l2.u1), Title Case Copied From The
Slide ("Hành Vi Khách Hàng Mới", "Người Phụ Trách Triển Khai"), words cut at a line end ("Best-in", "(KPI /").
The writer also glued two phrases into "Cách nhìn nhìn vào …". This module

* removes the row labels and an immediately repeated word (unless the source itself has it, as in "Tự Sở Hữu Hữu
  Hạn", or it is a Vietnamese reduplication such as "luôn luôn", "dần dần", "chung chung");
* gives short Vietnamese labels (section headings, table row labels, the lead of a "label: …" list item, short list
  items) sentence case, keeping acronyms, English words and personal or place names;
* finds the cut fragments the server cannot complete (``cut_fragments``): a review finding for the writer.

Deterministic and meaning-preserving: it never adds or rewrites words, only removes a label or a repeat and
lowers the case of Vietnamese words inside a label.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Collection
from typing import Any, Final

from app.idm.text import idm_fold

# A "Row N:" label anywhere in a text (the facts carry it at the start; a writer may glue two rows).
ROW_LABEL_ANY_RE: Final = re.compile(r"(?<!\w)row\s+\d{1,3}\s*:\s*", re.IGNORECASE)
_REPEAT_RE: Final = re.compile(r"(?<![\w-])([^\W\d_]{2,})(\s+)(\1)(?![\w-])", re.IGNORECASE)
# Vietnamese words whose doubling is a word of its own ("luôn luôn" always, "dần dần" gradually, "chung chung" vague).
_REDUPLICATIONS: Final = frozenset({
    "luôn", "dần", "từ", "mãi", "đâu", "ai", "người", "nhà", "ngày", "đêm", "năm", "tháng", "đời", "kiếp",
    "thường", "chung", "riêng", "xa", "gần", "lâu", "tầng", "lớp", "vừa", "bình", "đều", "rất", "hằng", "mỗi",
    "lần", "nơi", "chỗ", "khắp", "hàng", "that", "had",
})
_WORD_SPLIT_RE: Final = re.compile(r"(\s+)")
# Names keep their capitals inside a Title Case run. A personal name is a surname followed by a common middle name
# ("Nguyễn Văn An", "Trần Thị Mai"); a bare surname is often an ordinary word ("Xử Lý", "Đào Tạo", "Hồ Sơ").
_SURNAMES: Final = frozenset({
    "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "phan", "vũ", "võ", "đặng", "bùi", "đỗ", "hồ", "ngô", "dương",
    "lý", "đinh", "trương", "lương", "đào", "lâm", "mai", "cao", "tô", "hà", "đoàn", "trịnh", "vương",
})
_MIDDLE_NAMES: Final = frozenset({
    "văn", "thị", "hữu", "đức", "minh", "ngọc", "thanh", "quốc", "công", "xuân", "thu", "hoàng", "anh", "kim",
    "gia", "bảo", "đình", "quang", "thành", "trọng", "tuấn", "hồng", "phương", "mạnh", "tiến",
})
_NAME_MAX_WORDS: Final = 3
_PLACE_NAMES: Final = (("việt", "nam"), ("hồ", "chí", "minh"), ("hà", "nội"), ("đà", "nẵng"), ("hải", "phòng"),
                       ("cần", "thơ"), ("sài", "gòn"), ("thành", "phố"))
_PUNCTUATION: Final = ".,;:!?()\"'"
# An unaccented Vietnamese syllable: an initial, a vowel group, a final ("vi", "khai", "phi", "tranh").
_ASCII_SYLLABLE_RE: Final = re.compile(r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?[aeiouy]{1,3}"
                                       r"(?:ch|ng|nh|[cmnpt])?")
_TITLE_RUN_MIN_WORDS: Final = 3
_MIN_TITLE_WORD_CHARS: Final = 2
_SHORT_ITEM_MAX_WORDS: Final = 8
_LABEL_MAX_WORDS: Final = 8
_SENTENCE_STARTS: Final = (": ", ". ", "! ", "? ", "(", "- ", "– ", "— ")  # noqa: RUF001 - dashes
# A text cut at a line end: an open parenthesis or a joiner at its end, or a hyphenated compound left at "-in".
_CUT_END_RE: Final = re.compile(r"(?:[(/&,\-–—]|\([^)]*|\b[A-Z]\w*-(?:in|of|to|on|by|for|and))\s*$")  # noqa: RUF001


def _letters_vietnamese(word: str) -> bool:
    """A Vietnamese syllable: a letter outside ASCII (diacritics, đ), or an unaccented syllable ("Vi", "Khai",
    "Phi") that English words such as "Best", "Digital" or "Think" do not fit."""

    core = word.strip(_PUNCTUATION)
    return (any(char.isalpha() and not char.isascii() for char in core)
            or _ASCII_SYLLABLE_RE.fullmatch(core.lower()) is not None)


def _title_word(word: str) -> bool:
    """"Hàng", "Vi": a capital first letter, then lower-case letters only (never an acronym such as "AI")."""

    core = word.strip(_PUNCTUATION)
    return (len(core) >= _MIN_TITLE_WORD_CHARS and core[0].isupper() and core[1:].isalpha() and core[1:].islower()
            and core.isalpha())


def _names_in(words: list[str]) -> set[int]:
    """Positions of a run's words that belong to a personal or place name (lower-case words of the run)."""

    kept: set[int] = set()
    for start in range(len(words)):
        if words[start] in _SURNAMES and start + 1 < len(words) and words[start + 1] in _MIDDLE_NAMES:
            kept.update(range(start, min(start + _NAME_MAX_WORDS, len(words))))
        for place in _PLACE_NAMES:
            if tuple(words[start:start + len(place)]) == place:
                kept.update(range(start, start + len(place)))
    return kept


def sentence_case_vi(text: str) -> str:
    """``text`` with every run of 3+ Title Case words in which most words are Vietnamese lowered to sentence case.

    The first word of a run that starts the text (or a clause after ":" / "." / "(") keeps its capital; English
    words, acronyms and names (a surname with a middle name, a known place name) keep theirs.
    """

    parts = _WORD_SPLIT_RE.split(text)
    words = [index for index, part in enumerate(parts) if part and not part.isspace()]
    result = list(parts)
    run: list[int] = []

    def flush() -> None:
        vietnamese = sum(_letters_vietnamese(parts[index]) for index in run)
        if len(run) >= _TITLE_RUN_MIN_WORDS and 2 * vietnamese > len(run):
            kept = _names_in([parts[index].strip(_PUNCTUATION).lower() for index in run])
            for position, index in enumerate(run):
                word = parts[index]
                prefix = "".join(parts[:index])
                starts = position == 0 and (not prefix.strip() or prefix.endswith(_SENTENCE_STARTS))
                if not starts and position not in kept and _letters_vietnamese(word):
                    result[index] = word[:1].lower() + word[1:]
        run.clear()

    for index in words:
        word = parts[index]
        if _title_word(word):
            run.append(index)
            if word[-1] in ".,;:!?)":
                flush()
        elif not (run and word.isdigit()):  # "Trong 90 Ngày": a number does not end the run
            flush()
    flush()
    return "".join(result)


def _label_case(text: str, *, whole: bool = False) -> str:
    """Sentence case for a label (``whole``: a heading or a table row label), the lead of a "label: explanation"
    list item, and a short list item."""

    if whole:
        return sentence_case_vi(text)
    label, separator, rest = text.partition(":")
    if separator and rest.strip() and len(label.split()) <= _LABEL_MAX_WORDS:
        return sentence_case_vi(label) + separator + rest
    return sentence_case_vi(text) if len(text.split()) <= _SHORT_ITEM_MAX_WORDS else text


def clean_text(text: str, evidence_pairs: Collection[tuple[str, str]], fixes: Counter[str]) -> str:
    """``text`` without row labels and without a repeated word the source does not have."""

    cleaned = ROW_LABEL_ANY_RE.sub("", text)
    if cleaned != text:
        fixes["row_label_removed"] += 1

    def single(match: re.Match[str]) -> str:
        word = match.group(1)
        folded = idm_fold(word)
        if unicodedata.normalize("NFC", word).lower() in _REDUPLICATIONS or (folded, folded) in evidence_pairs:
            return match.group(0)
        fixes["repeated_word_removed"] += 1
        return word

    return _REPEAT_RE.sub(single, cleaned)


def _apply(value: Any, fix: Callable[[str], str]) -> Any:
    return fix(value) if isinstance(value, str) else value


def clean_semantic(semantic: dict[str, Any], evidence_pairs: Collection[tuple[str, str]], locale: str,
                   fixes: Counter[str]) -> dict[str, Any]:
    """An html slot's semantic content with the artefacts above removed (a new dict; the input is not changed)."""

    def text(value: str) -> str:
        return clean_text(value, evidence_pairs, fixes)

    def label(value: str, *, whole: bool = False) -> str:
        cleaned = text(value)
        cased = _label_case(cleaned, whole=whole) if locale == "vi" else cleaned
        if cased != cleaned:
            fixes["title_case_lowered"] += 1
        return cased

    def whole_label(value: str) -> str:
        return label(value, whole=True)

    if not isinstance(semantic.get("sections"), list):
        return semantic
    sections = []
    for section in semantic["sections"]:
        if not isinstance(section, dict) or not isinstance(section.get("blocks"), list):
            sections.append(section)  # an invalid shape stays for the validator to name
            continue
        blocks = []
        for block in section["blocks"]:
            if not isinstance(block, dict):
                blocks.append(block)
                continue
            cleaned = dict(block)
            if "text" in block:
                cleaned["text"] = _apply(block["text"], text)
            if isinstance(block.get("items"), list):
                cleaned["items"] = [_apply(item, label) for item in block["items"]]
            if isinstance(block.get("rows"), list):
                cleaned["rows"] = [{**row, "label": _apply(row.get("label"), whole_label),
                                    "value": _apply(row.get("value"), text)} if isinstance(row, dict) else row
                                   for row in block["rows"]]
            blocks.append(cleaned)
        sections.append({**section, "heading": _apply(section.get("heading"), whole_label), "blocks": blocks}
                        if "heading" in section else {**section, "blocks": blocks})
    return {**semantic, "sections": sections}


def clean_question(component: dict[str, Any], evidence_pairs: Collection[tuple[str, str]],
                   fixes: Counter[str]) -> dict[str, Any]:
    """A problem or FAQ payload without row labels and repeated words in its learner text."""

    def text(value: str) -> str:
        return clean_text(value, evidence_pairs, fixes)

    cleaned = dict(component)
    for key in ("question", "explanation"):
        if key in component:
            cleaned[key] = _apply(component[key], text)
    if isinstance(component.get("choices"), list):
        cleaned["choices"] = [{**choice, "text": _apply(choice.get("text"), text)} if isinstance(choice, dict)
                              else choice for choice in component["choices"]]
    if isinstance(component.get("items"), list):
        cleaned["items"] = [{**item, "question": _apply(item.get("question"), text),
                             "answer": _apply(item.get("answer"), text)} if isinstance(item, dict) else item
                            for item in component["items"]]
    return cleaned


def cut_fragments(semantic: Any) -> int:
    """List items, row labels and cells of an html slot that end cut ("Best-in", "(KPI /")."""

    if not isinstance(semantic, dict):
        return 0
    count = 0
    for section in semantic.get("sections") or []:
        for block in section.get("blocks") or [] if isinstance(section, dict) else []:
            if not isinstance(block, dict):
                continue
            entries = [*(block.get("items") or []), *(value for row in block.get("rows") or [] if isinstance(row, dict)
                                                      for value in (row.get("label"), row.get("value")))]
            count += sum(isinstance(entry, str) and _CUT_END_RE.search(entry.strip()) is not None
                         for entry in entries)
    return count


__all__ = ["ROW_LABEL_ANY_RE", "clean_question", "clean_semantic", "clean_text", "cut_fragments", "sentence_case_vi"]
