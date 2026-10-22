"""A drafted scenario reads as a supposition, never as a real past event (run 5496155c, QC 2026-10-22).

The run told three drafted cases as facts: "Trong quý vừa qua, bộ phận phát triển kinh doanh đã áp dụng quy trình 4
bước …", "Trong quý vừa qua, một chuyên viên … đã hoàn thiện mẫu …", "Trong quý vừa qua, khách hàng B2B gặp nút thắt
…". The writer is asked to open such a case with "Giả sử …" / "Hãy hình dung …"; what it still writes so is fixed here.

``as_supposition`` puts "Giả sử " ("Suppose that ") before a declarative sentence that opens with a relative past time
of the unit's language followed by a comma ("Trong quý vừa qua,", "Năm ngoái,", "Hồi tháng trước,"; "Last quarter,",
"In the past year,"), and lowers the time's capital. Safety (review of dfb6db2, item 3): only those fixed openings, at
the start of a sentence, with their comma; never a question ("Trong tuần qua, bạn đã nhận bao nhiêu …?"), an
instruction ("…, hãy liệt kê …") or a sentence about the learner's own past ("bạn đã"); never a longer time ("Trong
tháng trước Tết") and never a time the facts use themselves (read without its "Trong"/"Hồi"); only the language of the
unit. Two words are added and no word changes. It is applied to the stem of a question and to the texts, list items
and table values of an html slot, never to a worksheet's task.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from typing import Any, Final

from app.idm.text import EvidenceIndex, idm_fold

SUPPOSITION_ADDED: Final = "past_scenario_supposed"
_PAST: Final = {
    "vi": (r"(?:Trong[ \t]+)?(?:quý|năm|tháng|tuần|kỳ)[ \t]+(?:vừa[ \t]+qua|vừa[ \t]+rồi|trước|qua)"
           r"|Năm[ \t]+ngoái|Hồi[ \t]+(?:quý|năm|tháng|tuần)[ \t]+trước"),
    "en": r"(?:In[ \t]+the[ \t]+(?:last|past)[ \t]+|Last[ \t]+)(?:quarter|year|month|week)",
}
# The opening of a text or of a sentence inside it, then the past time and its comma.
_OPENING_RE: Final = {language: re.compile(rf"(?:^|(?<=[.!…][ \t])|(?<=\n))({pattern})(?=,)", re.IGNORECASE)
                      for language, pattern in _PAST.items()}
_SENTENCE_END_RE: Final = re.compile(r"[.!?…](?=\s|$)|\n")
# A full stop that ends an abbreviation, never a sentence ("chi nhánh TP. HCM", "ThS. Lan", "Dr. Smith", "v.v.").
_ABBREVIATION_RE: Final = re.compile(r"(?:(?<!\w)(?:TP|Tp|tp|Q|P|TS|ThS|PGS|GS|BS|KS|St|Mr|Mrs|Ms|Dr|No|Co|Ltd|Inc)"
                                     r"|v\.v|etc)\.$")
# A sentence that addresses the learner: an instruction or the learner's own past, never a scenario.
_LEARNER_RE: Final = {
    "vi": re.compile(r"(?<!\w)(?:hãy|bạn đã|bạn vừa)(?!\w)", re.IGNORECASE),
    "en": re.compile(r"(?<!\w)(?:you have|you did|you handled|please|list|write)(?!\w)", re.IGNORECASE)}
_FRAME_RE: Final = re.compile(r"^(?:trong|hoi|in the|last)\s+")
_LEAD: Final = {"vi": "Giả sử ", "en": "Suppose that "}


def _sourced(phrase: str, evidence: EvidenceIndex | None) -> bool:
    """The facts use the time themselves, with or without its framing word ("tháng trước" for "Trong tháng trước")."""

    tokens = _FRAME_RE.sub("", idm_fold(phrase)).split()
    return evidence is not None and all(pair in evidence.pairs for pair in pairwise(tokens)) and (
        len(tokens) > 1 or (bool(tokens) and tokens[0] in evidence.words))


def _sentence(text: str, start: int) -> str:
    """The sentence from ``start``: up to its end mark, never the full stop of an abbreviation."""

    for end in _SENTENCE_END_RE.finditer(text, start):
        if not _ABBREVIATION_RE.search(text, start, end.end()):
            return text[start:end.end()]
    return text[start:]


def _asks(text: str, start: int) -> bool:
    """Whether the line from ``start`` (to its line break or the end of the text) ends with a question mark: a
    question, however its sentences split (review 4: "Năm ngoái, chi nhánh TP. HCM … giao trễ?")."""

    end = text.find("\n", start)
    return text[start:end if end != -1 else len(text)].rstrip().endswith("?")


def as_supposition(text: str, locale: str, evidence: EvidenceIndex | None) -> tuple[str, int]:
    """``text`` with "Giả sử " before each declarative sentence that opens with a relative past time and its comma the
    facts do not use, and the number of sentences changed."""

    language = "en" if locale == "en" else "vi"
    normal = unicodedata.normalize("NFC", text)
    count = 0

    def suppose(match: re.Match[str]) -> str:
        nonlocal count
        phrase = match.group(1)
        sentence = _sentence(match.string, match.start())
        if (sentence.rstrip().endswith("?") or _asks(match.string, match.start())
                or _LEARNER_RE[language].search(sentence) or _sourced(phrase, evidence)):
            return phrase
        count += 1
        return _LEAD[language] + phrase[:1].lower() + phrase[1:]

    changed = _OPENING_RE[language].sub(suppose, normal)
    return (changed, count) if count else (text, 0)


def _texts(value: Any, rewrite: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return rewrite(value)
    if isinstance(value, list):
        return [_texts(item, rewrite) for item in value]
    return value


def suppose_semantic(semantic: dict[str, Any], locale: str, evidence: EvidenceIndex | None,
                     counts: Counter[str]) -> dict[str, Any]:
    """``semantic`` (an html slot's sections) with its texts, list items and table values as suppositions where they
    open with a past time (``as_supposition``); headings, row labels and a worksheet's task stay as written.
    ``counts`` gets the number of sentences changed."""

    added = 0

    def rewrite(text: str) -> str:
        nonlocal added
        changed, count = as_supposition(text, locale, evidence)
        added += count
        return changed

    def block_of(block: dict[str, Any]) -> dict[str, Any]:
        if block.get("kind") == "task":
            return block
        changed = dict(block)
        if isinstance(block.get("text"), str):
            changed["text"] = rewrite(block["text"])
        if isinstance(block.get("items"), list):
            changed["items"] = _texts(block["items"], rewrite)
        if isinstance(block.get("rows"), list):
            changed["rows"] = [{**row, "value": rewrite(row["value"])}
                               if isinstance(row, dict) and isinstance(row.get("value"), str) else row
                               for row in block["rows"]]
        return changed

    sections = [{**section, "blocks": [block_of(block) if isinstance(block, dict) else block
                                       for block in section.get("blocks") or []]}
                if isinstance(section, dict) else section for section in semantic.get("sections") or []]
    if not added:
        return semantic
    counts[SUPPOSITION_ADDED] += added
    return {**semantic, "sections": sections}


__all__ = ["SUPPOSITION_ADDED", "as_supposition", "suppose_semantic"]
