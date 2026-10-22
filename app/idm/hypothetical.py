"""A drafted scenario reads as a supposition, never as a real past event (run 5496155c, QC 2026-10-22).

The run told three drafted cases as facts: "Trong quý vừa qua, bộ phận phát triển kinh doanh đã áp dụng quy trình 4
bước …", "Trong quý vừa qua, một chuyên viên … đã hoàn thiện mẫu …", "Trong quý vừa qua, khách hàng B2B gặp nút thắt
…". The writer is asked to open such a case with "Giả sử …" / "Hãy hình dung …"; what it still writes so is fixed here.

``as_supposition`` puts "Giả sử " ("Suppose that ") before a sentence that OPENS with a relative past time used as an
adverbial ("Trong quý vừa qua", "Trong năm qua", "Hồi tháng trước" or any such time followed by a comma: "Năm ngoái,",
"Tháng trước,"; "Last quarter", "In the past year"), and lowers the time's capital. Safety: only those fixed openings,
only at the start of a sentence, only two words are added and no word changes; a time that is the subject ("Quý trước
là …") stays, and a time the facts themselves use ("quý vừa qua" in the source) is the source's own account and stays
as written.
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
_PAST_VI: Final = (r"(?:Trong[ \t]+)?(?:quý|năm|tháng|tuần|kỳ)[ \t]+(?:vừa[ \t]+qua|vừa[ \t]+rồi|trước|qua)"
                   r"|Năm[ \t]+ngoái|Hồi[ \t]+(?:quý|năm|tháng|tuần)[ \t]+trước")
_PAST_EN: Final = r"(?:In[ \t]+the[ \t]+(?:last|past)[ \t]+|Last[ \t]+)(?:quarter|year|month|week)"
# The opening of a text or of a sentence inside it, then the past time and a comma or a space.
_OPENING_RE: Final = re.compile(rf"(?:^|(?<=[.!?…][ \t])|(?<=\n))({_PAST_VI}|{_PAST_EN})(?=[ \t,])", re.IGNORECASE)
# An opening that frames a time on its own ("Trong …", "Hồi …", "Last …", "In the …"); a bare one needs its comma.
_FRAMED_RE: Final = re.compile(r"(?:trong|hồi|last|in)[ 	]", re.IGNORECASE)
_LEAD: Final = {"vi": "Giả sử ", "en": "Suppose that "}


def _sourced(phrase: str, evidence: EvidenceIndex | None) -> bool:
    tokens = idm_fold(phrase).split()
    return evidence is not None and all(pair in evidence.pairs for pair in pairwise(tokens))


def as_supposition(text: str, locale: str, evidence: EvidenceIndex | None) -> tuple[str, int]:
    """``text`` with "Giả sử " before each sentence that opens with a relative past time the facts do not use, and the
    number of sentences changed."""

    normal = unicodedata.normalize("NFC", text)
    lead = _LEAD["en" if locale == "en" else "vi"]
    count = 0

    def suppose(match: re.Match[str]) -> str:
        nonlocal count
        phrase = match.group(1)
        adverbial = match.string[match.end():match.end() + 1] == "," or _FRAMED_RE.match(phrase) is not None
        if not adverbial or _sourced(phrase, evidence):
            return phrase
        count += 1
        return lead + phrase[:1].lower() + phrase[1:]

    changed = _OPENING_RE.sub(suppose, normal)
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
    open with a past time (``as_supposition``); headings and row labels stay as written. ``counts`` gets the number of
    sentences changed."""

    added = 0

    def rewrite(text: str) -> str:
        nonlocal added
        changed, count = as_supposition(text, locale, evidence)
        added += count
        return changed

    def block_of(block: dict[str, Any]) -> dict[str, Any]:
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
