"""What the author must check in the source (owner decision 2026-10-22, "Cần xem").

The old W2 rule held a block whose source is contradictory, unclear or out of date (24 hours in one place, 48 in
another) and asked the SME. Since the owner's decisions of 2026-10-22 nothing is held and no SME question is written:
the block is taught, its unit is marked for the author's review the existing way (``quality_state``
review_required, "Cần xem" in the AI ID window) and its author note says, in one plain sentence per point, what the
source says and where. The sentence quotes the facts the W1 issue names (its ``fact_keys``) and the block the issue
belongs to; never the AI's own note, never a question. A block that only lacks grading criteria is not flagged: its
practice uses suggested criteria (A5).

Where the issues come from: the course design's blocks (the course note, ``architecture``) and, for a unit, the
blocks its brief describes (``faq_material.blocks``; Node sends them for a unit with an FAQ slot).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from app.contracts.orchestration_v2.provider import SourceSnapshotFactV2
from app.idm.artifacts import ROW_LABEL_ANY_RE
from app.idm.contracts import IdmIssueV1, IdmUnitBriefV1
from app.idm.text import single_line

FLAG_ISSUE_TYPES: Final = frozenset({"conflict", "unclear", "outdated"})
_MAX_QUOTES: Final = 3
# A unit note names at most this many points (the note has 1,500 characters).
MAX_UNIT_FLAGS: Final = 3
_QUOTE_CHARS: Final = 90
_NAME_CHARS: Final = 120
_LEAD_VI: Final = {"conflict": "Tài liệu ghi khác nhau", "unclear": "Tài liệu ghi chưa rõ",
                   "outdated": "Tài liệu có thể đã cũ"}
_LEAD_EN: Final = {"conflict": "The source differs", "unclear": "The source is unclear",
                   "outdated": "The source may be out of date"}


@dataclass(frozen=True)
class SourceFact:
    """A fact as the author can find it: its text and, when known, its page."""

    text: str
    page: int | None = None


def _quote(fact: SourceFact, vi: bool) -> str:
    where = "" if fact.page is None else (f" (trang {fact.page})" if vi else f" (page {fact.page})")
    # The PDF reader's "Row N:" label is not the author's text (QLT-8).
    return f"“{single_line(ROW_LABEL_ANY_RE.sub('', fact.text), _QUOTE_CHARS)}”{where}"


def flag_sentence(block_name: str, issue: IdmIssueV1, facts: Mapping[str, SourceFact], locale: str) -> str | None:
    """One plain sentence for a contradictory, unclear or out-of-date point of a block (``None`` for another issue):
    what kind of point, the block it is in, and the facts the issue names, quoted with their page."""

    if issue.type not in FLAG_ISSUE_TYPES:
        return None
    vi = locale != "en"
    quoted = list(dict.fromkeys(_quote(facts[key], vi) for key in issue.fact_keys if key in facts))[:_MAX_QUOTES]
    name = single_line(block_name, _NAME_CHARS)
    head = (f"{_LEAD_VI[issue.type]} ở phần “{name}”" if vi else f"{_LEAD_EN[issue.type]} in “{name}”")
    joiner = " và " if vi else " and "
    body = f"{head}: {joiner.join(quoted)}" if quoted else head
    return body + (". Hãy kiểm tra trước khi áp dụng." if vi else ". Check this before applying it.")


def flag_sentences(blocks: Iterable[tuple[str, Sequence[IdmIssueV1]]], facts: Mapping[str, SourceFact],
                   locale: str) -> list[str]:
    """The sentences of every flagged issue of ``blocks`` ((name, issues) pairs), in order, without repeats."""

    sentences = (flag_sentence(name, issue, facts, locale) for name, issues in blocks for issue in issues)
    return list(dict.fromkeys(sentence for sentence in sentences if sentence))


def unit_flags(brief: IdmUnitBriefV1, source_facts: Iterable[SourceSnapshotFactV2], locale: str) -> list[str]:
    """The points to check of the blocks a unit brief describes (``faq_material.blocks``), quoted from the unit's facts
    (with their page) or the lesson's context facts; at most ``MAX_UNIT_FLAGS``."""

    material = brief.faq_material
    if material is None:
        return []
    facts = {fact.fact_key: SourceFact(fact.fact_text, fact.source_page) for fact in source_facts}
    for fact in brief.lesson_context_facts:
        facts.setdefault(fact.fact_key, SourceFact(fact.fact_text))
    return flag_sentences(((block.name, block.issues) for block in material.blocks), facts, locale)[:MAX_UNIT_FLAGS]


__all__ = ["FLAG_ISSUE_TYPES", "MAX_UNIT_FLAGS", "SourceFact", "flag_sentence", "flag_sentences", "unit_flags"]
