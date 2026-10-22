"""What the author must check in the source (owner decision 2026-10-22, "Cần xem").

The old W2 rule held a block whose source is contradictory, unclear or out of date (24 hours in one place, 48 in
another) and asked the SME. Since the owner's decisions of 2026-10-22 nothing is held and no SME question is written:
the block is taught and its author note says, in one plain sentence per point, what the source says and where. The
sentence quotes the facts the W1 issue names (its ``fact_keys``) and the block the issue belongs to; never the AI's own
note, never a question. A block that only lacks grading criteria is not flagged: its practice uses suggested criteria
(A5).

Run 5496155c (2026-10-22, QC 7,2/10): 7 units were marked "Cần xem" and only 3 flags helped. So:

* only a conflict or an out-of-date point marks the unit for review (``REVIEW_FLAG_TYPES``, the existing way:
  ``quality_state`` review_required, "Cần xem" in the AI ID window); an unclear point is a soft edit note
  (``SOFT_FLAG_TYPES``), and a transition sentence of the source ("Chúng ta lại đào sâu thêm một lớp nữa, …") is no
  point at all;
* a quote is never a fragment (``clean_quote``): at least 4 words, mostly letters, no words the PDF reader glued
  together ("bộchân"); "(Chữ" and "I" were quoted. Without a clean quote the sentence names the block and the page;
* one sentence per kind of point and block, in the unit note and in the course note.

Where the issues come from: the course design's blocks (the course note, ``architecture``) and, for a unit, the
blocks its brief describes (``faq_material.blocks``; Node sends them for a unit with an FAQ slot).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from app.contracts.orchestration_v2.provider import SourceSnapshotFactV2
from app.idm.artifacts import ROW_LABEL_ANY_RE
from app.idm.contracts import IdmIssueV1, IdmUnitBriefV1
from app.idm.text import glued_words, idm_fold, single_line

# Owner decision 2026-10-22 (run 5496155c): a conflict or an out-of-date point marks the unit for review; an unclear
# one is a soft edit note.
REVIEW_FLAG_TYPES: Final = frozenset({"conflict", "outdated"})
SOFT_FLAG_TYPES: Final = frozenset({"unclear"})
FLAG_ISSUE_TYPES: Final = REVIEW_FLAG_TYPES | SOFT_FLAG_TYPES
# Review m6: a sentence quotes at most two facts of 60 characters (with their page) and a block name of 70.
_MAX_QUOTES: Final = 2
# The named facts read for a clean quote (a fragment among them leaves room for the next one).
_MAX_CANDIDATES: Final = 4
_QUOTE_CHARS: Final = 60
_NAME_CHARS: Final = 70
_MAX_PAGES: Final = 3
# A quote is a sentence the author can find, never a fragment: at least this many words, this share of letters among
# its non-space characters.
_QUOTE_MIN_WORDS: Final = 4
_QUOTE_MIN_LETTER_SHARE: Final = 0.6
# The punctuation and quotation marks around a word (straight and curly).
_QUOTE_MARKS: Final = ".,;:!?\"'()" + "".join(map(chr, (0x201C, 0x201D, 0x2018, 0x2019)))
# A quote that ends on one of these (accent-folded) stops where the PDF line did ("… các nhóm khách hàng của").
_DANGLING_ENDS: Final = frozenset({"cua", "va", "cho", "voi", "cac", "nhung", "la", "de", "hoac", "ve", "tu", "theo",
                                   "the", "of", "and", "to", "or", "for", "with", "a", "an"})
# A sentence of the source that only moves the talk on ("Chúng ta lại đào sâu thêm một lớp nữa, …", "Let's look at
# …"): an opening and a verb of moving within a few words (accent-folded). "Chúng ta phải phản hồi trong 24 giờ" is a
# rule, not a transition.
_TRANSITION_RE: Final = re.compile(
    r"^(?:chung ta|hay cung|bay gio|let'?s|let us|now|next|we will|we'll)\b[^.?!]{0,24}?"
    r"\b(?:dao sau|di sau|tim hieu|xem xet|xem qua|cung xem|chuyen sang|buoc sang|den voi|di vao|quay lai|nhin lai|"
    r"look at|turn to|move on|dig|explore|go deeper)\b")
# A unit note names at most this many points of each kind (the note has 1,500 characters).
MAX_UNIT_FLAGS: Final = 3
_NUMBER_RE: Final = re.compile(r"\d+(?:[.,:]\d+)*")
_LEAD_VI: Final = {"conflict": "Tài liệu ghi khác nhau", "unclear": "Tài liệu ghi chưa rõ",
                   "outdated": "Tài liệu có thể đã cũ"}
_LEAD_EN: Final = {"conflict": "The source differs", "unclear": "The source is unclear",
                   "outdated": "The source may be out of date"}


@dataclass(frozen=True)
class SourceFact:
    """A fact as the author can find it: its text and, when known, its page."""

    text: str
    page: int | None = None


@dataclass(frozen=True)
class UnitSourceFlags:
    """The points of a unit's blocks: ``review`` (conflicts and out-of-date points, the unit is marked for review) and
    ``soft`` (unclear points, an edit note)."""

    review: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)


def _plain(text: str) -> str:
    # The PDF reader's "Row N:" label is not the author's text (QLT-8).
    return " ".join(ROW_LABEL_ANY_RE.sub("", text).split())


def is_transition(text: str) -> bool:
    """A sentence of the source that only moves the talk on ("Chúng ta lại đào sâu thêm một lớp nữa, …")."""

    return _TRANSITION_RE.match(idm_fold(_plain(text))) is not None


def clean_quote(quote: str) -> bool:
    """A quote the author can find and read: at least ``_QUOTE_MIN_WORDS`` words, mostly letters, no glued words, no
    dangling end where the PDF line stopped, never a transition sentence (run 5496155c: "(Chữ", "I", "Xây dựng bộchân
    dung các nhóm khách hàng của")."""

    text = quote.strip(" …")
    words = [word for word in text.split() if any(char.isalpha() for char in word)]
    visible = [char for char in text if not char.isspace()]
    if len(words) < _QUOTE_MIN_WORDS or not visible:
        return False
    if sum(char.isalpha() for char in visible) / len(visible) < _QUOTE_MIN_LETTER_SHARE or glued_words(text):
        return False
    last = idm_fold(words[-1]).strip(_QUOTE_MARKS)
    cut_end = not quote.rstrip().endswith("…") and last in _DANGLING_ENDS
    return not cut_end and not is_transition(text)


def _excerpt(text: str, focus: Collection[str]) -> str:
    """``text`` within ``_QUOTE_CHARS``: its start, or, when a number of ``focus`` (one the other quoted facts do not
    state) would be cut, the words around that number, the cut sides marked with an ellipsis."""

    text = " ".join(text.split())
    if len(text) <= _QUOTE_CHARS:
        return text
    # The first differing number the start would cut (the start keeps those it holds).
    spot = next((match for match in _NUMBER_RE.finditer(text)
                 if match.group() in focus and match.end() > _QUOTE_CHARS - 1), None)
    if spot is None:
        return single_line(text, _QUOTE_CHARS)
    # Start at the word boundary before the window that centres the number.
    start = text.rfind(" ", 0, max(0, spot.start() - _QUOTE_CHARS // 2)) + 1
    end = min(len(text), start + _QUOTE_CHARS - 2)
    if end < len(text):
        end = text.rfind(" ", start, end) if text.rfind(" ", start, end) > spot.end() else end
    return ("…" if start else "") + text[start:end].strip(" ,;:") + ("…" if end < len(text) else "")


def _where(page: int | None, vi: bool) -> str:
    return "" if page is None else (f" (trang {page})" if vi else f" (page {page})")


def _named(issue: IdmIssueV1, facts: Mapping[str, SourceFact]) -> list[SourceFact]:
    return list(dict.fromkeys(facts[key] for key in issue.fact_keys if key in facts))


def _quotes(named: Sequence[SourceFact], vi: bool) -> list[str]:
    """The clean quotes of the facts an issue names, at most ``_MAX_QUOTES``, each quoted around a number the others do
    not state (the point that differs: "24 giờ" against "48 giờ")."""

    candidates = list(named)[:_MAX_CANDIDATES]
    # A number differs only between two facts or more; a single fact is quoted from its start.
    numbers = [set(_NUMBER_RE.findall(_plain(fact.text))) if len(candidates) > 1 else set() for fact in candidates]
    quotes: list[str] = []
    for index, fact in enumerate(candidates):
        others = set().union(*(other for position, other in enumerate(numbers) if position != index))
        excerpt = _excerpt(_plain(fact.text), numbers[index] - others)
        if clean_quote(excerpt):
            quotes.append(f"“{excerpt}”{_where(fact.page, vi)}")
    return list(dict.fromkeys(quotes))[:_MAX_QUOTES]


def _end(kind: str, vi: bool) -> str:
    if kind in SOFT_FLAG_TYPES:
        return " — nên đọc lại khi biên tập." if vi else " — reread it when editing."
    return ". Hãy kiểm tra trước khi áp dụng." if vi else ". Check this before applying it."


def _pages_text(pages: Sequence[int], vi: bool) -> str:
    shown = ", ".join(str(page) for page in sorted(set(pages))[:_MAX_PAGES])
    return (f" (trang {shown})" if vi else f" (page {shown})") if shown else ""


@dataclass
class _Point:
    """One kind of point of one block: its clean quotes and the pages of the facts it names."""

    kind: str
    name: str
    quotes: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


def _point_sentence(point: _Point, vi: bool) -> str:
    name = single_line(point.name, _NAME_CHARS)
    head = (f"{_LEAD_VI[point.kind]} ở phần “{name}”" if vi else f"{_LEAD_EN[point.kind]} in “{name}”")
    joiner = " và " if vi else " and "
    body = f"{head}: {joiner.join(point.quotes[:_MAX_QUOTES])}" if point.quotes else head + _pages_text(point.pages, vi)
    return body + _end(point.kind, vi)


def flag_sentence(block_name: str, issue: IdmIssueV1, facts: Mapping[str, SourceFact], locale: str,
                  partners: Sequence[str] = ()) -> str | None:
    """One plain sentence for a contradictory, unclear or out-of-date point of a block (``None`` for another issue, or
    an unclear point whose facts only move the talk on): what kind of point, the block it is in, and the clean quotes
    of the facts the issue names with their page (else the page alone). A conflict between sections carries no fact
    (W1-reduce, review m12): the sentence names the block and ``partners``, the other blocks of the same conflict, or
    "another part of the source"."""

    if issue.type not in FLAG_ISSUE_TYPES:
        return None
    vi = locale != "en"
    named = _named(issue, facts)
    if issue.type in SOFT_FLAG_TYPES and named and all(is_transition(fact.text) for fact in named):
        return None
    if issue.type == "conflict" and not named:
        name = single_line(block_name, _NAME_CHARS)
        others = [f"“{single_line(other, _NAME_CHARS)}”" for other in dict.fromkeys(partners) if other != block_name]
        between = (" và ".join(f"phần {other}" for other in others) if others else "một phần khác của tài liệu"
                   ) if vi else (" and ".join(others) if others else "another part of the source")
        return (f"Tài liệu ghi khác nhau giữa phần “{name}” và {between}" if vi
                else f"The source differs between “{name}” and {between}") + _end(issue.type, vi)
    point = _Point(issue.type, block_name, _quotes(named, vi), [fact.page for fact in named if fact.page is not None])
    return _point_sentence(point, vi)


def flag_sentences(blocks: Iterable[tuple[str, Sequence[IdmIssueV1]]], facts: Mapping[str, SourceFact],
                   locale: str, kinds: Collection[str] = FLAG_ISSUE_TYPES) -> list[str]:
    """The sentences of the flagged issues of ``blocks`` ((name, issues) pairs) whose type is one of ``kinds``, in
    order: one per kind of point and block, its quotes and pages merged (run 5496155c: the same block flagged twice),
    without repeats. The blocks that share one fact-less conflict (same note) are named in one sentence (review
    m12)."""

    vi = locale != "en"
    pairs = list(blocks)
    shared: dict[str, list[str]] = {}
    for name, issues in pairs:
        for issue in issues:
            if issue.type == "conflict" and not issue.fact_keys:
                shared.setdefault(issue.note, []).append(name)
    entries: list[str | _Point] = []
    points: dict[tuple[str, str], _Point] = {}
    told: set[frozenset[str]] = set()
    for name, issues in pairs:
        for issue in issues:
            if issue.type not in kinds or issue.type not in FLAG_ISSUE_TYPES:
                continue
            group = shared.get(issue.note, []) if issue.type == "conflict" and not issue.fact_keys else []
            if group:
                if frozenset(group) in told:
                    continue
                told.add(frozenset(group))
            named = _named(issue, facts)
            if group or (issue.type == "conflict" and not named):
                if sentence := flag_sentence(name, issue, facts, locale, partners=group):
                    entries.append(sentence)
                continue
            if issue.type in SOFT_FLAG_TYPES and named and all(is_transition(fact.text) for fact in named):
                continue
            key = (issue.type, name)
            point = points.get(key)
            if point is None:
                point = points[key] = _Point(issue.type, name)
                entries.append(point)
            point.quotes = list(dict.fromkeys([*point.quotes, *_quotes(named, vi)]))
            point.pages.extend(fact.page for fact in named if fact.page is not None)
    sentences = [entry if isinstance(entry, str) else _point_sentence(entry, vi) for entry in entries]
    return list(dict.fromkeys(sentences))


def fit_flags(sentences: Sequence[str], budget: int, locale: str, *, soft: bool = False) -> list[str]:
    """The sentences that fit ``budget`` characters in order (review m6: the unit note keeps room for its other
    lines), then one line counting the rest, which the course note lists."""

    vi = locale != "en"
    shown: list[str] = []
    used = 0
    for sentence in sentences:
        if used + len(sentence) + 1 > budget:
            break
        shown.append(sentence)
        used += len(sentence) + 1
    rest = len(sentences) - len(shown)
    if rest and soft:
        shown.append(f"Còn {rest} chỗ khác tài liệu ghi chưa rõ (xem ghi chú khoá học)." if vi
                     else f"{rest} more unclear point(s) in the source (see the course notes).")
    elif rest:
        shown.append(f"Còn {rest} chỗ khác cần kiểm tra trong tài liệu (xem ghi chú khoá học)." if vi
                     else f"{rest} more point(s) to check in the source (see the course notes).")
    return shown


def unit_flags(brief: IdmUnitBriefV1, source_facts: Iterable[SourceSnapshotFactV2], locale: str) -> UnitSourceFlags:
    """The points to check of the blocks a unit brief describes (``faq_material.blocks``), quoted from the unit's facts
    (with their page) or the lesson's context facts; at most ``MAX_UNIT_FLAGS`` of each kind."""

    material = brief.faq_material
    if material is None:
        return UnitSourceFlags()
    facts = {fact.fact_key: SourceFact(fact.fact_text, fact.source_page) for fact in source_facts}
    for fact in brief.lesson_context_facts:
        facts.setdefault(fact.fact_key, SourceFact(fact.fact_text))
    blocks = [(block.name, block.issues) for block in material.blocks]
    return UnitSourceFlags(flag_sentences(blocks, facts, locale, REVIEW_FLAG_TYPES)[:MAX_UNIT_FLAGS],
                           flag_sentences(blocks, facts, locale, SOFT_FLAG_TYPES)[:MAX_UNIT_FLAGS])


__all__ = ["FLAG_ISSUE_TYPES", "MAX_UNIT_FLAGS", "REVIEW_FLAG_TYPES", "SOFT_FLAG_TYPES", "SourceFact",
           "UnitSourceFlags", "clean_quote", "fit_flags", "flag_sentence", "flag_sentences", "is_transition",
           "unit_flags"]
