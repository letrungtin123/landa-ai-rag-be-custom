"""Deterministic content-mode checks of W5 teaching html slots (QLT-5 C): cheap, no provider call.

Measured on real runs (plan 2026-10-13 §2.2): 14-28% of html sentences were near-verbatim copies of the source, no
unit said "nghĩa là" or "tức là", 6 of 89 units had an example or an explanation, and the slot-level 8-gram check
never fired (it needed >= 300 characters and a 0.7 share over the whole slot, and exempted keep rows). The checks
here work per sentence of a teaching slot (``fidelity.is_teaching_slot``):

* sentence copy (explain, structured): the share of near-verbatim sentences; sentences that carry a definition,
  rule, number or threshold are not judged, they stay exact by design;
* explanation (explain): at least one explanation device in a sentence the writer did not copy; QLT-7: a labelled
  example, a clarifying phrase ("nghĩa là", "tức là" ...) or a why sentence, never a generic word ("để", "vì") alone;
* sentence length (explain, QLT-7): prose sentences of a teaching slot average at most 22 words and none is longer
  than 40 (table rows, list items, callouts and sentences that state a rule are not measured);
* new facts (explain): a number, date, amount or capitalised name neither the facts nor the approved plan state; list
  labels ("Shift 1", "Trụ cột 01", where the facts never number that word) and the count of the list that follows
  ("gồm 5 bước:" before five items) are layout; a new number inside an AI-drafted example ("Ví dụ minh hoạ: …" and the
  list it introduces) is a notice for the author note, not a finding (owner decision 2026-10-20);
* example role (every mode, QLT-7): a drafted example ("Ví dụ minh hoạ: …") about a person of another level than the
  audience (``example_roles``);
* example label (explain): "Ví dụ:" before text the facts do not ground becomes "Ví dụ minh hoạ:" (a fix, not a
  finding); every labelled example is counted for the author note.

A finding gets one targeted repair, then the slot is kept for author review: a copy or explanation finding never
brings back the source-locked verbatim copy (``storyboard``).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from typing import Any, Final

from app.idm.contracts import FidelityMode, IdmUnitBriefV1
from app.idm.example_roles import AudienceRoles, audience_roles, contradicting_role, example_body
from app.idm.fidelity import brief_mode, example_label, teaching_slots
from app.idm.policy import (
    EXPLANATION_DEVICES_EN,
    EXPLANATION_DEVICES_VI,
    IDM_SENTENCE_COPY_MAX_SHARE_EXPLAIN,
    IDM_SENTENCE_COPY_MAX_SHARE_STRUCTURED,
    IDM_SENTENCE_COPY_MIN_SENTENCES,
    IDM_SENTENCE_COPY_NGRAM,
    IDM_SENTENCE_COPY_NGRAM_SHARE,
    IDM_SENTENCE_COPY_RUN_SHARE,
    IDM_SENTENCE_LENGTH_MAX_AVERAGE_WORDS,
    IDM_SENTENCE_LENGTH_MAX_WORDS,
    IDM_SENTENCE_LENGTH_MIN_SENTENCES,
    IDM_SENTENCE_MIN_WORDS,
)
from app.idm.text import EvidenceIndex, idm_fold, number_spans, number_tokens

SENTENCE_COPY_CODE: Final = "IDM_W5_SENTENCE_COPY"
EXPLANATION_MISSING_CODE: Final = "IDM_W5_EXPLANATION_MISSING"
NEW_FACT_CODE: Final = "IDM_W5_NEW_FACT"
# QLT-7 (QC run fae66524): long sentences in an explained slot; a drafted example about a person of another level.
SENTENCE_LENGTH_CODE: Final = "IDM_W5_SENTENCE_LENGTH"
EXAMPLE_ROLE_CODE: Final = "IDM_W5_EXAMPLE_ROLE"
FIDELITY_CODES: Final = frozenset({SENTENCE_COPY_CODE, EXPLANATION_MISSING_CODE, NEW_FACT_CODE, SENTENCE_LENGTH_CODE,
                                   EXAMPLE_ROLE_CODE})

_WORD_RE: Final = re.compile(r"\w+")
# Sentences end at . ! ? … ; (a colon introduces a definition or a label, it does not end one).
_SENTENCE_SPLIT_RE: Final = re.compile(r"(?<=[.!?…;])\s+")
# For proper names a colon starts a new sentence too ("Ví dụ minh hoạ: Một nhân viên …").
_NAME_SPLIT_RE: Final = re.compile(r"(?<=[.!?…:;])\s+")
_LEADING_PUNCTUATION_RE: Final = re.compile(r"^[\s\"'“”‘’(\[«‹–—•*-]+")  # noqa: RUF001 - quotes and dashes
# A list ordinal at the start of an item ("Bước 2:", "3.") is layout, not a fact.
_ORDINAL_RE: Final = re.compile(r"^\s*(?:(?:bước|step|mục|phần|giai đoạn|stage)\s+)?\d{1,2}\s*[.):]\s+",
                                re.IGNORECASE)
_RULE_WORDS: Final = (
    "phải", "không được", "bắt buộc", "cấm", "nghiêm cấm", "tuyệt đối", "chỉ được", "trừ trường hợp", "trừ khi",
    "tối đa", "tối thiểu", "ít nhất", "không quá", "quy định", "must", "shall", "never", "always", "required",
    "prohibited", "forbidden", "unless", "only if", "at least", "at most", "no more than", "maximum", "minimum",
)


def _alternation(words: Iterable[str]) -> re.Pattern[str]:
    ordered = sorted(dict.fromkeys(words), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(word) for word in ordered) + r")(?!\w)")


_RULE_RE: Final = _alternation(_RULE_WORDS)
_DEVICE_RE: Final = _alternation([*EXPLANATION_DEVICES_VI, *EXPLANATION_DEVICES_EN])
# A why sentence (QLT-7): it says why something matters ("… quan trọng vì …", "… is important because …"), or a
# heading or sentence that opens with "Vì sao" / "Tại sao" / "Why" and is answered by the section.
_WHY_RE: Final = re.compile(
    r"(?<!\w)(?:quan trọng|cần thiết|có ý nghĩa|then chốt|sống còn|có lợi|hữu ích)\s+(?:là\s+)?(?:vì|bởi|nhờ)(?!\w)"
    r"|^(?:vì sao|tại sao|why)(?!\w)"
    r"|(?<!\w)(?:matters|is important|is essential|is critical|helps you)\s+because(?!\w)")
# "Ví dụ:" opening a sentence labels an example (a source example keeps this label, ``label_examples``).
_EXAMPLE_OPENING_RE: Final = re.compile(r"^(?:ví dụ|for example|example)\s*:")
# "<term of at most 6 words> là / is …" defines the term (Vietnamese words are syllables: "sứ mệnh của công ty" is 5);
# "Điều này nghĩa là …" and "Đây là …" define nothing.
_DEFINITION_RE: Final = re.compile(
    r"^(?P<subject>(?:[\w-]+\s+){1,6}?)(?:là|được gọi là|được hiểu là|được định nghĩa là|is|are|means|refers to"
    r"|is defined as)\s")
_EXPLAINING_SUBJECT_ENDS: Final = frozenset({
    "nghĩa", "tức", "đó", "này", "đây", "vậy", "chính", "nó", "điều", "chúng", "this", "that", "it", "there", "here",
    "they"})
# "<label of at most 6 words>: …" states a term or a field ("Sứ mệnh: …"), unless the label introduces an
# explanation the writer adds ("Ví dụ minh hoạ:", "Dễ nhầm:", "Lưu ý:").
_LABEL_HEAD_RE: Final = re.compile(r"^(?P<label>[^:]{1,60}):\s*\S")
_LABEL_MAX_WORDS: Final = 6
_EXPLAINING_LABELS: Final = (
    "vi du", "chang han", "de nham", "luu y", "nghia la", "tuc la", "vi sao", "tai sao", "meo", "ket qua",
    "illustrative example", "example", "for example", "common mistake", "common confusion", "note", "tip", "why",
)
_LABELLED_EXAMPLE_HEADS: Final = ("vi du minh hoa", "illustrative example")
_UNLABELLED_EXAMPLE_RE: Final = re.compile(r"^\s*(?:ví dụ|for example|example)\s*[:,]\s*", re.IGNORECASE)
# Capitalised words that are no name ("OK"; "Ai" asks "who").
_COMMON_CAPITALISED: Final = frozenset({"ok", "ai"})
_TEXT_KINDS: Final = frozenset({"paragraph", "task", "warning"})
_LIST_KINDS: Final = frozenset({"bullets", "steps"})
# New facts (owner decisions 2026-10-20; QC runs ffaf7318, c0030787 and 43fabc57: 17 teaching slots flagged
# IDM_W5_NEW_FACT, most for "SHIFT 1" … "SHIFT 5", "Trụ cột 01" … "03" or "gồm 5 bước … sau:" before five items). A list
# label opens an item, a table row, a text or a heading: "<word> <1-2 digits>" then ":", ".", "(", ")" or a dash; the
# last word before the number is the label's key ("Trụ cột 01" -> "cot").
_LABEL_START_RE: Final = re.compile(
    r"^[\W_]*(?:[^\W\d_]+\s+){0,2}(?P<word>[^\W\d_]+)\s+(?P<number>\d{1,2})(?!\d)(?![.,]\d)"
    r"\s*(?=[:.)(\-–—]|$)")  # noqa: RUF001 - dashes
# The same label anywhere in a text ("… như Shift 3 đã nêu").
_LABEL_ANY_RE: Final = re.compile(r"(?<!\w)(?P<word>[^\W\d_]+)\s+(?P<number>\d{1,2})(?!\d)(?![.,]\d)")
# A count before a noun ("5 bước", "6 mắt xích"); a number before a unit is an amount, a duration or a share.
_COUNT_RE: Final = re.compile(r"(?<![\w.,])(?P<number>\d{1,2})(?!\d)(?![.,]\d)\s+(?P<noun>[^\W\d_]+)")
_UNIT_WORDS: Final = frozenset({
    "phan", "ngay", "tuan", "thang", "nam", "quy", "gio", "phut", "giay", "lan", "trieu", "ty", "nghin", "ngan",
    "dong", "usd", "vnd", "percent", "day", "days", "week", "weeks", "month", "months", "year", "years", "quarter",
    "quarters", "hour", "hours", "minute", "minutes", "second", "seconds", "times", "million", "billion", "thousand",
})
# Replaces an exempt label or count before the number and name checks: lower case, no digit, never a name.
_LAYOUT_MASK: Final = "x"
# A list label numbers its items with one or two digits ("Bước 2", "Trụ cột 01").
_LABEL_DIGITS: Final = 2
_MIN_LIST_COUNT: Final = 2


def _nfc_lower(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(text.split())).lower()


def words_of(text: str) -> list[str]:
    return _WORD_RE.findall(idm_fold(text))


def sentences_of(text: str) -> list[str]:
    return [part for part in _SENTENCE_SPLIT_RE.split(" ".join(text.split())) if part.strip()]


def carries_exact_claim(sentence: str) -> bool:
    """A sentence that states a number, a rule or threshold, or a definition: its wording may stay exact."""

    text = _nfc_lower(sentence)
    if any(char.isdigit() for char in text) or _RULE_RE.search(text):
        return True
    head = _LABEL_HEAD_RE.match(text)
    if head is not None and len(head.group("label").split()) <= _LABEL_MAX_WORDS and not idm_fold(
            head.group("label")).startswith(_EXPLAINING_LABELS):
        return True
    definition = _DEFINITION_RE.match(text)
    return definition is not None and definition.group("subject").split()[-1] not in _EXPLAINING_SUBJECT_ENDS


@dataclass(frozen=True)
class TextUnit:
    """One learner-visible text of an html slot and its JSON location inside the component."""

    location: str
    text: str
    kind: str


def _sections(component: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    semantic = component.get("semantic_content")
    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    return [section for section in sections if isinstance(section, dict)] if isinstance(sections, list) else []


def text_units(component: Mapping[str, Any]) -> list[TextUnit]:
    """Paragraph, task and warning texts, list items and table rows ("label: value"); headings apart."""

    units: list[TextUnit] = []
    for section_index, section in enumerate(_sections(component)):
        blocks = section.get("blocks")
        for block_index, block in enumerate(blocks if isinstance(blocks, list) else []):
            if not isinstance(block, dict):
                continue
            base = f"semantic_content.sections[{section_index}].blocks[{block_index}]"
            kind = str(block.get("kind") or "")
            if kind in _TEXT_KINDS and isinstance(block.get("text"), str):
                units.append(TextUnit(base, block["text"], kind))
            elif kind in _LIST_KINDS and isinstance(block.get("items"), list):
                units.extend(TextUnit(f"{base}.items[{position}]", item, kind)
                             for position, item in enumerate(block["items"]) if isinstance(item, str))
            elif kind == "table" and isinstance(block.get("rows"), list):
                units.extend(TextUnit(f"{base}.rows[{position}]", f"{row.get('label') or ''}: {row.get('value') or ''}",
                                      kind) for position, row in enumerate(block["rows"]) if isinstance(row, dict))
    return units


def headings(component: Mapping[str, Any]) -> list[str]:
    return [str(section.get("heading")) for section in _sections(component) if isinstance(section.get("heading"), str)]


class EvidenceGrams:
    """Accent-folded word n-grams of a slot's facts, per fact (no n-gram crosses two facts), cached per size."""

    def __init__(self, texts: Iterable[str]) -> None:
        self._facts = [words for words in (words_of(text) for text in texts) if words]
        self._grams: dict[int, frozenset[tuple[str, ...]]] = {}

    def grams(self, size: int) -> frozenset[tuple[str, ...]]:
        cached = self._grams.get(size)
        if cached is None:
            cached = frozenset(tuple(words[start:start + size]) for words in self._facts
                               for start in range(len(words) - size + 1))
            self._grams[size] = cached
        return cached

    def near_verbatim(self, words: Sequence[str]) -> bool:
        """At least ``IDM_SENTENCE_COPY_NGRAM_SHARE`` of the sentence's 5-grams are in the facts, or one run of the
        facts covers ``IDM_SENTENCE_COPY_RUN_SHARE`` of its words. Sentences under the minimum are not judged."""

        if len(words) < IDM_SENTENCE_MIN_WORDS:
            return False
        size = IDM_SENTENCE_COPY_NGRAM
        grams = [tuple(words[start:start + size]) for start in range(len(words) - size + 1)]
        known = self.grams(size)
        if grams and sum(gram in known for gram in grams) >= IDM_SENTENCE_COPY_NGRAM_SHARE * len(grams):
            return True
        run = math.ceil(IDM_SENTENCE_COPY_RUN_SHARE * len(words))
        target = self.grams(run)
        return any(tuple(words[start:start + run]) in target for start in range(len(words) - run + 1))


@dataclass(frozen=True)
class CopyReport:
    """Judged sentences of a slot and the locations of the near-verbatim ones (one entry per sentence)."""

    judged: int
    copied: tuple[str, ...]
    words: int = 0
    sentences: int = 0

    @property
    def share(self) -> float:
        return len(self.copied) / self.judged if self.judged else 0.0


def copy_report(component: Mapping[str, Any], grams: EvidenceGrams) -> CopyReport:
    judged = words = count = 0
    copied: list[str] = []
    for unit in text_units(component):
        for sentence in sentences_of(unit.text):
            tokens = words_of(sentence)
            words += len(tokens)
            count += bool(tokens)
            if len(tokens) < IDM_SENTENCE_MIN_WORDS or carries_exact_claim(sentence):
                continue
            judged += 1
            if grams.near_verbatim(tokens):
                copied.append(unit.location)
    return CopyReport(judged, tuple(copied), words, count)


def explains(text: str) -> bool:
    """``text`` holds an explanation device: a labelled example, a clarifying phrase ("nghĩa là", "tức là", "vì
    vậy" …) or a why sentence ("… quan trọng vì …", "Vì sao …?"); a generic word ("để", "giúp", "vì") alone is none."""

    lowered = _nfc_lower(text).lstrip(" \"'(“‘")  # noqa: RUF001 - opening quotes
    return bool(_DEVICE_RE.search(lowered) or _WHY_RE.search(lowered) or _EXAMPLE_OPENING_RE.match(lowered))


def explained(component: Mapping[str, Any], grams: EvidenceGrams) -> bool:
    """An explanation device (``explains``) in a sentence or heading the writer did not copy from the facts."""

    texts = [*headings(component),
             *(sentence for unit in text_units(component) for sentence in sentences_of(unit.text))]
    return any(explains(text) and not grams.near_verbatim(words_of(text)) for text in texts)


@dataclass(frozen=True)
class LengthReport:
    """Prose sentences of a slot (paragraph and task text, rules excluded): count, average and longest word counts,
    and the locations of the sentences over the average bound (one entry per text)."""

    sentences: int
    average: float
    longest: int
    long_locations: tuple[str, ...]

    @property
    def too_long(self) -> bool:
        return self.longest > IDM_SENTENCE_LENGTH_MAX_WORDS or (
            self.sentences >= IDM_SENTENCE_LENGTH_MIN_SENTENCES
            and self.average > IDM_SENTENCE_LENGTH_MAX_AVERAGE_WORDS)


def length_report(component: Mapping[str, Any]) -> LengthReport:
    """QLT-7: how long the prose sentences of an html slot are. Table rows, list items and callouts (a quoted rule)
    are not prose; a sentence that states a rule keeps the facts' wording and is not measured either."""

    counts: list[int] = []
    long_locations: list[str] = []
    for unit in text_units(component):
        if unit.kind not in {"paragraph", "task"}:
            continue
        for sentence in sentences_of(unit.text):
            words = len(words_of(sentence))
            if not words or _RULE_RE.search(_nfc_lower(sentence)):
                continue
            counts.append(words)
            if words > IDM_SENTENCE_LENGTH_MAX_AVERAGE_WORDS and unit.location not in long_locations:
                long_locations.append(unit.location)
    average = sum(counts) / len(counts) if counts else 0.0
    return LengthReport(len(counts), average, max(counts, default=0), tuple(long_locations))


def is_labelled_example(text: str) -> bool:
    return idm_fold(text).lstrip(" \"'(").startswith(_LABELLED_EXAMPLE_HEADS)


@dataclass(frozen=True)
class KnownFacts:
    """The numbers, words and adjacent word pairs a text may state: the facts it is built on plus the approved plan.
    Shared by the content-mode check (``FidelityChecker``) and the FAQ check of each answer (``app.idm.faq_checks``)."""

    numbers: frozenset[str]
    words: frozenset[str]
    # Adjacent folded word pairs: a name of several words ("Hồ Chí Minh") is known only as a whole.
    pairs: frozenset[tuple[str, str]]

    @classmethod
    def of(cls, texts: Iterable[str], *, plan_texts: Iterable[str] = (),
           evidence: EvidenceIndex | None = None) -> KnownFacts:
        """``texts`` (and ``evidence``, an index of more facts) and ``plan_texts``, whose numbers and names may be
        repeated."""

        numbers: set[str] = set(evidence.numbers) if evidence is not None else set()
        words: set[str] = set(evidence.words) if evidence is not None else set()
        pairs: set[tuple[str, str]] = set(evidence.pairs) if evidence is not None else set()
        for text in [*texts, *plan_texts]:
            numbers.update(number_tokens(text))
            tokens = words_of(text)
            words.update(tokens)
            pairs.update(pairwise(tokens))
        return cls(frozenset(numbers), frozenset(words), frozenset(pairs))

    def _known_name(self, run: Sequence[str]) -> bool:
        folded = [idm_fold(token) for token in run]
        if len(folded) == 1 and folded[0] in _COMMON_CAPITALISED:
            return True
        return all(word in self.words for word in folded) and all(pair in self.pairs for pair in pairwise(folded))

    def new_names(self, text: str) -> list[str]:
        """Runs of capitalised words (after a sentence's first word) the facts and the plan do not contain: a single
        word must be known, a run of several ("Hà Nội") must appear as a whole."""

        found: list[str] = []
        for sentence in _NAME_SPLIT_RE.split(unicodedata.normalize("NFC", " ".join(text.split()))):
            run: list[str] = []
            for token in [*_WORD_RE.findall(_LEADING_PUNCTUATION_RE.sub("", sentence))[1:], ""]:
                if len(token) > 1 and token[0].isupper():
                    run.append(token)
                    continue
                if run and not self._known_name(run):
                    found.append(" ".join(run))
                run = []
        return found

    def new_tokens(self, text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The numbers (as written) and the names of ``text`` the facts and the plan do not state, in reading order (a
        list ordinal at its start is layout, not a fact). For a repair prompt only: never logged."""

        stripped = _ORDINAL_RE.sub("", text, count=1)
        numbers = tuple(dict.fromkeys(raw for raw, normal, _start, _end in number_spans(stripped)
                                      if normal not in self.numbers))
        return numbers, tuple(dict.fromkeys(self.new_names(stripped)))

    def states_new_fact(self, text: str) -> bool:
        """``text`` states a number or a name the facts and the plan do not (a list ordinal is layout, not a fact)."""

        numbers, names = self.new_tokens(text)
        return bool(numbers or names)


@dataclass(frozen=True)
class NewFactHit:
    """A text of a teaching slot that states a number or a name the facts and the plan do not: its location, the new
    numbers (as written) and names, and whether it sits in an AI-drafted example (``example_segments``)."""

    location: str
    numbers: tuple[str, ...]
    names: tuple[str, ...]
    example: bool

    @property
    def review_tokens(self) -> tuple[str, ...]:
        """The tokens that make the text a finding: every name, and the numbers outside a drafted example (owner
        decision 2026-10-20: a number inside one is a notice)."""

        return self.names + (() if self.example else self.numbers)


def label_heads(texts: Iterable[str]) -> dict[str, set[int]]:
    """The numbers each list-label key opens ``texts`` with ("Shift 1: …", "Trụ cột 02 - …" -> {"shift": {1}, …})."""

    heads: dict[str, set[int]] = {}
    for text in texts:
        match = _LABEL_START_RE.match(unicodedata.normalize("NFC", text))
        if match is not None:
            heads.setdefault(idm_fold(match.group("word")), set()).add(int(match.group("number")))
    return heads


def masked_labels(text: str, labels: Mapping[str, int]) -> str:
    """``text`` with every "<key> <n>" of an exempt label sequence (``labels``: key -> its last number) masked."""

    def mask(match: re.Match[str]) -> str:
        last = labels.get(idm_fold(match.group("word")))
        return _LAYOUT_MASK if last is not None and 1 <= int(match.group("number")) <= last else match.group(0)

    return _LABEL_ANY_RE.sub(mask, unicodedata.normalize("NFC", text))


def masked_count(text: str, count: int) -> str:
    """``text`` with the count "<count> <noun>" of the list that follows it masked (never an amount or a duration)."""

    def mask(match: re.Match[str]) -> str:
        counted = int(match.group("number")) == count and idm_fold(match.group("noun")) not in _UNIT_WORDS
        return f"{_LAYOUT_MASK} {match.group('noun')}" if counted else match.group(0)

    return _COUNT_RE.sub(mask, text)


def _block_units(block: Mapping[str, Any], base: str) -> list[str]:
    """The text-unit locations of a list or table block (one per item or row)."""

    kind = block.get("kind")
    if kind in _LIST_KINDS and isinstance(block.get("items"), list):
        return [f"{base}.items[{position}]" for position, item in enumerate(block["items"]) if isinstance(item, str)]
    if kind == "table" and isinstance(block.get("rows"), list):
        return [f"{base}.rows[{position}]" for position, row in enumerate(block["rows"]) if isinstance(row, dict)]
    return []


def list_counts(component: Mapping[str, Any]) -> dict[str, int]:
    """Text location -> the item count of the list or table block right after it in its section."""

    counts: dict[str, int] = {}
    for section_index, section in enumerate(_sections(component)):
        for block_index, (block, after) in enumerate(pairwise(section.get("blocks") or [])):
            if not isinstance(block, dict) or not isinstance(after, dict) or block.get("kind") not in _TEXT_KINDS:
                continue
            items = len(_block_units(after, ""))
            if items >= _MIN_LIST_COUNT:
                counts[f"semantic_content.sections[{section_index}].blocks[{block_index}]"] = items
    return counts


def example_segments(component: Mapping[str, Any]) -> set[str]:
    """Locations of the AI-drafted examples of an html slot: a text or list item opening with the illustrative label
    ("Ví dụ minh hoạ:", ``is_labelled_example``), the list or table blocks right after such a text, and every block of
    a section whose heading opens with the label."""

    found: set[str] = set()
    for section_index, section in enumerate(_sections(component)):
        whole = is_labelled_example(str(section.get("heading") or ""))
        inside = False
        for block_index, block in enumerate(section.get("blocks") or []):
            if not isinstance(block, dict):
                continue
            base = f"semantic_content.sections[{section_index}].blocks[{block_index}]"
            if block.get("kind") in _TEXT_KINDS:
                inside = is_labelled_example(str(block.get("text") or ""))
                if inside or whole:
                    found.add(base)
            elif block.get("kind") in _LIST_KINDS and isinstance(block.get("items"), list):
                found.update(f"{base}.items[{position}]" for position, item in enumerate(block["items"])
                             if isinstance(item, str) and (inside or whole or is_labelled_example(item)))
            elif inside or whole:
                found.update(_block_units(block, base))
    return found


@dataclass
class FidelityChecker:
    """The content mode of one unit and what its checks compare against (built once per unit)."""

    mode: FidelityMode
    locale: str
    teaching: frozenset[int]
    # Teaching slots whose treatments are all "keep": exempt from the copy check in structured mode (as before).
    keep_only: frozenset[int]
    slot_grams: Mapping[int, EvidenceGrams]
    known: KnownFacts
    # QLT-7: the roles of the course audience; a drafted example about a person of another level is a finding.
    roles: AudienceRoles = field(default_factory=lambda: AudienceRoles(frozenset(), ()))

    @classmethod
    def build(cls, brief: IdmUnitBriefV1, fact_text: Mapping[str, str], evidence: EvidenceIndex, *,
              plan_texts: Iterable[str], locale: str) -> FidelityChecker:
        """``fact_text``: the unit's source facts; ``evidence``: everything the writer was given; ``plan_texts``:
        the approved plan (titles, purpose, server-written support briefs), whose numbers and names may be
        repeated."""

        teaching = frozenset(teaching_slots(brief))
        everything = list(fact_text.values())
        grams: dict[int, EvidenceGrams] = {}
        for index in teaching:
            slot = brief.components[index]
            keys = [key for key in dict.fromkeys([*slot.owned_fact_keys, *slot.supporting_fact_keys])
                    if key in fact_text]
            grams[index] = EvidenceGrams([fact_text[key] for key in keys] if keys else everything)
        keep_only = frozenset(index for index in teaching if brief.components[index].treatments
                              and {item.treatment for item in brief.components[index].treatments} == {"keep"})
        return cls(brief_mode(brief), locale, teaching, keep_only, grams,
                   KnownFacts.of((), plan_texts=plan_texts, evidence=evidence),
                   audience_roles(brief.target_audience, brief.purpose_profile, locale))

    def report(self, index: int, component: Mapping[str, Any]) -> CopyReport:
        return copy_report(component, self.slot_grams[index])

    def copy_limit(self, index: int) -> float | None:
        """The highest near-verbatim share of slot ``index``; ``None`` when the slot has no copy check."""

        if self.mode == "explain":
            return IDM_SENTENCE_COPY_MAX_SHARE_EXPLAIN
        if self.mode == "structured" and index not in self.keep_only:
            return IDM_SENTENCE_COPY_MAX_SHARE_STRUCTURED
        return None

    def copied_too_much(self, index: int, component: Mapping[str, Any]) -> bool:
        limit = self.copy_limit(index)
        if limit is None:
            return False
        report = self.report(index, component)
        return report.judged >= IDM_SENTENCE_COPY_MIN_SENTENCES and report.share > limit

    def new_names(self, text: str) -> list[str]:
        return self.known.new_names(text)

    @cached_property
    def numbered_words(self) -> frozenset[str]:
        """Folded words the facts or the plan write a number right after ("Bậc 01"): their numbering is a fact."""

        return frozenset(first for first, second in self.known.pairs
                         if second.isdigit() and len(second) <= _LABEL_DIGITS)

    def list_labels(self, component: Mapping[str, Any]) -> dict[str, int]:
        """Label key -> last number of each list-label sequence the slot opens its items, rows, texts or headings with,
        when the numbers run 1, 2 … without a gap and the facts never number that word (a "Bậc 07" after the facts'
        "Bậc 01" … "Bậc 06" stays a new fact)."""

        heads = label_heads([*headings(component), *(unit.text for unit in text_units(component))])
        return {key: max(numbers) for key, numbers in heads.items()
                if key not in self.numbered_words and numbers == set(range(1, max(numbers) + 1))}

    def new_fact_hits(self, component: Mapping[str, Any]) -> list[NewFactHit]:
        """Texts with a number or a name the facts and the plan do not state, list labels and the count of the list
        that follows a text left out (layout, not facts)."""

        labels = self.list_labels(component)
        counts = list_counts(component)
        examples = example_segments(component)
        hits: list[NewFactHit] = []
        for unit in text_units(component):
            text = masked_labels(unit.text, labels) if labels else unit.text
            if unit.location in counts:
                text = masked_count(text, counts[unit.location])
            numbers, names = self.known.new_tokens(text)
            if numbers or names:
                hits.append(NewFactHit(unit.location, numbers, names, unit.location in examples))
        return hits

    def new_fact_locations(self, component: Mapping[str, Any]) -> list[str]:
        """Locations of the texts that make the slot a new-fact finding (never the text)."""

        return [hit.location for hit in self.new_fact_hits(component) if hit.review_tokens]

    def example_numbers(self, components: Sequence[Any]) -> dict[int, int]:
        """Explain mode, per teaching slot: the AI-drafted example texts that state a number the facts do not (a notice
        for the author note, owner decision 2026-10-20)."""

        counts: dict[int, int] = {}
        for index in sorted(self.teaching) if self.mode == "explain" else ():
            component = components[index] if index < len(components) else None
            if isinstance(component, dict) and component.get("type") == "html":
                count = sum(hit.example and bool(hit.numbers) for hit in self.new_fact_hits(component))
                if count:
                    counts[index] = count
        return counts

    def role_locations(self, component: Mapping[str, Any]) -> list[str]:
        """Locations of drafted examples whose main person is of another level than the audience (QLT-7)."""

        return [unit.location for unit in text_units(component)
                if (body := example_body(unit.text)) is not None and contradicting_role(self.roles, body)]

    def findings(self, components: Sequence[Any]) -> list[tuple[str, int]]:
        """(code, slot) of every content-mode finding of the unit's teaching html slots, and of every html slot whose
        drafted example is about a person of another level than the audience (any mode)."""

        found: list[tuple[str, int]] = []
        for index, component in enumerate(components):
            if isinstance(component, dict) and component.get("type") == "html" and self.role_locations(component):
                found.append((EXAMPLE_ROLE_CODE, index))
        if self.mode == "keep":
            return found
        for index in sorted(self.teaching):
            component = components[index] if index < len(components) else None
            if not isinstance(component, dict) or component.get("type") != "html":
                continue
            if self.copied_too_much(index, component):
                found.append((SENTENCE_COPY_CODE, index))
            if self.mode == "explain":
                if not explained(component, self.slot_grams[index]):
                    found.append((EXPLANATION_MISSING_CODE, index))
                if length_report(component).too_long:
                    found.append((SENTENCE_LENGTH_CODE, index))
                if self.new_fact_locations(component):
                    found.append((NEW_FACT_CODE, index))
        return sorted(found, key=lambda item: item[1])

    def from_facts(self, index: int, text: str) -> bool:
        """Every judged sentence of ``text`` repeats the slot's facts: an example the source gives."""

        judged = [words for words in (words_of(sentence) for sentence in sentences_of(text))
                  if len(words) >= IDM_SENTENCE_MIN_WORDS]
        return bool(judged) and all(self.slot_grams[index].near_verbatim(words) for words in judged)

    def label_examples(self, index: int, component: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Explain mode: "Ví dụ:" / "Example:" before an example the facts do not give gets the illustrative label,
        so a learner never takes a drafted example for a fact. A source example restated in other words is labelled
        too: a needless label costs the author a look, a missing one misleads the learner. Returns the component and
        how many were labelled."""

        if self.mode != "explain" or index not in self.teaching:
            return component, 0
        label = example_label(self.locale)
        labelled = 0

        def relabel(text: Any) -> Any:
            nonlocal labelled
            if not isinstance(text, str) or is_labelled_example(text):
                return text
            match = _UNLABELLED_EXAMPLE_RE.match(text)
            rest = text[match.end():] if match else ""
            if match is None or not rest.strip() or self.from_facts(index, rest):
                return text
            labelled += 1
            return f"{label} {rest}"

        sections = []
        for section in _sections(component):
            blocks = []
            for block in section.get("blocks") or []:
                if isinstance(block, dict) and block.get("kind") in _TEXT_KINDS:
                    blocks.append({**block, "text": relabel(block.get("text"))})
                elif isinstance(block, dict) and block.get("kind") in _LIST_KINDS and isinstance(block.get("items"),
                                                                                                   list):
                    blocks.append({**block, "items": [relabel(item) for item in block["items"]]})
                else:
                    blocks.append(block)
            sections.append({**section, "blocks": blocks})
        if not labelled:
            return component, 0
        return {**component, "semantic_content": {**component["semantic_content"], "sections": sections}}, labelled

    def drafted_examples(self, components: Sequence[Any]) -> dict[int, int]:
        """Labelled (AI-drafted) examples per teaching slot, for the author note."""

        counts: dict[int, int] = {}
        for index in sorted(self.teaching):
            component = components[index] if index < len(components) else None
            if isinstance(component, dict) and component.get("type") == "html":
                count = sum(is_labelled_example(unit.text) for unit in text_units(component))
                if count:
                    counts[index] = count
        return counts

    def metrics(self, components: Sequence[Any]) -> dict[str, Any]:
        """Numbers only, for the unit log: near-verbatim and judged sentences and average words per sentence."""

        slots: dict[str, dict[str, float]] = {}
        for index in sorted(self.teaching):
            component = components[index] if index < len(components) else None
            if isinstance(component, dict) and component.get("type") == "html":
                report = self.report(index, component)
                lengths = length_report(component)
                slots[f"c{index}"] = {"judged": report.judged, "copied": len(report.copied),
                                      "avg_words": round(report.words / report.sentences, 1) if report.sentences
                                      else 0.0,
                                      "prose_avg_words": round(lengths.average, 1), "prose_max_words": lengths.longest}
        return {"mode": self.mode, "slots": slots, "examples": sum(self.drafted_examples(components).values()),
                "example_numbers": sum(self.example_numbers(components).values())}


__all__ = [
    "EXAMPLE_ROLE_CODE", "EXPLANATION_MISSING_CODE", "FIDELITY_CODES", "NEW_FACT_CODE", "SENTENCE_COPY_CODE",
    "SENTENCE_LENGTH_CODE", "CopyReport", "EvidenceGrams", "FidelityChecker", "KnownFacts", "LengthReport",
    "NewFactHit", "TextUnit", "carries_exact_claim", "copy_report", "example_segments", "explained", "explains",
    "is_labelled_example", "label_heads", "length_report", "list_counts", "masked_count", "masked_labels",
    "sentences_of", "text_units", "words_of",
]
