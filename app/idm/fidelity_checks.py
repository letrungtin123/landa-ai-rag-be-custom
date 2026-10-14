"""Deterministic content-mode checks of W5 teaching html slots (QLT-5 C): cheap, no provider call.

Measured on real runs (plan 2026-10-13 §2.2): 14-28% of html sentences were near-verbatim copies of the source, no
unit said "nghĩa là" or "tức là", 6 of 89 units had an example or an explanation, and the slot-level 8-gram check
never fired (it needed >= 300 characters and a 0.7 share over the whole slot, and exempted keep rows). The checks
here work per sentence of a teaching slot (``fidelity.is_teaching_slot``):

* sentence copy (explain, structured): the share of near-verbatim sentences; sentences that carry a definition,
  rule, number or threshold are not judged, they stay exact by design;
* explanation (explain): at least one explanation device in a sentence the writer did not copy;
* new facts (explain): a number, date, amount or capitalised name neither the facts nor the approved plan state;
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
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Final

from app.idm.contracts import FidelityMode, IdmUnitBriefV1
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
    IDM_SENTENCE_MIN_WORDS,
)
from app.idm.text import EvidenceIndex, idm_fold, number_tokens

SENTENCE_COPY_CODE: Final = "IDM_W5_SENTENCE_COPY"
EXPLANATION_MISSING_CODE: Final = "IDM_W5_EXPLANATION_MISSING"
NEW_FACT_CODE: Final = "IDM_W5_NEW_FACT"
FIDELITY_CODES: Final = frozenset({SENTENCE_COPY_CODE, EXPLANATION_MISSING_CODE, NEW_FACT_CODE})

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


def explained(component: Mapping[str, Any], grams: EvidenceGrams) -> bool:
    """An explanation device ("nghĩa là", "vì", "Ví dụ minh hoạ", a why-question …) in a sentence or heading the
    writer did not copy from the facts."""

    texts = [*headings(component),
             *(sentence for unit in text_units(component) for sentence in sentences_of(unit.text))]
    return any(_DEVICE_RE.search(_nfc_lower(text)) and not grams.near_verbatim(words_of(text)) for text in texts)


def is_labelled_example(text: str) -> bool:
    return idm_fold(text).lstrip(" \"'(").startswith(_LABELLED_EXAMPLE_HEADS)


@dataclass
class FidelityChecker:
    """The content mode of one unit and what its checks compare against (built once per unit)."""

    mode: FidelityMode
    locale: str
    teaching: frozenset[int]
    # Teaching slots whose treatments are all "keep": exempt from the copy check in structured mode (as before).
    keep_only: frozenset[int]
    slot_grams: Mapping[int, EvidenceGrams]
    known_numbers: frozenset[str]
    known_words: frozenset[str]
    # Adjacent folded word pairs: a name of several words ("Hồ Chí Minh") is known only as a whole.
    known_pairs: frozenset[tuple[str, str]]

    @classmethod
    def build(cls, brief: IdmUnitBriefV1, fact_text: Mapping[str, str], evidence: EvidenceIndex, *,
              plan_texts: Iterable[str], locale: str) -> FidelityChecker:
        """``fact_text``: the unit's source facts; ``evidence``: everything the writer was given; ``plan_texts``:
        the approved plan (titles, purpose, support briefs), whose numbers and names may be repeated."""

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
        plan = list(plan_texts)
        numbers = set(evidence.numbers).union(*(number_tokens(text) for text in plan))
        words = set(evidence.words).union(*(words_of(text) for text in plan))
        pairs = set(evidence.pairs).union(*(set(pairwise(words_of(text))) for text in plan))
        return cls(brief_mode(brief), locale, teaching, keep_only, grams, frozenset(numbers), frozenset(words),
                   frozenset(pairs))

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

    def _known_name(self, run: Sequence[str]) -> bool:
        folded = [idm_fold(token) for token in run]
        if len(folded) == 1 and folded[0] in _COMMON_CAPITALISED:
            return True
        return all(word in self.known_words for word in folded) and all(
            pair in self.known_pairs for pair in pairwise(folded))

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

    def new_fact_locations(self, component: Mapping[str, Any]) -> list[str]:
        """Locations of texts with a number or a name the facts and the plan do not state (never the text)."""

        locations: list[str] = []
        for unit in text_units(component):
            text = _ORDINAL_RE.sub("", unit.text, count=1)
            if number_tokens(text) - self.known_numbers or self.new_names(text):
                locations.append(unit.location)
        return locations

    def findings(self, components: Sequence[Any]) -> list[tuple[str, int]]:
        """(code, slot) of every content-mode finding of the unit's teaching html slots."""

        if self.mode == "keep":
            return []
        found: list[tuple[str, int]] = []
        for index in sorted(self.teaching):
            component = components[index] if index < len(components) else None
            if not isinstance(component, dict) or component.get("type") != "html":
                continue
            if self.copied_too_much(index, component):
                found.append((SENTENCE_COPY_CODE, index))
            if self.mode == "explain":
                if not explained(component, self.slot_grams[index]):
                    found.append((EXPLANATION_MISSING_CODE, index))
                if self.new_fact_locations(component):
                    found.append((NEW_FACT_CODE, index))
        return found

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
                slots[f"c{index}"] = {"judged": report.judged, "copied": len(report.copied),
                                      "avg_words": round(report.words / report.sentences, 1) if report.sentences
                                      else 0.0}
        return {"mode": self.mode, "slots": slots, "examples": sum(self.drafted_examples(components).values())}


__all__ = [
    "EXPLANATION_MISSING_CODE", "FIDELITY_CODES", "NEW_FACT_CODE", "SENTENCE_COPY_CODE",
    "CopyReport", "EvidenceGrams", "FidelityChecker", "TextUnit", "carries_exact_claim", "copy_report", "explained",
    "is_labelled_example", "sentences_of", "text_units", "words_of",
]
