"""FAQ that digs deeper, W5 side (QLT-5 D): the writer rules and the deterministic checks of an la_faq slot.

Measured on 100 FAQ items of 6 runs (plan 2026-10-13 §2.3): 58% recall or yes/no fact checks, 5% what-if, about half of
each answer's word pairs straight from the source, 23% of the answers a near copy of one source sentence. The old W5
rule made it so ("answers restate only what the facts say"), and the only check compared an FAQ with the html above
it. The checks here, cheap and without a provider call:

* question type (``question_type``): a keyword classifier for Vietnamese and English questions; text written without
  diacritics is matched accent-folded;
* ``IDM_W5_FAQ_TYPE_MIX``: more than one plain recall question, or none that asks why, what if or what goes wrong;
* ``IDM_W5_FAQ_RESTATES_SOURCE``: an answer ONE fact holds almost word for word (the longest common word subsequence
  covers >= 0.6 of the answer, or >= 0.6 of its 5-grams are in that fact); not measured in the keep content mode, where
  an answer may quote its rule;
* ``IDM_W5_FAQ_NEW_FACT``: a number or a name that the item's basis facts (its planned anchor facts and the slot's own
  facts) and the approved plan do not state; a whole-lesson vocabulary no longer grounds a stray number;
* inferred items (never a finding): an answer that explains beyond the words of its basis facts. The author note asks
  the SME to check it; it is never repaired back into a restatement.

A finding gets one targeted repair, then the items concerned are dropped while two remain, else the slot is kept for
author review; the source-locked FAQ (verbatim source sentences) is never used for these findings (``storyboard``).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from app.idm.contracts import FaqQuestionType, FidelityMode, IdmFaqItemPlanV1, IdmUnitBriefV1
from app.idm.fidelity import brief_mode
from app.idm.fidelity_checks import KnownFacts, words_of
from app.idm.policy import (
    FAQ_COMPONENT_TYPE,
    FAQ_DEEP_TYPES,
    IDM_FAQ_MAX_RECALL_ITEMS,
    IDM_FAQ_SOURCE_COPY_LCS_SHARE,
    IDM_FAQ_SOURCE_COPY_MIN_WORDS,
    IDM_FAQ_SOURCE_COPY_NGRAM,
    IDM_FAQ_SOURCE_COPY_NGRAM_SHARE,
)
from app.idm.text import EvidenceIndex, evidence_index, grounding_verdict, idm_fold

FAQ_RESTATES_SOURCE_CODE: Final = "IDM_W5_FAQ_RESTATES_SOURCE"
FAQ_TYPE_MIX_CODE: Final = "IDM_W5_FAQ_TYPE_MIX"
FAQ_NEW_FACT_CODE: Final = "IDM_W5_FAQ_NEW_FACT"
FAQ_VALUE_CODES: Final = frozenset({FAQ_RESTATES_SOURCE_CODE, FAQ_TYPE_MIX_CODE, FAQ_NEW_FACT_CODE})
RECALL: Final[FaqQuestionType] = "recall"

# --- question types -------------------------------------------------------------------------------------------------
# Checked in this order: a question about a mistake that also asks "why" is a mistake question; a question that asks
# for an amount or a list of cases is recall even when it says "how" or "case". Vietnamese phrases keep
# their diacritics ("nhầm" is a mistake, "nhằm" an aim; "nếu" is if, "nêu" to state); a question written without any
# diacritic is matched against their folded form.
_TYPE_PHRASES: Final[tuple[tuple[FaqQuestionType, tuple[str, ...]], ...]] = (
    ("common_mistake", (
        "nhầm", "nhầm lẫn", "dễ nhầm", "hay nhầm", "hiểu sai", "hiểu lầm", "hiểu nhầm", "sai lầm", "mắc lỗi",
        "lỗi thường gặp", "lỗi hay gặp", "lỗi nào", "lỗi phổ biến", "hay gặp", "thường gặp", "ngộ nhận", "cần tránh",
        "nên tránh", "làm sai", "sai ở đâu", "mistake", "mistakes", "misconception", "misunderstand",
        "misunderstanding", "confuse", "confused", "confusion", "pitfall", "go wrong", "goes wrong", "wrong", "error",
        "errors", "avoid")),
    # Asking for an amount, a list of cases or a definition stays recall, even with "how", "case" or "thế nào".
    ("recall", (
        "bao nhiêu", "bao lâu", "thế nào là", "trường hợp nào", "how many", "how much", "how long", "how often",
        "which cases", "in which cases")),
    ("what_if", (
        "nếu", "giả sử", "trường hợp", "điều gì xảy ra", "chuyện gì xảy ra", "thì sao", "what if", "what happens",
        "if", "suppose", "in case", "otherwise")),
    ("why", (
        "vì sao", "tại sao", "lý do", "để làm gì", "nhằm mục đích", "ý nghĩa", "sao lại", "why", "reason", "what for",
        "purpose of", "the point of")),
    ("comparison", (
        "khác nhau", "khác gì", "khác với", "khác biệt", "so với", "giống", "phân biệt", "difference", "different",
        "differ", "compare", "compared", "versus", "vs")),
    ("how_to_apply", (
        "làm thế nào", "làm sao", "như thế nào", "thế nào", "ra sao", "cách nào", "cách làm", "cách xử lý", "áp dụng",
        "nên làm gì", "cần làm gì", "phải làm gì", "how", "should", "apply", "what to do", "handle")),
)
# "Khi <case> thì …" / "When <case>, …" asks what to do in a case (not "khi nào", which asks for the rule itself).
_WHEN_CLAUSE_VI: Final = re.compile(r"(?<!\w)khi(?!\s+nào)(?!\w)[^?]*?(?<!\w)thì(?!\w)")
_WHEN_CLAUSE_FOLDED: Final = re.compile(r"(?<!\w)khi(?!\s+nao)(?!\w)[^?]*?(?<!\w)thi(?!\w)")
_WHEN_CLAUSE_EN: Final = re.compile(r"(?<!\w)when(?!\w)[^?,]+,")


def _alternation(phrases: Iterable[str]) -> re.Pattern[str]:
    ordered = sorted(dict.fromkeys(phrases), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(phrase) for phrase in ordered) + r")(?!\w)")


_ACCENTED_PATTERNS: Final = tuple((kind, _alternation(phrases)) for kind, phrases in _TYPE_PHRASES)
_FOLDED_PATTERNS: Final = tuple((kind, _alternation(idm_fold(phrase) for phrase in phrases))
                                for kind, phrases in _TYPE_PHRASES)


def _has_diacritics(text: str) -> bool:
    return any(unicodedata.category(char) == "Mn" for char in unicodedata.normalize("NFD", text)) or "đ" in text


def question_type(question: str) -> FaqQuestionType:
    """The type a learner question asks for; "recall" when it asks for nothing beyond a fact or a list."""

    lowered = unicodedata.normalize("NFC", " ".join(question.split())).lower()
    accented = _has_diacritics(lowered)
    text = lowered if accented else idm_fold(lowered)
    for kind, pattern in _ACCENTED_PATTERNS if accented else _FOLDED_PATTERNS:
        if pattern.search(text):
            return kind
        if kind == "what_if" and ((_WHEN_CLAUSE_VI if accented else _WHEN_CLAUSE_FOLDED).search(text)
                                  or _WHEN_CLAUSE_EN.search(text)):
            return kind
    return RECALL


def _questions(component: Mapping[str, Any]) -> list[str]:
    items = component.get("items")
    return [str(item.get("question") or "") if isinstance(item, dict) else "" for item in items] \
        if isinstance(items, list) else []


def _answers(component: Mapping[str, Any]) -> list[str]:
    items = component.get("items")
    return [str(item.get("answer") or "") if isinstance(item, dict) else "" for item in items] \
        if isinstance(items, list) else []


def question_types(component: Mapping[str, Any]) -> list[FaqQuestionType]:
    return [question_type(question) for question in _questions(component)]


def type_mix_ok(types: Sequence[FaqQuestionType]) -> bool:
    """At most one plain recall question and at least one why, what-if or common-mistake question."""

    return types.count(RECALL) <= IDM_FAQ_MAX_RECALL_ITEMS and any(kind in FAQ_DEEP_TYPES for kind in types)


def extra_recall_items(types: Sequence[FaqQuestionType]) -> list[int]:
    """Positions of the recall questions over the allowance (the first ones are kept)."""

    recalls = [position for position, kind in enumerate(types) if kind == RECALL]
    return recalls[IDM_FAQ_MAX_RECALL_ITEMS:]


# --- copy of one source fact ----------------------------------------------------------------------------------------
def _grams(words: Sequence[str]) -> list[tuple[str, ...]]:
    size = IDM_FAQ_SOURCE_COPY_NGRAM
    return [tuple(words[start:start + size]) for start in range(len(words) - size + 1)]


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Length of the longest common subsequence of two word lists (one row of the classic table at a time)."""

    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for word in right:
        current = [0]
        for position, other in enumerate(left, start=1):
            current.append(previous[position - 1] + 1 if word == other
                           else max(previous[position], current[position - 1]))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class _Fact:
    words: tuple[str, ...]
    counts: Counter[str]
    grams: frozenset[tuple[str, ...]]

    @classmethod
    def of(cls, text: str) -> _Fact:
        words = words_of(text)
        return cls(tuple(words), Counter(words), frozenset(_grams(words)))


def copies_one_fact(answer: str, facts: Sequence[_Fact]) -> bool:
    """One fact holds the answer almost word for word: its longest common subsequence with the answer covers
    ``IDM_FAQ_SOURCE_COPY_LCS_SHARE`` of the answer's words, or ``IDM_FAQ_SOURCE_COPY_NGRAM_SHARE`` of the answer's
    5-grams are in that fact. Answers under ``IDM_FAQ_SOURCE_COPY_MIN_WORDS`` words are not judged."""

    words = words_of(answer)
    if len(words) < IDM_FAQ_SOURCE_COPY_MIN_WORDS:
        return False
    grams = _grams(words)
    counts = Counter(words)
    need = math.ceil(IDM_FAQ_SOURCE_COPY_LCS_SHARE * len(words))
    for fact in facts:
        if grams and sum(gram in fact.grams for gram in grams) >= IDM_FAQ_SOURCE_COPY_NGRAM_SHARE * len(grams):
            return True
        # The shared words bound the subsequence from above: most facts are ruled out without the table.
        if sum(min(count, fact.counts[word]) for word, count in counts.items()) < need:
            continue
        if lcs_length(words, fact.words) >= need:
            return True
    return False


# --- the checker ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Basis:
    known: KnownFacts
    evidence: EvidenceIndex


@dataclass
class FaqChecker:
    """The FAQ plan and facts of one unit and what its checks compare against (built once per unit)."""

    mode: FidelityMode
    fact_text: Mapping[str, str]
    facts: tuple[_Fact, ...]
    plan_texts: tuple[str, ...]
    plans: Mapping[int, tuple[IdmFaqItemPlanV1, ...]]
    slot_keys: Mapping[int, tuple[str, ...]]
    _bases: dict[tuple[int, int], _Basis] = field(default_factory=dict)

    @classmethod
    def build(cls, brief: IdmUnitBriefV1, fact_text: Mapping[str, str], *, plan_texts: Iterable[str]) -> FaqChecker:
        """``fact_text``: every fact the writer was given (the unit's and the lesson context); ``plan_texts``: the
        approved plan, whose numbers and names an answer may repeat."""

        plans: dict[int, tuple[IdmFaqItemPlanV1, ...]] = {}
        slot_keys: dict[int, tuple[str, ...]] = {}
        for index, slot in enumerate(brief.components):
            if slot.type == FAQ_COMPONENT_TYPE:
                plans[index] = tuple(slot.faq_items)
                slot_keys[index] = tuple(key for key in dict.fromkeys([*slot.owned_fact_keys,
                                                                       *slot.supporting_fact_keys])
                                         if key in fact_text)
        return cls(brief_mode(brief), dict(fact_text), tuple(_Fact.of(text) for text in fact_text.values()),
                   tuple(plan_texts), plans, slot_keys)

    def basis_keys(self, index: int, position: int) -> list[str]:
        """The facts item ``position`` of slot ``index`` is built on: its planned anchors (all anchors for an item
        beyond the plan) and the slot's own facts; every fact of the unit when the slot has neither."""

        plan = self.plans.get(index, ())
        anchors = (plan[position].anchor_fact_keys if position < len(plan)
                   else [key for item in plan for key in item.anchor_fact_keys])
        keys = [key for key in dict.fromkeys([*anchors, *self.slot_keys.get(index, ())]) if key in self.fact_text]
        return keys or list(self.fact_text)

    def basis(self, index: int, position: int) -> _Basis:
        cached = self._bases.get((index, position))
        if cached is None:
            texts = [self.fact_text[key] for key in self.basis_keys(index, position)]
            cached = _Basis(KnownFacts.of(texts, plan_texts=self.plan_texts), evidence_index(texts))
            self._bases[(index, position)] = cached
        return cached

    def restated(self, component: Mapping[str, Any]) -> list[int]:
        """Items whose answer copies one source fact (none in the keep mode, where answers may quote their rule)."""

        if self.mode == "keep":
            return []
        return [position for position, answer in enumerate(_answers(component)) if copies_one_fact(answer, self.facts)]

    def with_new_facts(self, index: int, component: Mapping[str, Any]) -> list[int]:
        """Items whose question or answer states a number or a name their basis facts and the plan do not."""

        return [position for position, (question, answer) in enumerate(zip(_questions(component),
                                                                           _answers(component), strict=True))
                if self.basis(index, position).known.states_new_fact(f"{question}\n{answer}")]

    def inferred(self, index: int, component: Mapping[str, Any]) -> list[int]:
        """Items whose answer explains beyond the words of its basis facts (for the SME, never a finding)."""

        return [position for position, answer in enumerate(_answers(component))
                if answer.strip() and grounding_verdict(answer, self.basis(index, position).evidence) != "grounded"]

    def findings(self, index: int, component: Mapping[str, Any]) -> list[str]:
        codes: list[str] = []
        if not type_mix_ok(question_types(component)):
            codes.append(FAQ_TYPE_MIX_CODE)
        if self.restated(component):
            codes.append(FAQ_RESTATES_SOURCE_CODE)
        if self.with_new_facts(index, component):
            codes.append(FAQ_NEW_FACT_CODE)
        return codes

    def plan_hint(self, index: int) -> str:
        """The planned type and anchor facts of each item, for a repair line (server values, never text)."""

        plan = self.plans.get(index, ())
        parts = [f"items[{position}] {item.question_type} on {', '.join(f'[{key}]' for key in item.anchor_fact_keys)}"
                 for position, item in enumerate(plan)]
        return f" Plan: {'; '.join(parts)}." if parts else ""

    def inferred_counts(self, components: Sequence[Any], exempt: Iterable[int] = ()) -> dict[int, int]:
        skip = set(exempt)
        counts: dict[int, int] = {}
        for index in sorted(self.plans):
            component = components[index] if index < len(components) else None
            if index in skip or not isinstance(component, dict) or component.get("type") != FAQ_COMPONENT_TYPE:
                continue
            count = len(self.inferred(index, component))
            if count:
                counts[index] = count
        return counts

    def metrics(self, components: Sequence[Any]) -> dict[str, Any]:
        """Numbers and server enums only, for the unit log: planned and written types, inferred items."""

        slots: dict[str, dict[str, Any]] = {}
        for index in sorted(self.plans):
            component = components[index] if index < len(components) else None
            if isinstance(component, dict) and component.get("type") == FAQ_COMPONENT_TYPE:
                slots[f"c{index}"] = {"planned": [item.question_type for item in self.plans[index]],
                                      "written": question_types(component),
                                      "inferred": len(self.inferred(index, component))}
        return slots


# --- writer rules ---------------------------------------------------------------------------------------------------
def faq_writer_rules(mode: FidelityMode) -> str:
    """The W5 rules of an la_faq slot (server-owned text; replaces "answers restate only what the facts say")."""

    keep = ("  * CONTENT MODE keep: an answer may quote its rule word for word, but its question still asks why, what "
            "if or what goes wrong, and the answer adds the reason or the consequence the facts give.\n"
            if mode == "keep" else "")
    return (
        "- An la_faq slot digs deeper than the html: each item answers a question a learner really asks after the "
        "lesson and EXPLAINS the answer from the facts (grounded inference): based on its facts, why it is so, what "
        "happens if ..., what to do when ..., or what goes wrong when .... Rules:\n"
        "  * Follow the slot's faq_items in order: item i answers faq_items[i] with its question_type and is built on "
        'its anchor_fact_keys. origin "source": a mistake, condition or exception the facts state; "distractor": the '
        "wrong belief behind a wrong option of this unit's question (or of a lesson practice in FAQ_MATERIAL) and why "
        'the criterion rules it out; "ai_drafted": a confusion a learner is likely to have, inferred from the facts. '
        "Write 2-4 items, at least one per faq_items entry.\n"
        "  * Question types: why | what_if | common_mistake | how_to_apply | comparison. At most ONE plain recall "
        "question (what is, which ones, how many, does it include); at least one why, what_if or common_mistake "
        "question.\n"
        "  * Each answer: 2-4 sentences in your own words that explain (the reason, the consequence, what to do "
        "instead, how two cases differ). Never restate a fact sentence, a table row, a list item or a paragraph of "
        "the html.\n"
        "  * Inference yes, new facts no: never a number, amount, date, threshold, deadline, rule, exception, person, "
        "company or product name that the item's anchor facts do not state. When the facts cannot answer a question, "
        "ask another one.\n"
        "  * FAQ_MATERIAL in UNIT_BRIEF (when present) lists what the source analysis found for this unit's blocks "
        "(their role, unclear points, gaps, open SME questions) and the feedback focus of the lesson practices: use "
        "it to pick real confusions. A gap means the source is silent: never fill it with an invented rule. An SME "
        "question is still open: never answer it as if it were settled.\n"
        "  * When this unit has a problem slot, base the common_mistake items on its own distractors: the belief "
        "behind a wrong option and why the criterion rules it out, consistent with the answer key.\n"
        + keep +
        "- The la_faq title names what its questions are about.\n")


__all__ = [
    "FAQ_NEW_FACT_CODE", "FAQ_RESTATES_SOURCE_CODE", "FAQ_TYPE_MIX_CODE", "FAQ_VALUE_CODES", "RECALL", "FaqChecker",
    "copies_one_fact", "extra_recall_items", "faq_writer_rules", "lcs_length", "question_type", "question_types",
    "type_mix_ok",
]
