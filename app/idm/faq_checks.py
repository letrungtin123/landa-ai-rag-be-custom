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
* ``IDM_W5_FAQ_OPEN_QUESTION`` (QLT-7): a question that asks an open SME question of the unit again (c1.l3.u1 of QC
  run fae66524 answered "… triển khai đồng loạt toàn bộ 9 công cụ … theo đúng thứ tự PDCA không?", which the SME had
  not settled); the writer gets those questions as OPEN_QUESTIONS, never to be answered;
* ``IDM_W5_FAQ_INVENTED_REASON`` (QLT-7): a reason for a number, percentage, deadline or threshold that its basis facts
  do not explain (c4.l3.u1: "Vì sao … ngưỡng … tối thiểu 30% …?" answered with a reason the source never gives);
* inferred items (never a finding): an answer that explains beyond the words of its basis facts, or (QLT-7) that has
  a reasoning sentence ("vì", "nhằm", "giúp" …) no basis fact holds. The author note asks the SME to check it; it is
  never repaired back into a restatement.

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
from app.idm.org_grounding import WithheldTokens
from app.idm.policy import (
    FAQ_COMPONENT_TYPE,
    FAQ_DEEP_TYPES,
    IDM_FAQ_MAX_RECALL_ITEMS,
    IDM_FAQ_OPEN_QUESTION_MIN_SHARE,
    IDM_FAQ_OPEN_QUESTION_MIN_SHARED_WORDS,
    IDM_FAQ_REASON_SENTENCE_MIN_SUPPORT,
    IDM_FAQ_SOURCE_COPY_LCS_SHARE,
    IDM_FAQ_SOURCE_COPY_MIN_WORDS,
    IDM_FAQ_SOURCE_COPY_NGRAM,
    IDM_FAQ_SOURCE_COPY_NGRAM_SHARE,
)
from app.idm.text import EvidenceIndex, content_words, evidence_index, grounding_verdict, idm_fold, number_tokens

FAQ_RESTATES_SOURCE_CODE: Final = "IDM_W5_FAQ_RESTATES_SOURCE"
FAQ_TYPE_MIX_CODE: Final = "IDM_W5_FAQ_TYPE_MIX"
FAQ_NEW_FACT_CODE: Final = "IDM_W5_FAQ_NEW_FACT"
FAQ_OPEN_QUESTION_CODE: Final = "IDM_W5_FAQ_OPEN_QUESTION"
FAQ_INVENTED_REASON_CODE: Final = "IDM_W5_FAQ_INVENTED_REASON"
FAQ_VALUE_CODES: Final = frozenset({FAQ_RESTATES_SOURCE_CODE, FAQ_TYPE_MIX_CODE, FAQ_NEW_FACT_CODE,
                                    FAQ_OPEN_QUESTION_CODE, FAQ_INVENTED_REASON_CODE})
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


# --- open SME questions and invented reasons (QLT-7) -----------------------------------------------------------------
def asks_open_question(question: str, open_words: Sequence[frozenset[str]]) -> bool:
    """The FAQ question asks one of the open SME questions again: at least ``IDM_FAQ_OPEN_QUESTION_MIN_SHARED_WORDS``
    of its content words, and ``IDM_FAQ_OPEN_QUESTION_MIN_SHARE`` of them, are in that question."""

    words = set(content_words(question))
    return bool(words) and any(
        len(shared := words & other) >= IDM_FAQ_OPEN_QUESTION_MIN_SHARED_WORDS
        and len(shared) >= IDM_FAQ_OPEN_QUESTION_MIN_SHARE * len(words) for other in open_words)


def _nfc_lower(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(text.split())).lower()


_NUMBER_RE: Final = re.compile(r"\d+(?:[.,]\d+)*")
# A number is a threshold, a deadline or an amount when a unit follows it ("30%", "90 ngày", "50 triệu") or a threshold
# word comes just before it ("tối thiểu 30", "dưới ngưỡng 5"); "6 bậc", "8 cam kết" or "Bậc 02" count items or name a
# level and are not.
_UNIT_AFTER_RE: Final = re.compile(
    r"^\s*(?:%|phần trăm|ngày|tuần|tháng|năm|giờ|phút|lần|triệu|tỷ|nghìn|ngàn|đồng|usd|vnd|percent|days?|weeks?|months?"
    r"|years?|hours?|minutes?|times|million|billion)(?!\w)")
_THRESHOLD_BEFORE_RE: Final = re.compile(
    r"(?:ngưỡng|tối thiểu|tối đa|ít nhất|nhiều nhất|không quá|dưới|trên|vượt|mức|threshold|minimum|maximum|at least"
    r"|at most|no more than|under|over|below|above|limit)\s*(?:là\s*)?$")
_BEFORE_CHARS: Final = 24
# Words that give a reason in a fact ("… vì …", "… nhằm …", "… để …"); "giúp" only says what something does, and
# "thay vì" means "instead of".
_FACT_REASON_RE: Final = re.compile(
    r"(?<!\w)(?:(?<!thay )vì|bởi|lý do|nhằm|để|do đó|because|since|so that|in order to|due to|the reason)(?!\w)")
# Words that give a reason in an answer sentence about a number ("Ngưỡng 30% đảm bảo …", "… vì …").
_ANSWER_REASON_RE: Final = re.compile(
    r"(?<!\w)(?:(?<!thay )vì|bởi|lý do|nhằm|đảm bảo|because|the reason|due to|ensures?|so that)(?!\w)")
# Reasoning connectives that mark an answer sentence as inference unless one basis fact holds most of its words.
_REASONING_RE: Final = re.compile(
    r"(?<!\w)(?:(?<!thay )vì|bởi|nhằm|nên|giúp|để|do đó|vì vậy|nhờ|đảm bảo|because|so that|therefore|helps?|in order to"
    r"|ensures?)(?!\w)")
_SENTENCE_RE: Final = re.compile(r"(?<=[.!?;])\s+")


def threshold_numbers(text: str) -> set[str]:
    """The numbers of ``text`` that state a threshold, a deadline or an amount (normalised like ``number_tokens``)."""

    lowered = _nfc_lower(text)
    found: set[str] = set()
    for match in _NUMBER_RE.finditer(lowered):
        before = lowered[max(0, match.start() - _BEFORE_CHARS):match.start()]
        if _UNIT_AFTER_RE.match(lowered[match.end():]) or _THRESHOLD_BEFORE_RE.search(before.rstrip()):
            found |= number_tokens(match.group(0))
    return found


def reasoned(numbers: set[str], facts: Sequence[str]) -> bool:
    """Every number of ``numbers`` sits in a fact sentence that gives a reason ("… vì …", "… nhằm …")."""

    sentences = [sentence for fact in facts for sentence in _SENTENCE_RE.split(_nfc_lower(fact))]
    return all(any(number in number_tokens(sentence) and _FACT_REASON_RE.search(sentence) for sentence in sentences)
               for number in numbers)


def invents_reason(question: str, answer: str, facts: Sequence[str]) -> bool:
    """The item gives a reason for a threshold its facts do not explain: a why-question about a threshold, or an
    answer sentence that holds a threshold and a reason word, while no fact sentence holding that number says why."""

    numbers: set[str] = set()
    if question_type(question) == "why":
        numbers |= threshold_numbers(question)
    for sentence in _SENTENCE_RE.split(_nfc_lower(answer)):
        if _ANSWER_REASON_RE.search(sentence):
            numbers |= threshold_numbers(sentence)
    return bool(numbers) and not reasoned(numbers, facts)


def unsupported_reasoning(answer: str, facts: Sequence[str]) -> bool:
    """An answer sentence with a reasoning connective whose content words no single basis fact mostly holds."""

    fact_words = [set(content_words(fact)) for fact in facts]
    for sentence in _SENTENCE_RE.split(" ".join(answer.split())):
        if not _REASONING_RE.search(_nfc_lower(sentence)):
            continue
        words = set(content_words(sentence))
        if words and max((len(words & other) for other in fact_words), default=0) < (
                IDM_FAQ_REASON_SENTENCE_MIN_SUPPORT * len(words)):
            return True
    return False


def open_questions(brief: IdmUnitBriefV1) -> tuple[str, ...]:
    """The SME questions still open for the unit: none since the owner decision of 2026-10-22 (the AI writes no
    question for an SME). The questions a brief built from an older design may still carry (its blocks' W1 questions,
    ``faq_material.open_questions``) are ignored, so they never reach the writer, the checks or the note."""

    del brief
    return ()


def faq_prompt_material(brief: IdmUnitBriefV1, *, has_faq: bool) -> dict[str, Any]:
    """The FAQ_MATERIAL of the writer prompt: only for a unit with an la_faq slot, and without any SME question (owner
    decision 2026-10-22: none reaches the writer, not even from a brief built from an older design)."""

    material = brief.faq_material
    if material is None or not has_faq:
        return {}
    data = material.model_dump(mode="json")
    data.pop("open_questions", None)
    data["blocks"] = [{key: value for key, value in block.items() if key != "sme_questions"}
                      for block in data["blocks"]]
    return {"faq_material": data}


# --- the checker ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Basis:
    known: KnownFacts
    evidence: EvidenceIndex
    texts: tuple[str, ...] = ()


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
    # QLT-7: the open SME questions of the unit (its blocks' W1 questions and the lesson's Hold questions), never to be
    # answered by the FAQ, and their content words.
    open_questions: tuple[str, ...] = ()
    open_words: tuple[frozenset[str], ...] = ()
    # The organization profile's details the facts do not state: never known through the plan (QC 21/10).
    withheld: WithheldTokens | None = None

    @classmethod
    def build(cls, brief: IdmUnitBriefV1, fact_text: Mapping[str, str], *, plan_texts: Iterable[str],
              withheld: WithheldTokens | None = None) -> FaqChecker:
        """``fact_text``: every fact the writer was given (the unit's and the lesson context); ``plan_texts``: the
        approved plan, whose numbers and names an answer may repeat, except the ``withheld`` details of the
        organization profile (``app.idm.org_grounding.withheld_details``)."""

        plans: dict[int, tuple[IdmFaqItemPlanV1, ...]] = {}
        slot_keys: dict[int, tuple[str, ...]] = {}
        for index, slot in enumerate(brief.components):
            if slot.type == FAQ_COMPONENT_TYPE:
                plans[index] = tuple(slot.faq_items)
                slot_keys[index] = tuple(key for key in dict.fromkeys([*slot.owned_fact_keys,
                                                                       *slot.supporting_fact_keys])
                                         if key in fact_text)
        questions = open_questions(brief)
        return cls(brief_mode(brief), dict(fact_text), tuple(_Fact.of(text) for text in fact_text.values()),
                   tuple(plan_texts), plans, slot_keys, open_questions=questions,
                   open_words=tuple(frozenset(content_words(question)) for question in questions), withheld=withheld)

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
            cached = _Basis(KnownFacts.of(texts, plan_texts=self.plan_texts, withheld=self.withheld),
                            evidence_index(texts), tuple(texts))
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

    def new_fact_tokens(self, index: int, component: Mapping[str, Any]) -> dict[int, tuple[str, ...]]:
        """Item -> the numbers and names its basis facts and the plan do not state (owner P2, 2026-10-20: the repair
        prompt names them so the writer can remove exactly those; never logged)."""

        found: dict[int, tuple[str, ...]] = {}
        for position, (question, answer) in enumerate(zip(_questions(component), _answers(component), strict=True)):
            numbers, names = self.basis(index, position).known.new_tokens(f"{question}\n{answer}")
            if numbers or names:
                found[position] = names + numbers
        return found

    def slot_texts(self, index: int) -> list[str]:
        """The facts slot ``index`` is built on: every planned anchor and the slot's own facts (the title check)."""

        keys = [*(key for item in self.plans.get(index, ()) for key in item.anchor_fact_keys),
                *self.slot_keys.get(index, ())]
        return [self.fact_text[key] for key in dict.fromkeys(keys) if key in self.fact_text]

    def open_items(self, component: Mapping[str, Any]) -> list[int]:
        """Items whose question asks an open SME question of the unit again (QLT-7)."""

        if not self.open_words:
            return []
        return [position for position, question in enumerate(_questions(component))
                if asks_open_question(question, self.open_words)]

    def invented_reasons(self, index: int, component: Mapping[str, Any]) -> list[int]:
        """Items that give a reason for a threshold their basis facts do not explain (QLT-7)."""

        return [position for position, (question, answer) in enumerate(zip(_questions(component),
                                                                           _answers(component), strict=True))
                if invents_reason(question, answer, self.basis(index, position).texts)]

    def inferred(self, index: int, component: Mapping[str, Any]) -> list[int]:
        """Items whose answer explains beyond the words of its basis facts, or has a reasoning sentence no basis fact
        holds (QLT-7: the whole-answer measure alone marked 3 of 25 answers of run fae66524). For the SME, never a
        finding."""

        found = []
        for position, answer in enumerate(_answers(component)):
            basis = self.basis(index, position)
            if answer.strip() and (grounding_verdict(answer, basis.evidence) != "grounded"
                                   or unsupported_reasoning(answer, basis.texts)):
                found.append(position)
        return found

    def findings(self, index: int, component: Mapping[str, Any]) -> list[str]:
        codes: list[str] = []
        if not type_mix_ok(question_types(component)):
            codes.append(FAQ_TYPE_MIX_CODE)
        if self.restated(component):
            codes.append(FAQ_RESTATES_SOURCE_CODE)
        if self.with_new_facts(index, component):
            codes.append(FAQ_NEW_FACT_CODE)
        if self.open_items(component):
            codes.append(FAQ_OPEN_QUESTION_CODE)
        if self.invented_reasons(index, component):
            codes.append(FAQ_INVENTED_REASON_CODE)
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
        "  * Each answer: 2-3 sentences (at most about 60 words) in your own words that explain (the reason, the "
        "consequence, what to do instead, how two cases differ). Never restate a fact sentence, a table row, a list "
        "item or a paragraph of the html.\n"
        "  * Inference yes, new facts no: never a number, amount, date, threshold, deadline, rule, exception, person, "
        "company or product name that the item's anchor facts do not state. When the facts cannot answer a question, "
        "ask another one.\n"
        "  * Never give a reason, cause or purpose for a number, percentage, amount, deadline or threshold that its "
        'anchor facts do not explain (no "Vì sao lại là 30% …?" answered with a reason the facts never give); ask '
        "what the facts do say about it, or another question.\n"
        "  * FAQ_MATERIAL in UNIT_BRIEF (when present) lists what the source analysis found for this unit's blocks "
        "(their role, unclear points, gaps) and the feedback focus of the lesson practices: use it to pick real "
        "confusions. A gap means the source is silent: never fill it with an invented rule; a point the source states "
        "in two different ways is never settled by an answer.\n"
        "  * When this unit has a problem slot, base the common_mistake items on its own distractors: the belief "
        "behind a wrong option and why the criterion rules it out, consistent with the answer key. An answer never "
        "places a case in another level, group or label than the unit's question and table do, and never states a "
        "classification rule the facts do not.\n"
        + keep +
        "- The la_faq title names what its questions are about.\n")


__all__ = [
    "FAQ_INVENTED_REASON_CODE", "FAQ_NEW_FACT_CODE", "FAQ_OPEN_QUESTION_CODE", "FAQ_RESTATES_SOURCE_CODE",
    "FAQ_TYPE_MIX_CODE", "FAQ_VALUE_CODES", "RECALL", "FaqChecker", "asks_open_question", "copies_one_fact",
    "extra_recall_items", "faq_prompt_material", "faq_writer_rules", "invents_reason", "lcs_length", "open_questions",
    "question_type", "question_types", "reasoned", "threshold_numbers", "type_mix_ok", "unsupported_reasoning",
]
