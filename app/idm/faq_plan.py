"""FAQ that digs deeper, W4 side (QLT-5 D): the evidence gate and the item plan of an la_faq slot.

Measured on 100 FAQ items of 6 real runs (plan 2026-10-13 §2.3): 58% recalled a list or checked a yes/no fact, 5% asked
"what if", and an FAQ sat in 39 of 71 units because the layout template put one there, not because the unit had
anything to clarify. After the W4 answer is accepted the server decides, per unit:

* gate: an la_faq stays only when its unit has something to clarify: a decision or policy block, a condition or
  exception the source states, a common-mistake block (strong signals), or a W1 issue or gap that marks a likely
  confusion (a weak signal: QLT-7 keeps at most one such FAQ per lesson). The wrong options of the unit's question feed
  the plan but are no reason on their own (QC run fae66524: an FAQ in 11 of 19 units, 3 of them on that alone; the QC
  target is 25-40%). Otherwise the slot is dropped, its blocks join the unit's teaching component and the lesson note
  says why in plain words (``IDM_W4_AUTOFIX_FAQ_DROPPED``);
* plan: a kept la_faq gets up to three items, each with a question type (why, what if, common mistake first; how to
  apply or comparison only to reach two items), the facts its answer is built on and the origin of its point (a
  mistake or condition the source states, the misconception behind a question's wrong option, or a confusion inferred
  from the facts). The W5 writer follows the plan and the W5 checks ground each answer in its anchor facts;
* floor (QLT-8, QC run 0a11aa9e: an FAQ in 2 of 23 units, the W4 designer had placed only 3): when fewer than
  ``IDM_FAQ_MIN_UNIT_SHARE`` of a module's learning units keep an FAQ, the server adds one to the units with the most to
  clarify (strong signal, then W1 issue or gap, then the misconceptions behind the wrong options of the unit's own
  question; a decision Must Do first), at most ``IDM_FAQ_FLOOR_PER_LESSON`` per lesson, lessons without an FAQ first.

Everything here is deterministic and server-owned; the W4 provider never writes the plan.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from app.content.instructional_quality import source_clarification_signals
from app.idm.contracts import (
    FaqItemOrigin,
    FaqQuestionType,
    IdmComponentDesignV1,
    IdmContentBlockV1,
    IdmFaqItemPlanV1,
    IdmLessonDesignV1,
    IdmUnitDesignV1,
)
from app.idm.module_layout import ModuleScope
from app.idm.policy import (
    FAQ_COMPONENT_TYPE,
    FAQ_DEEP_TYPES,
    IDM_COMPONENT_MAX_BLOCKS,
    IDM_CRITERIA_LABEL_MAX_WORDS,
    IDM_FAQ_ANCHOR_MAX_FACTS,
    IDM_FAQ_FLOOR_PER_LESSON,
    IDM_FAQ_MIN_ITEMS,
    IDM_FAQ_MIN_UNIT_SHARE,
    IDM_FAQ_PLAN_MAX_ITEMS,
    IDM_FAQ_WEAK_SIGNAL_UNITS_PER_LESSON,
)
from app.idm.signals import states_rule
from app.idm.text import sanitize_author_text
from app.lesson_author_orchestration_v2 import ArchitectureComponentAuthorReviewV2

FAQ_DROPPED_CODE: Final = "IDM_W4_AUTOFIX_FAQ_DROPPED"
FAQ_PLANNED_CODE: Final = "IDM_W4_FAQ_ITEMS_PLANNED"
FAQ_ADDED_CODE: Final = "IDM_W4_FAQ_FLOOR_ADDED"
FAQ_TYPE: Final = FAQ_COMPONENT_TYPE
_HTML: Final = "html"
_PROBLEM: Final = "problem"
_JOB_AID: Final = "job_aid"
_MAX_COMPONENTS: Final = 4
_TITLE_MAX_CHARS: Final = 180
# Floor ranks: a strong signal, a W1 issue or gap, the wrong options of the unit's own question.
_FLOOR_RANK: Final[dict[str, int]] = {"strong": 0, "weak": 1, "material": 2}
# The question types that clarify; an FAQ needs at least one of them.
DEEP_TYPES: Final = FAQ_DEEP_TYPES
# W1 findings that mark a likely confusion of the learner (not a quality note about the source such as a duplicate or
# an outdated screenshot, and not a conflict, which the SME settles before anyone can explain it).
_CONFUSION_ISSUES: Final = frozenset({"unclear", "too_general"})
_CONFUSION_GAPS: Final = frozenset({"missing_common_mistake", "missing_exception"})
_DECIDING_KINDS: Final = frozenset({"decision", "policy"})
_DOING_KINDS: Final = frozenset({"procedure", "skill", "system"})
_CONCEPT_KINDS: Final = frozenset({"concept", "mindset"})
_MAX_CONDITION_ITEMS: Final = 2
# Pass 1 takes one item of each clarifying type, in this order; ties go to the origin the source backs best.
_TYPE_ORDER: Final[tuple[FaqQuestionType, ...]] = ("common_mistake", "what_if", "why")
_ORIGIN_RANK: Final[dict[FaqItemOrigin, int]] = {"source": 0, "distractor": 1, "ai_drafted": 2}
# What a candidate says about the gate (QLT-7): "strong" opens it (a decision or policy block, a condition or exception,
# a common-mistake block), "weak" opens it in at most one unit per lesson (a W1 issue or gap), "material" never opens
# it alone (the wrong options of the unit's own question).
Signal = Literal["strong", "weak", "material"]


@dataclass(frozen=True)
class FaqCandidate:
    question_type: FaqQuestionType
    anchors: tuple[str, ...]
    origin: FaqItemOrigin
    signal: Signal = "strong"

    def plan(self) -> IdmFaqItemPlanV1:
        return IdmFaqItemPlanV1(question_type=self.question_type, anchor_fact_keys=list(self.anchors),
                                origin=self.origin)


def _anchors(keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(keys))[:IDM_FAQ_ANCHOR_MAX_FACTS]


def _statements(block: IdmContentBlockV1, scope: ModuleScope) -> list[str]:
    """The block's facts that state something (not a label or a heading), in source order."""

    keys = [key for key in block.fact_keys if key in scope.fact_text]
    statements = [key for key in keys if len(scope.fact_text[key].split()) > IDM_CRITERIA_LABEL_MAX_WORDS]
    return statements or keys


def _is_condition(text: str) -> bool:
    """A condition, exception or caution the source states ("Nếu …", "Trừ khi …", "Không được …", "Lưu ý: …")."""

    return bool(source_clarification_signals([text]))


def _unit_blocks(unit: IdmUnitDesignV1, scope: ModuleScope) -> list[IdmContentBlockV1]:
    return [scope.blocks[block_id] for block_id in unit.block_ids if block_id in scope.blocks]


def faq_evidence(unit: IdmUnitDesignV1, lesson: IdmLessonDesignV1, scope: ModuleScope) -> list[FaqCandidate]:
    """What the unit gives an FAQ to clarify; empty when nothing (the gate). Every candidate is a clarifying type."""

    found: list[FaqCandidate] = []
    practices = {practice.practice_id: practice for practice in lesson.practice_tasks if not practice.hold}
    for component in unit.components:
        if component.type != "problem":
            continue
        # The wrong options of a question carry the misconceptions; its criteria facts say why they are wrong.
        practice = practices.get(component.practice_id or "")
        keys = [key for key in practice.criteria_fact_keys if key in scope.fact_text] if practice else []
        keys = keys or [key for block_id in component.block_ids if block_id in scope.blocks
                        for key in _statements(scope.blocks[block_id], scope)]
        if keys:
            found.append(FaqCandidate("common_mistake", _anchors(keys), "distractor", "material"))
    for block in _unit_blocks(unit, scope):
        statements = _statements(block, scope)
        if not statements:
            continue
        if block.support_role == "common_mistake":
            found.append(FaqCandidate("common_mistake", _anchors(statements), "source"))
        conditions = [key for key in statements if _is_condition(scope.fact_text[key])]
        found.extend(FaqCandidate("what_if", (key,), "source") for key in conditions[:_MAX_CONDITION_ITEMS])
        if block.intent == "decide" or block.content_kind in _DECIDING_KINDS:
            rules = [key for key in statements if states_rule(scope.fact_text[key])]
            found.append(FaqCandidate("why", _anchors(rules or statements), "source"))
        if any(issue.type in _CONFUSION_ISSUES for issue in block.issues) or any(
                gap.type in _CONFUSION_GAPS for gap in block.gaps):
            # A confusion inferred from the facts the issue names (else the block's statements); never a new rule.
            confused = [key for issue in block.issues if issue.type in _CONFUSION_ISSUES for key in issue.fact_keys
                        if key in statements]
            found.append(FaqCandidate("common_mistake", _anchors(confused or statements), "ai_drafted", "weak"))
    return list(dict.fromkeys(found))


def _fillers(unit: IdmUnitDesignV1, scope: ModuleScope) -> list[FaqCandidate]:
    """Items that only complete a plan to two items: how to apply a step or skill, how two concepts differ."""

    blocks = _unit_blocks(unit, scope)
    fillers: list[FaqCandidate] = []
    concepts: list[str] = []
    for block in blocks:
        statements = _statements(block, scope)
        if not statements:
            continue
        if block.intent == "do" or block.content_kind in _DOING_KINDS:
            fillers.append(FaqCandidate("how_to_apply", _anchors(statements), "source"))
        if block.content_kind in _CONCEPT_KINDS:
            concepts.append(statements[0])
    if len(concepts) >= IDM_FAQ_MIN_ITEMS:
        fillers.append(FaqCandidate("comparison", _anchors(concepts), "source"))
    first = next((statements for block in blocks if (statements := _statements(block, scope))), [])
    if first:
        fillers.append(FaqCandidate("how_to_apply", _anchors(first), "ai_drafted"))
    return fillers


def plan_faq_items(evidence: Sequence[FaqCandidate], fillers: Sequence[FaqCandidate]) -> list[IdmFaqItemPlanV1]:
    """At most ``IDM_FAQ_PLAN_MAX_ITEMS`` items: one of each clarifying type first (the origin the source backs best),
    then the other evidence, then fillers until the FAQ has two items. Never two items on the same type and facts."""

    ranked = sorted(evidence, key=lambda item: _ORIGIN_RANK[item.origin])
    chosen: list[FaqCandidate] = []
    for question_type in _TYPE_ORDER:
        best = next((item for item in ranked if item.question_type == question_type), None)
        if best is not None:
            chosen.append(best)
    taken = {(item.question_type, item.anchors) for item in chosen}
    for limit, pool in ((IDM_FAQ_PLAN_MAX_ITEMS, ranked), (IDM_FAQ_MIN_ITEMS, fillers)):
        for item in pool:
            if len(chosen) >= limit:
                break
            if (item.question_type, item.anchors) not in taken:
                chosen.append(item)
                taken.add((item.question_type, item.anchors))
    return [item.plan() for item in chosen[:IDM_FAQ_PLAN_MAX_ITEMS]]


# --- dropping a slot without evidence -----------------------------------------------------------------------------
def _without_faq(unit: IdmUnitDesignV1, allowed: frozenset[str], title: str) -> IdmUnitDesignV1 | None:
    """``unit`` without its la_faq, its blocks kept by the teaching component (``None`` when that breaks a bound).

    A unit whose only component is the FAQ keeps its blocks as a teaching html instead (when html is allowed)."""

    components = list(unit.components)
    position = next(index for index, component in enumerate(components) if component.type == FAQ_TYPE)
    faq = components.pop(position)
    if not components:
        if _HTML not in allowed:
            return None
        html = faq.model_copy(update={"type": _HTML, "role": "explain", "title": title, "component_index": 1,
                                      "faq_items": []})
        return unit.model_copy(update={"components": [html]})
    used = {block_id for component in components for block_id in component.block_ids}
    missing = [block_id for block_id in faq.block_ids if block_id not in used]
    target = next((index for index, component in enumerate(components)
                   if component.type == _HTML and component.role != "practice"),
                  next((index for index, component in enumerate(components) if component.role != "practice"), 0))
    if missing:
        order = {block_id: index for index, block_id in enumerate(unit.block_ids)}
        merged = sorted([*components[target].block_ids, *missing], key=lambda block_id: order.get(block_id, len(order)))
        if len(merged) > IDM_COMPONENT_MAX_BLOCKS:
            return None
        components[target] = components[target].model_copy(update={"block_ids": merged})
    renumbered = [component.model_copy(update={"component_index": index})
                  for index, component in enumerate(components, start=1)]
    return unit.model_copy(update={"components": renumbered})


def _dropped_note(titles: Sequence[str], locale: str) -> str:
    named = ", ".join(f'"{title}"' for title in titles)
    if locale == "vi":
        return (f"Đã bỏ phần hỏi đáp của {named}: tài liệu chưa nêu quyết định, điều kiện, ngoại lệ hay lỗi hay gặp "
                "cần làm rõ (một câu hỏi luyện tập thôi là chưa đủ; khi chỉ có chỗ có thể gây nhầm, mỗi bài giữ tối đa "
                "một phần hỏi đáp), nên phần hỏi đáp sẽ chỉ nhắc lại bài học.")
    return (f"The FAQ of {named} was left out: the source states no decision, condition, exception or common mistake "
            "to clarify (a practice question alone is not enough; where only a likely confusion exists, a lesson keeps "
            "at most one FAQ), so it would only repeat the lesson.")


@dataclass(frozen=True)
class FaqPlanResult:
    lessons: list[IdmLessonDesignV1]
    dropped: int
    planned: int
    # QLT-8: FAQs the floor added.
    added: int = 0


def _with_plan(component: IdmComponentDesignV1, items: list[IdmFaqItemPlanV1]) -> IdmComponentDesignV1:
    return component.model_copy(update={"faq_items": items})


def gate_signal(evidence: Sequence[FaqCandidate]) -> Signal | None:
    """The strongest signal of a unit's evidence that can open the gate: "strong", "weak" or None (QLT-7)."""

    signals = {item.signal for item in evidence}
    return "strong" if "strong" in signals else "weak" if "weak" in signals else None


def gate_and_plan_faq(lessons: Sequence[IdmLessonDesignV1], scope: ModuleScope) -> FaqPlanResult:
    """Drop every la_faq whose unit has nothing to clarify (QLT-7: a practice question alone is not enough, and at
    most ``IDM_FAQ_WEAK_SIGNAL_UNITS_PER_LESSON`` FAQs per lesson stand on a W1 issue or gap alone) and plan the items
    of the others."""

    allowed = scope.allowed_types
    result: list[IdmLessonDesignV1] = []
    dropped = planned = 0
    for lesson in lessons:
        units: list[IdmUnitDesignV1] = []
        left_out: list[str] = []
        weak_left = IDM_FAQ_WEAK_SIGNAL_UNITS_PER_LESSON
        for unit in lesson.units:
            if not any(component.type == FAQ_TYPE for component in unit.components):
                units.append(unit)
                continue
            evidence = faq_evidence(unit, lesson, scope)
            signal = gate_signal(evidence)
            keep = signal == "strong" or (signal == "weak" and weak_left > 0)
            if keep and signal == "weak":
                weak_left -= 1
            if not keep:
                lean = _without_faq(unit, allowed, unit.title)
                if lean is not None:
                    units.append(lean)
                    left_out.append(unit.title)
                    dropped += 1
                    continue
            items = plan_faq_items(evidence, _fillers(unit, scope))
            components = [_with_plan(component, items) if component.type == FAQ_TYPE else component
                          for component in unit.components]
            units.append(unit.model_copy(update={"components": components}))
            planned += 1
        notes = lesson.notes
        if left_out:
            notes = sanitize_author_text(f"{notes}\n{_dropped_note(left_out, scope.locale)}", 2000)
        result.append(lesson.model_copy(update={"units": units, "notes": notes}))
    floored, added = faq_floor(result, scope)
    return FaqPlanResult(floored, dropped, planned + added, added)


# --- the floor (QLT-8) --------------------------------------------------------------------------------------------
def _has_faq(unit: IdmUnitDesignV1) -> bool:
    return any(component.type == FAQ_TYPE for component in unit.components)


def floor_signal(evidence: Sequence[FaqCandidate], unit: IdmUnitDesignV1) -> str | None:
    """What a unit offers an added FAQ: the gate's signal, or the misconceptions behind the wrong options of its own
    question ("material"); None when it has nothing to clarify."""

    signal = gate_signal(evidence)
    if signal is not None:
        return signal
    has_question = any(component.type == _PROBLEM for component in unit.components)
    return "material" if has_question and any(item.signal == "material" for item in evidence) else None


def _faq_blocks(unit: IdmUnitDesignV1, items: Sequence[IdmFaqItemPlanV1], scope: ModuleScope) -> list[str]:
    """The unit's blocks that hold the planned anchor facts first, then the others (at most the component bound)."""

    anchors = {key for item in items for key in item.anchor_fact_keys}
    holding = [block_id for block_id in unit.block_ids if block_id in scope.blocks
               and anchors & set(scope.blocks[block_id].fact_keys)]
    rest = [block_id for block_id in unit.block_ids if block_id not in holding]
    return [*holding, *rest][:IDM_COMPONENT_MAX_BLOCKS]


def _added_faq(unit: IdmUnitDesignV1, items: list[IdmFaqItemPlanV1], scope: ModuleScope) -> IdmComponentDesignV1:
    vi = scope.locale == "vi"
    title = (f"Hỏi đáp: {unit.title}" if vi else f"FAQ: {unit.title}")[:_TITLE_MAX_CHARS]
    review = ArchitectureComponentAuthorReviewV2(
        purpose=("Phần hỏi đáp do hệ thống thêm để làm rõ chỗ người học dễ nhầm hoặc hay thắc mắc sau phần này; mỗi "
                 "câu bám vào tài liệu." if vi else "FAQ added by the system to clarify what learners are likely to "
                                                    "confuse or ask after this part; each answer stays on the source."),
        example_scenario=None, visual_asset=None,
        user_behavior_navigation=("Người học mở từng câu hỏi để đọc phần giải thích." if vi
                                  else "The learner opens each question to read the explanation."))
    return IdmComponentDesignV1(
        component_index=len(unit.components) + 1, type=FAQ_TYPE, role="clarify", title=title,
        rationale=("Làm rõ chỗ người học dễ nhầm sau phần này." if vi
                   else "Clarifies what learners are likely to confuse after this part."),
        block_ids=_faq_blocks(unit, items, scope), practice_id=None, support_items=[], author_review=review,
        faq_items=items)


def _added_note(titles: Sequence[str], locale: str) -> str:
    named = ", ".join(f'"{title}"' for title in titles)
    if locale == "vi":
        return (f"Đã thêm phần hỏi đáp cho {named} để làm rõ chỗ người học dễ nhầm (lỗi hay gặp, điều kiện, lý do của "
                "phương án sai trong câu hỏi luyện tập).")
    return (f"An FAQ was added to {named} to clarify what learners are likely to confuse (common mistakes, conditions, "
            "why the wrong options of the practice question are wrong).")


def faq_floor(lessons: Sequence[IdmLessonDesignV1], scope: ModuleScope) -> tuple[list[IdmLessonDesignV1], int]:
    """Add an la_faq to the learning units with the most to clarify until ``IDM_FAQ_MIN_UNIT_SHARE`` of the module's
    learning units have one (QLT-8). A unit takes one only when its type is allowed, it has room for a fourth component
    and its lesson holds no held practice (an assessment obligation takes the slot after the components)."""

    result = list(lessons)
    learning = [(lesson_index, unit_index) for lesson_index, lesson in enumerate(result)
                for unit_index, unit in enumerate(lesson.units) if unit.segment != _JOB_AID]
    target = math.ceil(IDM_FAQ_MIN_UNIT_SHARE * len(learning))
    have = sum(_has_faq(result[lesson_index].units[unit_index]) for lesson_index, unit_index in learning)
    if FAQ_TYPE not in scope.allowed_types or have >= target:
        return result, 0
    kinds = {plan.lesson_key: scope.must_do_kind.get(plan.primary_must_do_id or "") for plan in scope.lesson_plans}
    candidates: list[tuple[tuple[int, int, int, int, int], int, int, list[IdmFaqItemPlanV1]]] = []
    for lesson_index, unit_index in learning:
        lesson = result[lesson_index]
        unit = lesson.units[unit_index]
        if (_has_faq(unit) or len(unit.components) >= _MAX_COMPONENTS
                or any(practice.hold for practice in lesson.practice_tasks)):
            continue
        evidence = faq_evidence(unit, lesson, scope)
        signal = floor_signal(evidence, unit)
        if signal is None:
            continue
        items = plan_faq_items(evidence, _fillers(unit, scope))
        if len(items) < IDM_FAQ_MIN_ITEMS:
            continue
        lesson_has_faq = any(_has_faq(other) for other in lesson.units)
        rank = (int(lesson_has_faq), _FLOOR_RANK[signal], 0 if kinds.get(lesson.lesson_key) == "decide" else 1,
                lesson_index, unit_index)
        candidates.append((rank, lesson_index, unit_index, items))
    added: dict[int, list[str]] = {}
    for _rank, lesson_index, unit_index, items in sorted(candidates, key=lambda candidate: candidate[0]):
        if have >= target:
            break
        if len(added.get(lesson_index, [])) >= IDM_FAQ_FLOOR_PER_LESSON:
            continue
        lesson = result[lesson_index]
        unit = lesson.units[unit_index]
        grown = unit.model_copy(update={"components": [*unit.components, _added_faq(unit, items, scope)]})
        units = [grown if position == unit_index else other for position, other in enumerate(lesson.units)]
        result[lesson_index] = lesson.model_copy(update={"units": units})
        added.setdefault(lesson_index, []).append(unit.title)
        have += 1
    for lesson_index, titles in added.items():
        lesson = result[lesson_index]
        notes = sanitize_author_text(f"{lesson.notes}\n{_added_note(titles, scope.locale)}", 2000)
        result[lesson_index] = lesson.model_copy(update={"notes": notes})
    return result, sum(len(titles) for titles in added.values())


__all__ = [
    "DEEP_TYPES", "FAQ_ADDED_CODE", "FAQ_DROPPED_CODE", "FAQ_PLANNED_CODE", "FAQ_TYPE", "FaqCandidate", "FaqPlanResult",
    "Signal", "faq_evidence", "faq_floor", "floor_signal", "gate_and_plan_faq", "gate_signal", "plan_faq_items",
]
