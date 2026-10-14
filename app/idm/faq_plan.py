"""FAQ that digs deeper, W4 side (QLT-5 D): the evidence gate and the item plan of an la_faq slot.

Measured on 100 FAQ items of 6 real runs (plan 2026-10-13 §2.3): 58% recalled a list or checked a yes/no fact, 5% asked
"what if", and an FAQ sat in 39 of 71 units because the layout template put one there, not because the unit had
anything to clarify. After the W4 answer is accepted the server decides, per unit:

* gate: an la_faq stays only when its unit has something to clarify: a decision or policy block, a condition or
  exception the source states, a common-mistake block, a W1 issue or gap that marks a likely confusion, or a question
  whose wrong options carry misconceptions. Otherwise the slot is dropped, its blocks join the unit's teaching
  component and the lesson note says why in plain words (``IDM_W4_AUTOFIX_FAQ_DROPPED``);
* plan: a kept la_faq gets up to three items, each with a question type (why, what if, common mistake first; how to
  apply or comparison only to reach two items), the facts its answer is built on and the origin of its point (a
  mistake or condition the source states, the misconception behind a question's wrong option, or a confusion inferred
  from the facts). The W5 writer follows the plan and the W5 checks ground each answer in its anchor facts.

Everything here is deterministic and server-owned; the W4 provider never writes the plan.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

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
    IDM_FAQ_MIN_ITEMS,
    IDM_FAQ_PLAN_MAX_ITEMS,
)
from app.idm.signals import states_rule
from app.idm.text import sanitize_author_text
from app.instructional_quality import source_clarification_signals

FAQ_DROPPED_CODE: Final = "IDM_W4_AUTOFIX_FAQ_DROPPED"
FAQ_PLANNED_CODE: Final = "IDM_W4_FAQ_ITEMS_PLANNED"
FAQ_TYPE: Final = FAQ_COMPONENT_TYPE
_HTML: Final = "html"
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


@dataclass(frozen=True)
class FaqCandidate:
    question_type: FaqQuestionType
    anchors: tuple[str, ...]
    origin: FaqItemOrigin

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
            found.append(FaqCandidate("common_mistake", _anchors(keys), "distractor"))
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
            found.append(FaqCandidate("common_mistake", _anchors(confused or statements), "ai_drafted"))
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
        return (f"Đã bỏ phần hỏi đáp của {named}: nội dung chưa có điều kiện, ngoại lệ, lỗi hay gặp hay câu hỏi luyện "
                "tập cần làm rõ, nên phần hỏi đáp sẽ chỉ nhắc lại bài học.")
    return (f"The FAQ of {named} was left out: the content has no condition, exception, common mistake or practice "
            "question to clarify, so it would only repeat the lesson.")


@dataclass(frozen=True)
class FaqPlanResult:
    lessons: list[IdmLessonDesignV1]
    dropped: int
    planned: int


def _with_plan(component: IdmComponentDesignV1, items: list[IdmFaqItemPlanV1]) -> IdmComponentDesignV1:
    return component.model_copy(update={"faq_items": items})


def gate_and_plan_faq(lessons: Sequence[IdmLessonDesignV1], scope: ModuleScope) -> FaqPlanResult:
    """Drop every la_faq whose unit has nothing to clarify and plan the items of the others."""

    allowed = scope.allowed_types
    result: list[IdmLessonDesignV1] = []
    dropped = planned = 0
    for lesson in lessons:
        units: list[IdmUnitDesignV1] = []
        left_out: list[str] = []
        for unit in lesson.units:
            if not any(component.type == FAQ_TYPE for component in unit.components):
                units.append(unit)
                continue
            evidence = faq_evidence(unit, lesson, scope)
            if not evidence:
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
    return FaqPlanResult(result, dropped, planned)


__all__ = [
    "DEEP_TYPES", "FAQ_DROPPED_CODE", "FAQ_PLANNED_CODE", "FAQ_TYPE", "FaqCandidate", "FaqPlanResult", "faq_evidence",
    "gate_and_plan_faq", "plan_faq_items",
]
