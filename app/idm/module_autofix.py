"""Deterministic fixes of a W4 lesson design before it is rejected (QC run 8de1c76b, Q1).

3 of 4 chapters of the QC run lost a lesson to the deterministic fallback for reasons the server can fix
itself:

* list bounds: a practice listed all 33 facts of the Canvas block as criteria (bound 24) and the whole answer
  was rejected before any lesson was checked (``trim_module_answer``, applied before strict validation);
* the partition rule: single-block lessons were split into an explanation unit and a practice unit that
  shared the block (Node requires every block in exactly one unit), so the units are merged;
* layout slips: components out of order, a unit block no component used, a lesson block no unit held, a
  diagram or sortable the facts give no evidence for.

Each fix keeps the provider's components and wording; a lesson the fixes cannot make valid is repaired by
the provider as before. Every applied fix is counted (``IDM_W4_AUTOFIX_*``, never content).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Final

from pydantic import ValidationError

from app.idm.contracts import IdmComponentDesignV1, IdmLessonDesignV1, IdmLessonPlanV1
from app.idm.module_layout import ModuleScope
from app.idm.policy import (
    IDM_COMPONENT_MAX_BLOCKS,
    IDM_CRITERIA_LABEL_MAX_WORDS,
    IDM_PRACTICE_MAX_CRITERIA_FACTS,
    IDM_UNIT_MAX_BLOCKS,
    MAX_COMPONENTS_PER_UNIT,
)
from app.idm.signals import idm_has_ordered_steps, idm_relationship_pairs, idm_term_definitions

CRITERIA_TRIMMED_CODE: Final = "IDM_W4_CRITERIA_TRIMMED"
UNITS_MERGED_CODE: Final = "IDM_W4_AUTOFIX_UNITS_MERGED"
FOREIGN_BLOCK_DROPPED_CODE: Final = "IDM_W4_AUTOFIX_FOREIGN_BLOCK_DROPPED"
BLOCK_ATTACHED_CODE: Final = "IDM_W4_AUTOFIX_BLOCK_ATTACHED"
BLOCK_COVERED_CODE: Final = "IDM_W4_AUTOFIX_BLOCK_COVERED"
FORMAT_TO_HTML_CODE: Final = "IDM_W4_AUTOFIX_FORMAT_TO_HTML"
COMPONENT_ORDER_CODE: Final = "IDM_W4_AUTOFIX_COMPONENT_ORDER"
_MIN_TERM_DEFINITIONS: Final = 3
_MAX_SUPPORT_ITEMS: Final = 6
_EVIDENCE_TYPES: Final = frozenset({"la_sortable", "la_crossword", "la_diagram"})
_FIRST_TYPE: Final = "html"
_LAST_TYPE: Final = "la_faq"


# --- list bounds ---------------------------------------------------------------------------------------------
def _criteria_rank(scope: ModuleScope, plan: IdmLessonPlanV1 | None) -> Callable[[str], tuple[int, int]]:
    """Rank of a criteria fact: facts of the lesson's own Must Do block first (``must_do`` row of its primary
    Must Do), then blocks serving its Must Dos, then the rest of the lesson, foreign facts last; within a rank,
    a statement before a short label or heading."""

    tier: dict[str, int] = {}
    if plan is not None:
        must_dos = {item for item in (plan.primary_must_do_id, *plan.secondary_must_do_ids) if item}
        for block_id in plan.block_ids:
            row = scope.rows.get(block_id)
            block = scope.blocks.get(block_id)
            if row is None or block is None:
                continue
            serves = set(row.must_do_ids) & must_dos
            level = (0 if plan.primary_must_do_id in row.must_do_ids and row.classification == "must_do"
                     else 1 if serves else 2)
            for key in block.fact_keys:
                tier[key] = min(level, tier.get(key, level))

    def rank(key: str) -> tuple[int, int]:
        words = len(scope.fact_text.get(key, "").split())
        return tier.get(key, 3), int(words <= IDM_CRITERIA_LABEL_MAX_WORDS)

    return rank


def ranked_criteria(keys: Sequence[str], scope: ModuleScope, plan: IdmLessonPlanV1 | None,
                    limit: int = IDM_PRACTICE_MAX_CRITERIA_FACTS) -> list[str]:
    """``keys`` without duplicates, cut to the ``limit`` most relevant ones, in their original order."""

    unique = list(dict.fromkeys(keys))
    if len(unique) <= limit:
        return unique
    rank = _criteria_rank(scope, plan)
    chosen = sorted(range(len(unique)), key=lambda index: (*rank(unique[index]), index))[:limit]
    return [unique[index] for index in sorted(chosen)]


def trim_module_answer(value: Any, scope: ModuleScope) -> tuple[Any, Counter[str]]:
    """Cut every over-long ``criteria_fact_keys`` of a W4 answer before strict validation (``idm_call`` prepare).

    Only lists over the bound are touched; a list of anything but strings is left for validation to reject.
    """

    codes: Counter[str] = Counter()
    lessons = value.get("lessons") if isinstance(value, dict) else None
    if not isinstance(lessons, list):
        return value, codes
    plans = {plan.lesson_key: plan for plan in scope.lesson_plans}
    for lesson in lessons:
        practices = lesson.get("practice_tasks") if isinstance(lesson, dict) else None
        for practice in practices if isinstance(practices, list) else []:
            keys = practice.get("criteria_fact_keys") if isinstance(practice, dict) else None
            if (isinstance(keys, list) and len(keys) > IDM_PRACTICE_MAX_CRITERIA_FACTS
                    and all(isinstance(item, str) for item in keys)):
                practice["criteria_fact_keys"] = ranked_criteria(
                    [item.strip() for item in keys], scope, plans.get(str(lesson.get("lesson_key"))))
                codes[CRITERIA_TRIMMED_CODE] += 1
    return value, codes


# --- format evidence -------------------------------------------------------------------------------------------
def format_has_evidence(component: IdmComponentDesignV1 | dict[str, Any], scope: ModuleScope) -> bool:
    """A sortable needs source order, a crossword three definitions, a diagram stated relations."""

    kind = component.type if isinstance(component, IdmComponentDesignV1) else component.get("type")
    block_ids = component.block_ids if isinstance(component, IdmComponentDesignV1) else component.get("block_ids")
    texts = scope.block_texts([block_id for block_id in block_ids or [] if block_id in scope.blocks])
    if kind == "la_sortable":
        return idm_has_ordered_steps(texts, locale=scope.locale)
    if kind == "la_crossword":
        return len(idm_term_definitions(texts)) >= _MIN_TERM_DEFINITIONS
    if kind == "la_diagram":
        return bool(idm_relationship_pairs(texts))
    return True


# --- layout fixes ----------------------------------------------------------------------------------------------
class _Unfixable(Exception):
    """A fix would break another rule; the lesson goes to the provider repair unchanged."""


def _union(first: Sequence[str], second: Sequence[str], order: dict[str, int], limit: int) -> list[str]:
    merged = sorted(dict.fromkeys([*first, *second]), key=lambda block_id: order.get(block_id, len(order)))
    if len(merged) > limit:
        raise _Unfixable
    return merged


def _fold(kept: dict[str, Any], dropped: dict[str, Any], order: dict[str, int]) -> None:
    """``kept`` takes over the blocks and support items of a component that is removed."""

    kept["block_ids"] = _union(kept["block_ids"], dropped["block_ids"], order, IDM_COMPONENT_MAX_BLOCKS)
    support = [*kept["support_items"], *dropped["support_items"]]
    kept["support_items"] = list({_support_key(item): item for item in support}.values())[:_MAX_SUPPORT_ITEMS]


def _support_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return str(item.get("kind")), str(item.get("brief")), str(item.get("block_id"))


def _type_rank(component: dict[str, Any]) -> int:
    return 0 if component["type"] == _FIRST_TYPE else 2 if component["type"] == _LAST_TYPE else 1


def _merge_components(components: list[dict[str, Any]], order: dict[str, int]) -> list[dict[str, Any]]:
    """One component per type: a practice component wins over a teaching one of the same type and takes over
    its blocks and support items; two practices of one type cannot share a unit."""

    kept: dict[str, dict[str, Any]] = {}
    for component in components:
        current = kept.get(component["type"])
        if current is None:
            kept[component["type"]] = component
            continue
        both = current["role"] == "practice" and component["role"] == "practice"
        if both and current["practice_id"] != component["practice_id"]:
            raise _Unfixable
        later_wins = component["role"] == "practice" and current["role"] != "practice"
        winner, loser = (component, current) if later_wins else (current, component)
        _fold(winner, loser, order)
        kept[component["type"]] = winner
    merged = sorted(kept.values(), key=_type_rank)
    if len(merged) > MAX_COMPONENTS_PER_UNIT:
        raise _Unfixable
    return merged


def _merge_shared_units(units: list[dict[str, Any]], order: dict[str, int]) -> tuple[list[dict[str, Any]], int]:
    """Units that share a block become one unit at the first one's position (Node partition rule)."""

    groups: list[list[dict[str, Any]]] = []
    for unit in units:
        touching = [group for group in groups
                    if set(unit["block_ids"]) & {block_id for member in group for block_id in member["block_ids"]}]
        merged = [member for group in touching for member in group] + [unit]
        groups = [group for group in groups if not any(group is item for item in touching)]
        groups.append(merged)
    position = {id(unit): index for index, unit in enumerate(units)}
    groups.sort(key=lambda group: min(position[id(member)] for member in group))
    result: list[dict[str, Any]] = []
    merges = 0
    for group in groups:
        group.sort(key=lambda member: position[id(member)])
        if len(group) == 1:
            result.append(group[0])
            continue
        merges += len(group) - 1
        first = group[0]
        components = _merge_components([component for member in group for component in member["components"]],
                                       order)
        block_ids: list[str] = []
        for member in group:
            block_ids = _union(block_ids, member["block_ids"], order, IDM_UNIT_MAX_BLOCKS)
        practice = any(component["role"] == "practice" for component in components)
        result.append({**first, "block_ids": block_ids, "components": components,
                       "segment": "practice_feedback" if practice else first["segment"],
                       "media_brief": next((member["media_brief"] for member in group if member["media_brief"]),
                                           None)})
    return result, merges


def _teaching_component(unit: dict[str, Any]) -> dict[str, Any]:
    """Where an extra block of a unit goes: its teaching html, else a teaching component, else the first."""

    components = unit["components"]
    return next((item for item in components if item["type"] == _FIRST_TYPE and item["role"] != "practice"),
                next((item for item in components if item["role"] != "practice"), components[0]))


def _drop_foreign_blocks(units: list[dict[str, Any]], order: dict[str, int]) -> int:
    dropped = 0
    for unit in units:
        if all(block_id in order for block_id in unit["block_ids"]):
            continue
        components = [[block_id for block_id in item["block_ids"] if block_id in order] for item in unit["components"]]
        kept = [block_id for block_id in unit["block_ids"] if block_id in order]
        if not kept or not all(components):
            raise _Unfixable
        dropped += len(unit["block_ids"]) - len(kept)
        unit["block_ids"] = kept
        for component, block_ids in zip(unit["components"], components, strict=True):
            component["block_ids"] = block_ids
    return dropped


def _attach_missing_blocks(units: list[dict[str, Any]], plan: IdmLessonPlanV1, order: dict[str, int]) -> int:
    """A lesson block no unit holds joins the unit that holds the block before it (else the first unit)."""

    held = {block_id for unit in units for block_id in unit["block_ids"]}
    attached = 0
    for block_id in plan.block_ids:
        if block_id in held:
            continue
        earlier = [index for index, unit in enumerate(units)
                   if any(order[item] < order[block_id] for item in unit["block_ids"])]
        unit = units[earlier[-1] if earlier else 0]
        unit["block_ids"] = _union(unit["block_ids"], [block_id], order, IDM_UNIT_MAX_BLOCKS)
        target = _teaching_component(unit)
        target["block_ids"] = _union(target["block_ids"], [block_id], order, IDM_COMPONENT_MAX_BLOCKS)
        held.add(block_id)
        attached += 1
    return attached


def _cover_unit_blocks(units: list[dict[str, Any]], order: dict[str, int]) -> int:
    covered = 0
    for unit in units:
        used = {block_id for item in unit["components"] for block_id in item["block_ids"]}
        missing = [block_id for block_id in unit["block_ids"] if block_id not in used]
        if missing:
            target = _teaching_component(unit)
            target["block_ids"] = _union(target["block_ids"], missing, order, IDM_COMPONENT_MAX_BLOCKS)
            covered += len(missing)
    return covered


def _formats_without_evidence(units: list[dict[str, Any]], scope: ModuleScope, allowed: set[str],
                              order: dict[str, int]) -> int:
    """A teaching diagram, sortable or crossword the facts give no evidence for is folded into the unit's html,
    or becomes the html; a practice of that type is left to the provider repair."""

    changed = 0
    for unit in units:
        for component in list(unit["components"]):
            if (component["type"] not in _EVIDENCE_TYPES or component["role"] == "practice"
                    or format_has_evidence(component, scope)):
                continue
            html = next((item for item in unit["components"] if item["type"] == _FIRST_TYPE), None)
            if html is not None:
                _fold(html, component, order)
                unit["components"].remove(component)
            elif _FIRST_TYPE in allowed:
                component["type"] = _FIRST_TYPE
            else:
                raise _Unfixable
            changed += 1
    return changed


def _order_components(units: list[dict[str, Any]]) -> int:
    reordered = 0
    for unit in units:
        types = [item["type"] for item in unit["components"]]
        ordered = sorted(unit["components"], key=_type_rank)
        if len(types) == len(set(types)) and ordered != unit["components"]:
            unit["components"] = ordered
            reordered += 1
    return reordered


def autofix_lesson(lesson: IdmLessonDesignV1, plan: IdmLessonPlanV1, scope: ModuleScope,
                   allowed: set[str]) -> tuple[IdmLessonDesignV1, Counter[str]]:
    """``lesson`` with the deterministic layout fixes applied, and their counts; unchanged when none applies or
    when a fix would break a bound."""

    data = lesson.model_dump(mode="json")
    order = {block_id: index for index, block_id in enumerate(plan.block_ids)}
    codes: Counter[str] = Counter()
    try:
        units = data["units"]
        codes[FOREIGN_BLOCK_DROPPED_CODE] += _drop_foreign_blocks(units, order)
        units, merges = _merge_shared_units(units, order)
        codes[UNITS_MERGED_CODE] += merges
        codes[BLOCK_ATTACHED_CODE] += _attach_missing_blocks(units, plan, order)
        codes[BLOCK_COVERED_CODE] += _cover_unit_blocks(units, order)
        codes[FORMAT_TO_HTML_CODE] += _formats_without_evidence(units, scope, allowed, order)
        codes[COMPONENT_ORDER_CODE] += _order_components(units)
        for unit_position, unit in enumerate(units, start=1):
            unit["unit_index"] = unit_position
            for component_position, component in enumerate(unit["components"], start=1):
                component["component_index"] = component_position
        data["units"] = units
        fixed = IdmLessonDesignV1.model_validate(data)
    except (_Unfixable, ValidationError):
        return lesson, Counter()
    codes = +codes
    return (fixed, codes) if codes else (lesson, codes)


__all__ = [
    "BLOCK_ATTACHED_CODE", "BLOCK_COVERED_CODE", "COMPONENT_ORDER_CODE", "CRITERIA_TRIMMED_CODE",
    "FOREIGN_BLOCK_DROPPED_CODE", "FORMAT_TO_HTML_CODE", "UNITS_MERGED_CODE", "autofix_lesson",
    "format_has_evidence", "ranked_criteria", "trim_module_answer",
]
