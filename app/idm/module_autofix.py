"""Deterministic fixes of a W4 lesson design before it is rejected (QC run 8de1c76b, Q1).

3 of 4 chapters of the QC run lost a lesson to the deterministic fallback for reasons the server can fix
itself:

* list bounds: a practice listed all 33 facts of the Canvas block as criteria (bound 24) and the whole answer
  was rejected before any lesson was checked (``trim_module_answer``, applied before strict validation);
* the partition rule: single-block lessons were split into an explanation unit and a practice unit that
  shared the block (Node requires every block in exactly one unit), so the units are merged;
* layout slips: components out of order, a unit block no component used, a lesson block no unit held, a
  diagram or sortable the facts give no evidence for;
* a PRACTICE sortable, crossword or diagram the facts give no evidence for (QLT-5 E: in 3 consecutive real runs a
  lesson fell back to the deterministic design on IDM_W4_FORMAT_EVIDENCE at a practice slot, and the provider repair
  failed too): it becomes a single-choice scenario question, or a worksheet with the question that checks it, keeping
  the linked practice task (``IDM_W4_AUTOFIX_PRACTICE_FORMAT``, a plain note for the author).

Each fix keeps the provider's components and wording; a lesson the fixes cannot make valid is repaired by
the provider as before. Every applied fix is counted (``IDM_W4_AUTOFIX_*``, never content).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Final

from pydantic import ValidationError

from app.idm.contracts import IdmComponentDesignV1, IdmLessonDesignV1, IdmLessonPlanV1
from app.idm.module_layout import ModuleScope, worksheet_check_texts
from app.idm.policy import (
    CASE_DECISION_VERBS,
    IDM_COMPONENT_MAX_BLOCKS,
    IDM_CRITERIA_LABEL_MAX_WORDS,
    IDM_PRACTICE_MAX_CRITERIA_FACTS,
    IDM_UNIT_MAX_BLOCKS,
    MAX_COMPONENTS_PER_UNIT,
    MAX_SUPPORT_ITEMS,
    WORKSHEET_COMPONENT_TYPE,
)
from app.idm.signals import idm_has_ordered_steps, idm_relationship_pairs, idm_term_definitions
from app.idm.text import first_main_verb, idm_fold, sanitize_author_text, single_line, with_suggested_label

CRITERIA_TRIMMED_CODE: Final = "IDM_W4_CRITERIA_TRIMMED"
# A5: a practice the answer holds is released before validation (``_release_hold``).
HOLD_RELEASED_CODE: Final = "IDM_W3_HOLD_RELEASED"
# Bound of IdmFeedbackFocusV1.criterion.
_FOCUS_MAX_CHARS: Final = 400
UNITS_MERGED_CODE: Final = "IDM_W4_AUTOFIX_UNITS_MERGED"
FOREIGN_BLOCK_DROPPED_CODE: Final = "IDM_W4_AUTOFIX_FOREIGN_BLOCK_DROPPED"
BLOCK_ATTACHED_CODE: Final = "IDM_W4_AUTOFIX_BLOCK_ATTACHED"
BLOCK_COVERED_CODE: Final = "IDM_W4_AUTOFIX_BLOCK_COVERED"
FORMAT_TO_HTML_CODE: Final = "IDM_W4_AUTOFIX_FORMAT_TO_HTML"
PRACTICE_FORMAT_CODE: Final = "IDM_W4_AUTOFIX_PRACTICE_FORMAT"
COMPONENT_ORDER_CODE: Final = "IDM_W4_AUTOFIX_COMPONENT_ORDER"
_MIN_TERM_DEFINITIONS: Final = 3
_EVIDENCE_TYPES: Final = frozenset({"la_sortable", "la_crossword", "la_diagram"})
_FIRST_TYPE: Final = "html"
_LAST_TYPE: Final = "la_faq"
# Bounds of IdmLessonDesignV1.notes and IdmComponentDesignV1.title.
_NOTES_MAX_CHARS: Final = 2000
_TITLE_MAX_CHARS: Final = 180
# A single-choice question shows its facts only in its explanation; sortable items and crossword clues show theirs.
_QUESTION_TYPE: Final = "problem"
TAUGHT_BEFORE_QUESTION_CODE: Final = "IDM_W4_AUTOFIX_TAUGHT_BEFORE_QUESTION"
QUESTION_ONLY_BLOCK_CODE: Final = "IDM_W4_QUESTION_ONLY_BLOCK"


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


def _release_hold(practice: dict[str, Any], plan: IdmLessonPlanV1 | None, scope: ModuleScope) -> bool:
    """A5 (owner decision 2026-10-22, contract K2): a practice the answer still holds is designed like any other. It
    loses its hold and SME question; without criteria it takes the facts of its lesson's blocks (what the lesson
    teaches, ranked, within the bound); its feedback criterion is led by the suggested-criteria label, so the worksheet
    and the author read the criteria as suggested. True when the practice was held."""

    held = practice.get("hold") is True
    if not held and practice.get("hold_question") is None:
        return False
    practice.update(hold=False, hold_question=None)
    if not held:
        return False
    keys = practice.get("criteria_fact_keys")
    if plan is not None and not (isinstance(keys, list) and keys):
        taught = [key for block_id in plan.block_ids if block_id in scope.blocks
                  for key in scope.blocks[block_id].fact_keys]
        practice["criteria_fact_keys"] = ranked_criteria(taught, scope, plan)
    focus = practice.get("feedback_focus")
    if isinstance(focus, dict) and isinstance(focus.get("criterion"), str):
        focus["criterion"] = with_suggested_label(focus["criterion"], scope.locale, _FOCUS_MAX_CHARS)
    return True


def trim_module_answer(value: Any, scope: ModuleScope) -> tuple[Any, Counter[str]]:
    """Cut every over-long ``criteria_fact_keys`` of a W4 answer before strict validation (``idm_call`` prepare), after
    releasing every held practice (``_release_hold``, A5).

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
            if isinstance(practice, dict) and _release_hold(practice, plans.get(str(lesson.get("lesson_key"))), scope):
                codes[HOLD_RELEASED_CODE] += 1
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
    kept["support_items"] = list({_support_key(item): item for item in support}.values())[:MAX_SUPPORT_ITEMS]


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
    or becomes the html; a practice of that type is converted by ``_practices_without_evidence``."""

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


# --- practice formats without evidence (QLT-5 E) ----------------------------------------------------------------------
# A practice whose verb decides, recognises or chooses is a single-choice scenario question; "phân biệt" (tell apart)
# and "determine" too. Folded leading verbs, as ``CASE_DECISION_VERBS``.
_DECIDING_VERBS: Final = (*CASE_DECISION_VERBS, "phan biet", "distinguish", "determine")
_LEARNER_LEADS: Final = ("nguoi hoc ", "the learner ")
_FORMAT_LABELS: Final[dict[str, tuple[str, str]]] = {
    "la_sortable": ("Sắp xếp", "ordering"), "la_crossword": ("Ô chữ", "crossword"), "la_diagram": ("Sơ đồ", "diagram")}
_MISSING_EVIDENCE: Final[dict[str, tuple[str, str]]] = {
    "la_sortable": ("các bước theo thứ tự", "steps in a stated order"),
    "la_crossword": ("đủ ba định nghĩa thuật ngữ", "three term definitions"),
    "la_diagram": ("quan hệ giữa các ý", "stated relations between the ideas"),
}
_TARGET_LABELS: Final[dict[str, tuple[str, str]]] = {
    _QUESTION_TYPE: ("câu hỏi tình huống một lựa chọn", "a single-choice scenario question"),
    WORKSHEET_COMPONENT_TYPE: ("phiếu thực hành kèm câu hỏi đối chiếu bài mẫu",
                               "a worksheet with a question on a sample answer"),
}


def _decides(practice: dict[str, Any]) -> bool:
    """The practice asks the learner to decide, recognise or choose: its learner action, or the action named after
    "người học"/"the learner" in its sentence, starts with such a verb."""

    actions = [idm_fold(first_main_verb(str(practice.get("learner_action") or "")))]
    sentence = idm_fold(str(practice.get("sentence") or ""))
    actions.extend(sentence.split(lead, 1)[1] for lead in _LEARNER_LEADS if lead in sentence)
    return any(action == verb or action.startswith(verb + " ") for action in actions for verb in _DECIDING_VERBS)


def _worksheet_check(component: dict[str, Any], locale: str) -> dict[str, Any]:
    """The single-choice problem that checks a converted worksheet (same practice, same blocks)."""

    title, rationale, navigation = worksheet_check_texts(locale)
    review = component["author_review"]
    # component_index is renumbered with the unit (``autofix_lesson``).
    return {"component_index": component["component_index"], "type": _QUESTION_TYPE, "role": "practice",
            "title": title, "rationale": rationale, "block_ids": list(component["block_ids"]),
            "practice_id": component["practice_id"], "support_items": [],
            "author_review": {"purpose": review.get("purpose"), "example_scenario": review.get("example_scenario"),
                              "visual_asset": None, "user_behavior_navigation": navigation}}


def _converted_rationale(source: str, target: str, locale: str) -> str:
    if locale == "vi":
        return (f"{_TARGET_LABELS[target][0].capitalize()} thay cho dạng {_FORMAT_LABELS[source][0]}: tài liệu không "
                f"nêu {_MISSING_EVIDENCE[source][0]}.")
    return (f"{_TARGET_LABELS[target][1].capitalize()} instead of a {_FORMAT_LABELS[source][1]}: the source states no "
            f"{_MISSING_EVIDENCE[source][1]}.")


def practice_format_note(conversions: Sequence[tuple[str, str]], locale: str) -> str:
    """One plain line per conversion for the lesson note: what the practice became and why."""

    if locale == "vi":
        return "\n".join(
            f"Lưu ý: bài luyện tập dạng {_FORMAT_LABELS[source][0]} đã được đổi thành {_TARGET_LABELS[target][0]} vì "
            f"tài liệu không có {_MISSING_EVIDENCE[source][0]}; nhiệm vụ, tiêu chí và phản hồi của bài luyện tập giữ "
            "nguyên." for source, target in conversions)
    return "\n".join(
        f"Note: the {_FORMAT_LABELS[source][1]} practice became {_TARGET_LABELS[target][1]} because the source has no "
        f"{_MISSING_EVIDENCE[source][1]}; the practice task, criteria and feedback are unchanged."
        for source, target in conversions)


def _practices_without_evidence(units: list[dict[str, Any]], practices: dict[str, dict[str, Any]],
                                scope: ModuleScope, allowed: set[str]) -> list[tuple[str, str]]:
    """A practice sortable, crossword or diagram the facts give no evidence for becomes a single-choice problem (a
    practice that decides, recognises or chooses) or a worksheet html checked by a problem of the same practice
    (anything else), keeping its practice_id, criteria and feedback focus. Both need a problem; the other target is
    tried when the preferred one is not allowed. Returns (old type, new type) per conversion; a practice that is
    unknown or held, or a lesson where neither target is allowed, is left to the provider repair."""

    conversions: list[tuple[str, str]] = []
    for unit in units:
        for component in list(unit["components"]):
            practice = practices.get(component.get("practice_id") or "")
            if (component["role"] != "practice" or component["type"] not in _EVIDENCE_TYPES or practice is None
                    or practice.get("hold") or _QUESTION_TYPE not in allowed
                    or format_has_evidence(component, scope)):
                continue
            targets = ((_QUESTION_TYPE, WORKSHEET_COMPONENT_TYPE) if _decides(practice)
                       else (WORKSHEET_COMPONENT_TYPE, _QUESTION_TYPE))
            target = next(kind for kind in targets if kind in allowed)
            source = component["type"]
            component.update(type=target, rationale=_converted_rationale(source, target, scope.locale))
            if target == WORKSHEET_COMPONENT_TYPE:
                # The slot title names the task the learner completes on the worksheet (as a fallback worksheet does).
                lead = "Phiếu thực hành" if scope.locale == "vi" else "Worksheet"
                component["title"] = single_line(f"{lead}: {practice.get('learner_action') or ''}", _TITLE_MAX_CHARS)
            checked = any(item is not component and item["type"] == _QUESTION_TYPE
                          and item["practice_id"] == component["practice_id"] for item in unit["components"])
            if target == WORKSHEET_COMPONENT_TYPE and not checked:
                unit["components"].append(_worksheet_check(component, scope.locale))
            conversions.append((source, target))
    return conversions


def _order_components(units: list[dict[str, Any]], order: dict[str, int]) -> int:
    """Components in the required order: html first, la_faq last, one component per type.

    QC run ab8d67e1 (R7): 4 single-block lessons of chapters 2 and 5 put a teaching html and the worksheet html
    (role practice) in the same unit, each needing a whole W4 repair call (+78 s on the critical path). Two
    components of one type are merged as for merged units (``_merge_components``: the practice one wins and takes
    over the blocks and support items, which is the layout the repair produced); two practices of one type
    cannot share a unit and stay for the provider repair.
    """

    reordered = 0
    for unit in units:
        types = [item["type"] for item in unit["components"]]
        ordered = (sorted(unit["components"], key=_type_rank) if len(types) == len(set(types))
                   else _merge_components(unit["components"], order))
        if ordered != unit["components"]:
            unit["components"] = ordered
            reordered += 1
    return reordered


def teach_before_questions(lesson: IdmLessonDesignV1, plan: IdmLessonPlanV1) -> tuple[IdmLessonDesignV1, int, int]:
    """Every block a question uses is taught by the teaching html of its unit (QC run ab8d67e1, R4).

    Node gives each fact to the first component of the unit that lists its block, so a block only a problem listed
    (cb_0014: 12 facts "khi móng tư duy đổi / không đổi"; cb_0015: "90% thất bại") was "covered" by that question's
    explanation and never taught. Such a block joins the unit's teaching html (html first, so it now owns the facts;
    the question keeps them as supporting evidence). Returns the lesson, the blocks attached and the blocks only a
    question uses in a unit without a teaching html (left for the author).
    """

    order = {block_id: index for index, block_id in enumerate(plan.block_ids)}
    units = []
    attached = untaught = 0
    for unit in lesson.units:
        questions = {block_id for component in unit.components if component.type == _QUESTION_TYPE
                     for block_id in component.block_ids}
        taught = {block_id for component in unit.components if component.type != _QUESTION_TYPE
                  for block_id in component.block_ids}
        missing = sorted(questions - taught, key=lambda block_id: order.get(block_id, len(order)))
        teaching = next((index for index, component in enumerate(unit.components)
                         if component.type == _FIRST_TYPE and component.role != "practice"), None)
        if not missing:
            units.append(unit)
            continue
        if teaching is None or len(unit.components[teaching].block_ids) + len(missing) > IDM_COMPONENT_MAX_BLOCKS:
            untaught += len(missing)
            units.append(unit)
            continue
        components = list(unit.components)
        block_ids = sorted([*components[teaching].block_ids, *missing],
                           key=lambda block_id: order.get(block_id, len(order)))
        components[teaching] = components[teaching].model_copy(update={"block_ids": block_ids})
        units.append(unit.model_copy(update={"components": components}))
        attached += len(missing)
    return (lesson.model_copy(update={"units": units}) if attached else lesson), attached, untaught


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
        practices = {str(item["practice_id"]): item for item in data["practice_tasks"]}
        conversions = _practices_without_evidence(units, practices, scope, allowed)
        if conversions:
            codes[PRACTICE_FORMAT_CODE] += len(conversions)
            data["notes"] = sanitize_author_text(f"{data['notes']}\n{practice_format_note(conversions, scope.locale)}",
                                                 _NOTES_MAX_CHARS)
        codes[COMPONENT_ORDER_CODE] += _order_components(units, order)
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
    "FOREIGN_BLOCK_DROPPED_CODE", "FORMAT_TO_HTML_CODE", "HOLD_RELEASED_CODE", "PRACTICE_FORMAT_CODE",
    "QUESTION_ONLY_BLOCK_CODE", "TAUGHT_BEFORE_QUESTION_CODE", "UNITS_MERGED_CODE", "autofix_lesson",
    "format_has_evidence", "practice_format_note", "ranked_criteria", "teach_before_questions", "trim_module_answer",
]
