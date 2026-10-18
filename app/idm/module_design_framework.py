"""Framework and orientation steps of the module design: the framework items a lesson's units must present
whole, the framework the course purpose names, and questions moved after the html that teaches them.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Final

from app.idm.contracts import IdmLessonDesignV1, IdmSupportItemV1
from app.idm.framework import framework_promise, unit_framework_brief, unit_framework_detail
from app.idm.module_autofix import QUESTION_ONLY_BLOCK_CODE, TAUGHT_BEFORE_QUESTION_CODE, teach_before_questions
from app.idm.module_layout import ModuleScope
from app.idm.policy import MAX_SUPPORT_ITEMS
from app.idm.runtime import IdmRuntime
from app.idm.text import idm_fold, sanitize_author_text


def framework_orientation_gaps(lessons: Sequence[IdmLessonDesignV1], scope: ModuleScope) -> dict[str, list[str]]:
    """lesson_key -> frameworks ("5 chuyển dịch") the objective of the lesson's Must Do names while no unit title
    of the module names them (QC course 364564, N5: the five shifts were never presented together).

    Only the first lesson serving such an objective is listed: that lesson should open with the overview.
    """

    titled = set()
    for lesson in lessons:
        for unit in lesson.units:
            promise = framework_promise([unit.title])
            if promise is not None:
                titled.add((promise[0], idm_fold(promise[1])))
    gaps: dict[str, list[str]] = {}
    seen: set[tuple[int, str]] = set()
    for plan in scope.lesson_plans:
        promise = framework_promise([scope.must_do_objective.get(plan.primary_must_do_id or "", "")])
        if promise is None:
            continue
        key = (promise[0], idm_fold(promise[1]))
        if key not in titled and key not in seen:
            gaps.setdefault(plan.lesson_key, []).append(promise[2])
        seen.add(key)
    return gaps


def attach_framework_items(lessons: Sequence[IdmLessonDesignV1], scope: ModuleScope,
                           ) -> tuple[list[IdmLessonDesignV1], int]:
    """Give a unit that promises "N <items>" the names of the items its own blocks do not number (QC run 8de1c76b,
    Q3: "Bản đồ 5 chuyển dịch" owned only the introduction; the five shifts were in the next lessons' blocks).

    The names come from the facts of the shard that teach each item and reach the unit's teaching html as a
    server-written support item (``framework.unit_framework_brief``); the unit brief carries it to the writer
    and to the W5 check, without changing which facts the unit owns.
    """

    # One group per block: the items must all come from one block (QC run ab8d67e1, R1).
    shard_groups = [scope.block_texts([block_id]) for plan in scope.lesson_plans for block_id in plan.block_ids]
    attached = 0
    result: list[IdmLessonDesignV1] = []
    for lesson in lessons:
        units = []
        for unit in lesson.units:
            target = next((index for index, component in enumerate(unit.components)
                           if component.type == "html" and component.role != "practice"), None)
            titles = [] if target is None else [unit.title, unit.components[target].title]
            unit_groups = [scope.block_texts([block_id]) for block_id in unit.block_ids]
            brief = None if target is None else unit_framework_brief(titles, unit_groups, shard_groups, scope.locale)
            if target is None or brief is None:
                units.append(unit)
                continue
            component = unit.components[target]
            items = [IdmSupportItemV1(kind="explain_concept", brief=brief, block_id=component.block_ids[0])]
            # QLT-7: what each item is about (its "từ … sang …" clause), so the orientation unit explains the
            # framework instead of listing bare names (QC run fae66524, c2.l1.u1).
            detail = unit_framework_detail(titles, unit_groups, shard_groups, scope.locale)
            if detail is not None:
                items.append(IdmSupportItemV1(kind="explain_concept", brief=detail, block_id=component.block_ids[0]))
            support = [entry for entry in component.support_items
                       if entry.brief not in {item.brief for item in items}][:MAX_SUPPORT_ITEMS - len(items)]
            components = list(unit.components)
            components[target] = component.model_copy(update={"support_items": [*items, *support]})
            units.append(unit.model_copy(update={"components": components}))
            attached += 1
        result.append(lesson.model_copy(update={"units": units}))
    return result, attached


# "5 bước chuyển dịch tư duy" names the framework "chuyển dịch"; "bước" only counts its items.
_COUNTED_FRAMEWORK_RE: Final = re.compile(r"(?<!\w)(\d{1,2})\s+(?:bước|steps?)\s+(\w+(?:\s+\w+)?)", re.IGNORECASE)


def _purpose_phrases(texts: Sequence[str]) -> list[str]:
    """The "N <items>" phrases a course purpose names, the counted framework first ("5 chuyển dịch" for "5 bước
    chuyển dịch")."""

    phrases = [f"{match.group(1)} {match.group(2)}" for text in texts for match in _COUNTED_FRAMEWORK_RE.finditer(text)]
    promise = framework_promise(texts)
    return [*phrases, *([promise[2]] if promise is not None else [])]


def attach_purpose_framework(lessons: Sequence[IdmLessonDesignV1], scope: ModuleScope, purpose_texts: Sequence[str],
                             ) -> tuple[list[IdmLessonDesignV1], bool]:
    """QLT-9 (QC run ffaf7318: the purpose named "5 bước chuyển dịch tư duy cốt lõi" and no unit presented the five
    together): when the course purpose names a framework whose items this module's facts number, and no unit of the
    module already presents it, the module's first teaching html gets the framework's support items (the item names
    and what each changes), so its writer lists them all. Nothing is moved and no fact changes owner."""

    result = list(lessons)
    shard_groups = [scope.block_texts([block_id]) for plan in scope.lesson_plans for block_id in plan.block_ids]
    for phrase in _purpose_phrases(purpose_texts):
        brief = unit_framework_brief([phrase], [], shard_groups, scope.locale)
        if brief is None:
            continue
        lead = brief.split(":", 1)[0]
        if any(item.brief.split(":", 1)[0] == lead for lesson in result for unit in lesson.units
               for component in unit.components for item in component.support_items):
            return result, False
        for lesson_index, lesson in enumerate(result):
            for unit_index, unit in enumerate(lesson.units):
                target = next((index for index, component in enumerate(unit.components)
                               if component.type == "html" and component.role != "practice"), None)
                if target is None:
                    continue
                component = unit.components[target]
                items = [IdmSupportItemV1(kind="explain_concept", brief=brief, block_id=component.block_ids[0])]
                detail = unit_framework_detail([phrase], [], shard_groups, scope.locale)
                if detail is not None:
                    items.append(IdmSupportItemV1(kind="explain_concept", brief=detail,
                                                  block_id=component.block_ids[0]))
                support = component.support_items[:MAX_SUPPORT_ITEMS - len(items)]
                components = list(unit.components)
                components[target] = component.model_copy(update={"support_items": [*items, *support]})
                units = list(lesson.units)
                units[unit_index] = unit.model_copy(update={"components": components})
                result[lesson_index] = lesson.model_copy(update={"units": units})
                return result, True
        return result, False
    return result, False


def _taught_before_questions(lessons: Sequence[IdmLessonDesignV1], scope: ModuleScope, runtime: IdmRuntime,
                             codes: Counter[str]) -> list[IdmLessonDesignV1]:
    """Blocks only a question used join the unit's teaching html (``module_autofix.teach_before_questions``); a
    lesson where no teaching html can take them gets an author note (QC run ab8d67e1, R4)."""

    result: list[IdmLessonDesignV1] = []
    for original, plan in zip(lessons, scope.lesson_plans, strict=True):
        fixed, attached, untaught = teach_before_questions(original, plan)
        if attached:
            runtime.adjustments[TAUGHT_BEFORE_QUESTION_CODE] += attached
        if untaught:
            codes[QUESTION_ONLY_BLOCK_CODE] += untaught
            line = ("Lưu ý: có nội dung chỉ xuất hiện trong lời giải của câu hỏi, chưa được dạy trước câu hỏi — nên "
                    "thêm phần lý thuyết cho nội dung này trước câu hỏi." if scope.locale == "vi"
                    else "Note: some content appears only in a question's explanation and is not taught before the "
                         "question — add a theory part for it before the question.")
            fixed = fixed.model_copy(update={"notes": sanitize_author_text(fixed.notes + "\n" + line, 2000)})
        result.append(fixed)
    return result


def with_orientation_note(lesson: IdmLessonDesignV1, phrases: Sequence[str], locale: str) -> IdmLessonDesignV1:
    named = ", ".join(f'"{phrase}"' for phrase in phrases)
    line = (f"Lưu ý: mục tiêu nêu {named} nhưng chưa có unit định hướng liệt kê đủ các thành phần — nên thêm một "
            "unit tổng quan ở đầu mục này." if locale == "vi"
            else f"Note: the objective names {named} but no orientation unit lists all of its items — add an "
                 "overview unit at the start of this section.")
    return lesson.model_copy(update={"notes": sanitize_author_text(lesson.notes + "\n" + line, 2000)})
