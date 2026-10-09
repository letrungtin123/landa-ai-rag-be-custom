"""IDM ``chapter_blueprint`` task: W3 treatment + W4 lesson layout (spec §7.6, §8.2).

One call designs every lesson of a shard. Valid lessons are kept; a lesson that
still fails after one repair gets a deterministic layout of its own (partial
salvage), so a single bad lesson never discards the shard.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import ValidationError

from app.idm.contracts import (
    IdmLessonDesignV1,
    IdmLessonPlanV1,
    IdmModuleContextV1,
    IdmPracticeTaskV1,
    IdmShardDesignV1,
    IdmSupportItemV1,
    IdmW3W4ModuleResponseV1,
    StageOrigin,
    design_hash_of,
)
from app.idm.framework import framework_promise, unit_framework_brief
from app.idm.limits import answer_limits
from app.idm.mcq import encode_key_letters
from app.idm.module_autofix import autofix_lesson, format_has_evidence, trim_module_answer
from app.idm.module_layout import (
    ModuleScope,
    build_module_scope,
    clean_feedback,
    clean_media_brief,
    fallback_lesson,
    project_lesson,
    project_shard,
)
from app.idm.policy import (
    AI_DRAFTED_MARKER_EN,
    AI_DRAFTED_MARKER_VI,
    DOING_PRACTICE_TYPES,
    GRADED_PRACTICE_TYPES,
    IDM_MODULE_CONTENT_ATTEMPTS,
    IDM_MODULE_MAX_OUTPUT_TOKENS,
    IDM_MODULE_SCHEMA_REPAIRS,
    IDM_THEORY_RUN_MAX_COMPONENTS,
    MAX_SUPPORT_ITEMS,
    THINKING_MODULE,
    WORKSHEET_COMPONENT_TYPE,
)
from app.idm.prompts import COMPACT_MODULE, answer_repair, module_prompt, repair_suffix
from app.idm.runtime import (
    SCHEMA_INVALID_CODE,
    IdmBudgetError,
    IdmProviderError,
    IdmResponseInvalidError,
    IdmRuntime,
    InvocationKind,
    ThinkingLevel,
    idm_call,
    log_stage,
    record_deterministic_fallback,
    repair_thinking,
)
from app.idm.text import idm_fold, produces_output, sanitize_author_text, single_line
from app.idm.validation import IdmIssue, errors
from app.lesson_author_orchestration_v2 import (
    ArchitectureComponentAuthorReviewV2,
    CourseSkeletonV2,
    LessonArchitectureV2,
)
from app.lesson_author_orchestration_v2_provider import ChapterShardPlanV2, SourceSnapshotFactV2

# Graded practice types plus the worksheet html of a "do" Must Do (spec §10.1, QC 234653 R5).
PRACTICE_COMPONENT_TYPES: Final = GRADED_PRACTICE_TYPES | {WORKSHEET_COMPONENT_TYPE}
_MS: Final = 1000
# Failing codes/paths logged per W4 attempt (QC run 8de1c76b, Q1d); paths are server-built, never content.
_MAX_LOGGED_ISSUES: Final = 24
_UNIT_PATH_RE: Final = re.compile(r"\.units\[(\d+)\]")
_MAX_ACTIVITIES: Final = 3
_MAX_UNITS_PER_LESSON: Final = 12


# --- validation -------------------------------------------------------------------------------
def _sentence_has_shape(sentence: str, locale: str) -> bool:
    folded = idm_fold(sentence)
    if locale == "vi":
        return "nguoi hoc" in folded and ("cho" in folded or "voi" in folded) and " de " in f" {folded} "
    return "learner" in folded and ("given" in folded or "with" in folded) and " to " in f" {folded} "


def validate_lesson(
    lesson: IdmLessonDesignV1,
    plan: IdmLessonPlanV1,
    scope: ModuleScope,
    allowed_types: set[str],
    path: str,
) -> list[IdmIssue]:
    issues: list[IdmIssue] = []
    lesson_blocks = set(plan.block_ids)
    lesson_facts = scope.lesson_fact_keys(plan)
    practices = {practice.practice_id: practice for practice in lesson.practice_tasks}
    if len(practices) != len(lesson.practice_tasks):
        issues.append(IdmIssue("IDM_W3_PRACTICE_UNKNOWN", f"{path}.practice_tasks"))
    if plan.kind == "job_aid" and any(component.role == "practice" for unit in lesson.units
                                      for component in unit.components):
        issues.append(IdmIssue("IDM_W3_JOB_AID_HAS_PRACTICE", f"{path}.units"))
    if plan.kind == "learning" and not lesson.practice_tasks:
        issues.append(IdmIssue("IDM_W3_PRACTICE_MISSING", f"{path}.practice_tasks"))
    for index, practice in enumerate(lesson.practice_tasks):
        practice_path = f"{path}.practice_tasks[{index}]"
        if not practice.hold and not practice.criteria_fact_keys:
            issues.append(IdmIssue("IDM_W3_PRACTICE_CRITERIA_MISSING", practice_path))
        if any(key not in lesson_facts for key in practice.criteria_fact_keys):
            issues.append(IdmIssue("IDM_W3_PRACTICE_CRITERIA_FOREIGN", f"{practice_path}.criteria_fact_keys"))
        if not _sentence_has_shape(practice.sentence, scope.locale):
            issues.append(IdmIssue("IDM_W3_PRACTICE_SENTENCE_FORMAT", f"{practice_path}.sentence", "warning"))
    unit_blocks = [block_id for unit in lesson.units for block_id in unit.block_ids]
    if len(unit_blocks) != len(set(unit_blocks)) or set(unit_blocks) != lesson_blocks:
        issues.append(IdmIssue("IDM_W4_UNIT_BLOCK_PARTITION", f"{path}.units"))
    practice_unit: dict[str, int] = {}
    # practice_id -> units holding a graded check / (unit, path) of each worksheet; types per practice.
    graded_units: dict[str, set[int]] = {}
    worksheets: list[tuple[str, int, str]] = []
    practice_types: set[str] = set()
    unit_of_fact = {
        key: position
        for position, unit in enumerate(lesson.units)
        for block_id in unit.block_ids
        if block_id in scope.blocks
        for key in scope.blocks[block_id].fact_keys
    }
    theory_run = 0
    for unit_position, unit in enumerate(lesson.units):
        unit_path = f"{path}.units[{unit_position}]"
        own = set(unit.block_ids)
        types = [component.type for component in unit.components]
        if (
            len(types) != len(set(types))
            or ("html" in types and types[0] != "html")
            or ("la_faq" in types and types[-1] != "la_faq")
        ):
            issues.append(IdmIssue("IDM_W4_COMPONENT_ORDER", f"{unit_path}.components"))
        if len(unit.block_ids) != len(own) or any(
            len(component.block_ids) != len(set(component.block_ids)) for component in unit.components
        ):
            issues.append(IdmIssue("IDM_W4_BLOCK_IDS_DUPLICATED", f"{unit_path}.block_ids"))
        covered = {block_id for component in unit.components for block_id in component.block_ids}
        if not covered <= own:
            issues.append(IdmIssue("IDM_W4_COMPONENT_BLOCKS_OUTSIDE_UNIT", f"{unit_path}.components"))
        if own - covered:
            issues.append(IdmIssue("IDM_W4_UNIT_BLOCKS_UNCOVERED", f"{unit_path}.components"))
        for component_position, component in enumerate(unit.components):
            component_path = f"{unit_path}.components[{component_position}]"
            if component.type not in allowed_types:
                issues.append(IdmIssue("IDM_W4_COMPONENT_TYPE_NOT_ALLOWED", f"{component_path}.type"))
            if component.author_review.purpose is None:
                issues.append(IdmIssue("IDM_W4_AUTHOR_REVIEW_PURPOSE_MISSING", f"{component_path}.author_review"))
            if not format_has_evidence(component, scope):
                issues.append(IdmIssue("IDM_W4_FORMAT_EVIDENCE", f"{component_path}.type"))
            if component.role == "practice":
                theory_run = 0
                linked = practices.get(component.practice_id or "")
                if linked is None or component.type not in PRACTICE_COMPONENT_TYPES:
                    issues.append(IdmIssue("IDM_W3_PRACTICE_UNKNOWN", f"{component_path}.practice_id"))
                elif linked.hold:
                    issues.append(IdmIssue("IDM_W3_HELD_PRACTICE_HAS_COMPONENT", component_path))
                else:
                    practice_unit[linked.practice_id] = unit_position
                    practice_types.add(component.type)
                    if component.type == WORKSHEET_COMPONENT_TYPE:
                        worksheets.append((linked.practice_id, unit_position, component_path))
                    else:
                        graded_units.setdefault(linked.practice_id, set()).add(unit_position)
            else:
                theory_run += 1
                if theory_run > IDM_THEORY_RUN_MAX_COMPONENTS:
                    issues.append(IdmIssue("IDM_W4_THEORY_RUN", component_path, "warning"))
    # A worksheet is never graded by the LMS: the same unit checks it with a graded practice.
    issues.extend(IdmIssue("IDM_W3_WORKSHEET_CHECK_MISSING", path_of)
                  for practice_id, unit_position, path_of in worksheets
                  if unit_position not in graded_units.get(practice_id, set()))
    primary = plan.primary_must_do_id or ""
    if (plan.kind == "learning" and WORKSHEET_COMPONENT_TYPE in allowed_types and practice_types
            and produces_output(scope.must_do_statement.get(primary, ""), scope.must_do_kind.get(primary, ""))
            and not practice_types & DOING_PRACTICE_TYPES):
        issues.append(IdmIssue("IDM_W3_DO_PRACTICE_RECOGNITION_ONLY", f"{path}.practice_tasks", "warning"))
    for index, practice in enumerate(lesson.practice_tasks):
        if practice.hold:
            continue
        if practice.practice_id not in practice_unit:
            issues.append(IdmIssue("IDM_W3_PRACTICE_COMPONENT_MISSING", f"{path}.practice_tasks[{index}]"))
        elif any(
            unit_of_fact.get(key, -1) > practice_unit[practice.practice_id] for key in practice.criteria_fact_keys
        ):
            issues.append(IdmIssue("IDM_W4_PRACTICE_BEFORE_SUPPORT", f"{path}.practice_tasks[{index}]"))
    return list(dict.fromkeys(issues))


# --- normalisation, fallback and holds -----------------------------------------------------
def _clean_review(
    review: ArchitectureComponentAuthorReviewV2, *, drafted: bool, locale: str, drafted_hint: str
) -> ArchitectureComponentAuthorReviewV2:
    marker = AI_DRAFTED_MARKER_VI if locale == "vi" else AI_DRAFTED_MARKER_EN

    def clean(value: str | None, limit: int) -> str | None:
        return sanitize_author_text(value, limit) if value else None

    scenario = clean(review.example_scenario, 2000)
    if drafted and not (scenario or "").startswith(marker):
        scenario = sanitize_author_text(f"{marker} {scenario or drafted_hint}", 2000)
    return ArchitectureComponentAuthorReviewV2(
        purpose=clean(review.purpose, 1200),
        example_scenario=scenario,
        visual_asset=clean(review.visual_asset, 1200),
        user_behavior_navigation=clean(review.user_behavior_navigation, 1200),
    )


def normalize_lesson(
    lesson: IdmLessonDesignV1, plan: IdmLessonPlanV1, locale: str, warnings: Sequence[str]
) -> IdmLessonDesignV1:
    """Sanitise text, renumber indices, mark AI-drafted scenarios, keep warnings in notes."""

    practices = {practice.practice_id: practice for practice in lesson.practice_tasks}
    units = []
    for unit_position, unit in enumerate(lesson.units, start=1):
        components = []
        for component_position, component in enumerate(unit.components, start=1):
            practice = practices.get(component.practice_id or "")
            drafted = practice is not None and practice.scenario_origin == "ai_drafted"
            components.append(
                component.model_copy(
                    update={
                        "component_index": component_position,
                        "title": single_line(component.title, 180),
                        "rationale": single_line(component.rationale, 500),
                        "support_items": [
                            item.model_copy(update={"brief": single_line(item.brief, 300)})
                            for item in component.support_items
                        ],
                        "author_review": _clean_review(
                            component.author_review,
                            drafted=drafted,
                            locale=locale,
                            drafted_hint=practice.context_input if practice else "",
                        ),
                    }
                )
            )
        units.append(
            unit.model_copy(
                update={
                    "unit_index": unit_position,
                    "title": single_line(unit.title, 180),
                    "purpose": single_line(unit.purpose, 500),
                    "components": components,
                    "media_brief": clean_media_brief(unit.media_brief),
                }
            )
        )
    notes = sanitize_author_text(lesson.notes, 2000)
    worksheet = any(component.type == WORKSHEET_COMPONENT_TYPE and component.role == "practice"
                    for unit in lesson.units for component in unit.components)
    extra = _warning_notes(warnings, locale, worksheet=worksheet)
    if extra:
        notes = sanitize_author_text(f"{notes}\n{extra}", 2000)
    return lesson.model_copy(
        update={
            "title": single_line(lesson.title, 180),
            "objective": single_line(lesson.objective, 500),
            "learning_objectives": [single_line(item, 500) for item in lesson.learning_objectives],
            "practice_tasks": []
            if plan.kind == "job_aid"
            else [
                practice.model_copy(
                    update={
                        "sentence": single_line(practice.sentence, 280),
                        "context_input": single_line(practice.context_input, 300),
                        "learner_action": single_line(practice.learner_action, 200),
                        "result": single_line(practice.result, 200),
                        "hold_question": single_line(practice.hold_question, 400) if practice.hold_question else None,
                        "feedback_focus": clean_feedback(practice.feedback_focus),
                    }
                )
                for practice in lesson.practice_tasks
            ],
            "assessment": single_line(lesson.assessment, 500),
            "units": units,
            "notes": notes,
        }
    )


def _warning_notes(warnings: Sequence[str], locale: str, *, worksheet: bool = False) -> str:
    lines = []
    if worksheet:
        lines.append(
            "Practice dạng phiếu thực hành: hệ thống chưa chấm câu trả lời tự luận — người học tự đối chiếu bài "
            "làm với danh sách tự kiểm; câu hỏi đi kèm kiểm tra việc áp dụng tiêu chí."
            if locale == "vi"
            else "Worksheet practice: the LMS does not grade free-text answers — learners check their own work "
            "against the self-check list; the question that follows checks how they apply the criteria."
        )
    if "IDM_W3_DO_PRACTICE_RECOGNITION_ONLY" in warnings:
        lines.append(
            "Lưu ý: Must Do loại làm nhưng bài luyện tập chỉ kiểm tra nhận biết — nên bổ sung phiếu thực hành "
            "(worksheet) để người học tự làm sản phẩm."
            if locale == "vi"
            else "Note: the Must Do asks the learner to produce something, but the practice only checks "
            "recognition — add a worksheet practice."
        )
    if "IDM_W4_THEORY_RUN" in warnings:
        lines.append(
            "Cảnh báo: có hơn 5 khối giải thích liên tiếp không có câu hỏi."
            if locale == "vi"
            else "Warning: more than 5 explanation blocks in a row without a question."
        )
    if "IDM_W3_PRACTICE_SENTENCE_FORMAT" in warnings:
        lines.append(
            "Lưu ý: câu Practice chưa đủ bối cảnh/hành động/kết quả."
            if locale == "vi"
            else "Note: a practice sentence lacks context/action/result."
        )
    return "\n".join(lines)


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
            brief = None if target is None else unit_framework_brief(
                [unit.title, unit.components[target].title],
                [scope.block_texts([block_id]) for block_id in unit.block_ids], shard_groups, scope.locale)
            if target is None or brief is None:
                units.append(unit)
                continue
            component = unit.components[target]
            item = IdmSupportItemV1(kind="explain_concept", brief=brief, block_id=component.block_ids[0])
            support = [entry for entry in component.support_items if entry.brief != brief][:MAX_SUPPORT_ITEMS - 1]
            components = list(unit.components)
            components[target] = component.model_copy(update={"support_items": [item, *support]})
            units.append(unit.model_copy(update={"components": components}))
            attached += 1
        result.append(lesson.model_copy(update={"units": units}))
    return result, attached


def with_orientation_note(lesson: IdmLessonDesignV1, phrases: Sequence[str], locale: str) -> IdmLessonDesignV1:
    named = ", ".join(f'"{phrase}"' for phrase in phrases)
    line = (f"Lưu ý: mục tiêu nêu {named} nhưng chưa có unit định hướng liệt kê đủ các thành phần — nên thêm một "
            "unit tổng quan ở đầu mục này." if locale == "vi"
            else f"Note: the objective names {named} but no orientation unit lists all of its items — add an "
                 "overview unit at the start of this section.")
    return lesson.model_copy(update={"notes": sanitize_author_text(lesson.notes + "\n" + line, 2000)})


# --- orchestration -------------------------------------------------------------------------------
def _course_blocks_payload(scope: ModuleScope) -> list[dict[str, Any]]:
    payload = []
    for plan in scope.lesson_plans:
        for block_id in plan.block_ids:
            block, row = scope.blocks[block_id], scope.rows[block_id]
            payload.append(
                {
                    "block_id": block_id,
                    "lesson_key": plan.lesson_key,
                    "name": block.name,
                    "intent": block.intent,
                    "content_kind": block.content_kind,
                    "classification": row.classification,
                    "treatment": row.treatment,
                    "detail_level": row.detail_level,
                    "must_do_ids": row.must_do_ids,
                    "fact_keys": list(block.fact_keys),
                }
            )
    return payload


def _component_orders(lesson: IdmLessonDesignV1, issues: Sequence[IdmIssue]) -> list[str]:
    """The component types of each unit that fails the component order, as "units[i]=html,html,problem"."""

    orders: list[str] = []
    for issue in issues:
        match = _UNIT_PATH_RE.search(issue.path) if issue.code == "IDM_W4_COMPONENT_ORDER" else None
        if match is not None and int(match.group(1)) < len(lesson.units):
            unit = lesson.units[int(match.group(1))]
            orders.append(f"units[{match.group(1)}]=" + ",".join(component.type for component in unit.components))
    return orders


@dataclass
class _Salvage:
    accepted: dict[str, IdmLessonDesignV1] = field(default_factory=dict)
    failures: dict[str, list[IdmIssue]] = field(default_factory=dict)
    # Deterministic layout fixes applied to otherwise failing lessons (``module_autofix``), counted.
    fixes: Counter[str] = field(default_factory=Counter)
    practices: dict[str, list[IdmPracticeTaskV1]] = field(default_factory=dict)
    # lesson_key -> "units[i]=html,html,problem" of each unit still failing the component order after the autofix
    # (QC run ab8d67e1, R7: only CODE@path was logged, so the cause had to be guessed). Server enum values only.
    component_orders: dict[str, list[str]] = field(default_factory=dict)


def _salvage(
    lessons: Sequence[IdmLessonDesignV1],
    scope: ModuleScope,
    allowed: set[str],
    *,
    invalid: Mapping[str, list[IdmIssue]] | None = None,
) -> _Salvage:
    """Validate each lesson independently; return accepted lessons and per-lesson errors.

    A failing lesson first gets the deterministic layout fixes (QC run 8de1c76b, Q1c: two units sharing the
    only block of a lesson, components out of order, an uncovered block ...); it is accepted when they make it
    valid. ``invalid`` are lessons of a rejected answer that did not even parse (their schema paths).
    """

    result = _Salvage()
    by_key = {lesson.lesson_key: lesson for lesson in lessons}
    for position, plan in enumerate(scope.lesson_plans):
        lesson = by_key.get(plan.lesson_key)
        path = f"lessons[{position}]"
        if lesson is None:
            result.failures[plan.lesson_key] = (invalid or {}).get(plan.lesson_key) or [
                IdmIssue("IDM_W3_LESSON_SET_MISMATCH", path)]
            continue
        if lesson.practice_tasks:
            result.practices[plan.lesson_key] = list(lesson.practice_tasks)
        found = validate_lesson(lesson, plan, scope, allowed, path)
        if errors(found):
            fixed, fixes = autofix_lesson(lesson, plan, scope, allowed)
            refound = validate_lesson(fixed, plan, scope, allowed, path) if fixes else found
            if fixes and not errors(refound):
                lesson, found = fixed, refound
                result.fixes.update(fixes)
        if errors(found):
            result.failures[plan.lesson_key] = errors(found)
            orders = _component_orders(lesson, errors(found))
            if orders:
                result.component_orders[plan.lesson_key] = orders
        else:
            normalized = normalize_lesson(
                lesson, plan, scope.locale, [issue.code for issue in found if issue.severity == "warning"]
            )
            try:
                # Normalisation can shorten text below a bound; prove both projections before accepting.
                IdmLessonDesignV1.model_validate(normalized.model_dump(mode="json"))
                LessonArchitectureV2.model_validate(project_lesson(normalized, plan, scope))
            except ValidationError:
                result.failures[plan.lesson_key] = [IdmIssue("IDM_W4_PROJECTION_INVALID", path)]
            else:
                result.accepted[plan.lesson_key] = normalized
    if invalid is None and [lesson.lesson_key for lesson in lessons] != [plan.lesson_key
                                                                         for plan in scope.lesson_plans]:
        result.failures.setdefault("__set__", []).append(IdmIssue("IDM_W3_LESSON_SET_MISMATCH", "lessons"))
    return result


def _parse_rejected_lessons(value: Any, scope: ModuleScope) -> tuple[list[IdmLessonDesignV1], dict[str, list[IdmIssue]],
                                                                     dict[str, list[IdmPracticeTaskV1]]]:
    """The lessons of a schema-rejected W4 answer that are valid on their own (QC run 8de1c76b, Q1b).

    One over-long list in lesson 3 used to discard lessons 1 and 2 too; they are now checked like the lessons
    of an accepted answer. Lessons that do not parse are reported with their position; their valid practice
    tasks still reach the fallback layout.
    """

    raw = value.get("lessons") if isinstance(value, dict) else None
    known = {plan.lesson_key for plan in scope.lesson_plans}
    parsed: list[IdmLessonDesignV1] = []
    invalid: dict[str, list[IdmIssue]] = {}
    practices: dict[str, list[IdmPracticeTaskV1]] = {}
    for position, item in enumerate(raw if isinstance(raw, list) else []):
        key = item.get("lesson_key") if isinstance(item, dict) else None
        try:
            parsed.append(IdmLessonDesignV1.model_validate_json(json.dumps(item, ensure_ascii=False)))
            continue
        except ValidationError:
            if not isinstance(key, str) or key not in known:
                continue
        invalid[key] = [IdmIssue(SCHEMA_INVALID_CODE, f"lessons[{position}]")]
        tasks = item.get("practice_tasks") if isinstance(item, dict) else None
        valid_tasks = []
        for task in tasks if isinstance(tasks, list) else []:
            try:
                valid_tasks.append(IdmPracticeTaskV1.model_validate_json(json.dumps(task, ensure_ascii=False)))
            except ValidationError:
                continue
        if valid_tasks:
            practices[key] = valid_tasks
    return parsed, invalid, practices


def _issue_log(failures: Mapping[str, Sequence[IdmIssue]]) -> dict[str, list[str]]:
    """``CODE@path`` per failing lesson (server-built paths, never content), bounded."""

    logged: dict[str, list[str]] = {}
    budget = _MAX_LOGGED_ISSUES
    for lesson_key, issues in failures.items():
        items = [f"{issue.code}@{issue.path}" for issue in issues][:budget]
        if items:
            logged[lesson_key] = items
            budget -= len(items)
        if budget <= 0:
            break
    return logged


def _log_attempt(runtime: IdmRuntime, plan: ChapterShardPlanV2, *, call: int, kind: str, outcome: str,
                 salvage: _Salvage | None = None, schema_errors: Sequence[dict[str, Any]] = ()) -> None:
    """One line per W4 call (QC run 8de1c76b, Q1d: ch1.l2 failed twice and only the merged codes were logged)."""

    log_stage("idm_module_attempt", {
        "correlation_id": runtime.correlation_id, "stage": "idm_module", "chapter_key": plan.chapter_key,
        "shard_index": plan.shard_index, "call": call, "invocation_kind": kind, "outcome": outcome,
        "schema_errors": [f"{item.get('type')}@{'.'.join(str(part) for part in item.get('loc', []))}"
                          for item in schema_errors][:_MAX_LOGGED_ISSUES],
        "accepted_lessons": sorted(salvage.accepted) if salvage else [],
        "lesson_failures": _issue_log(salvage.failures) if salvage else {},
        "autofix": dict(sorted(salvage.fixes.items())) if salvage else {},
        "component_orders": dict(sorted(salvage.component_orders.items())) if salvage else {},
    })


async def run_idm_module_design(
    *,
    context: IdmModuleContextV1,
    skeleton: CourseSkeletonV2,
    plan: ChapterShardPlanV2,
    facts: Sequence[SourceSnapshotFactV2],
    runtime: IdmRuntime,
) -> dict[str, Any]:
    """Return the ``/chapter-shard`` response for an IDM shard (spec §7.6.6)."""

    started = time.perf_counter()
    scope = build_module_scope(context, plan, facts)
    allowed: set[str] = set(context.allowed_component_types)
    lesson_md = {
        md for lesson in scope.lesson_plans for md in [lesson.primary_must_do_id, *lesson.secondary_must_do_ids] if md
    }
    prompt = module_prompt(
        runtime.locale,
        module_plan={
            "module_key": context.module.module_key,
            "title": context.module.title,
            "performance_goal": context.module.performance_goal,
            "lo_ids": context.module.lo_ids,
        },
        lesson_plans=[lesson.model_dump(mode="json") for lesson in scope.lesson_plans],
        audience=context.target_audience.description,
        objectives=[
            item.model_dump(mode="json")
            for item in context.learning_objectives
            if item.lo_id in set(context.module.lo_ids)
        ]
        or [item.model_dump(mode="json") for item in context.learning_objectives],
        must_dos=[item.model_dump(mode="json") for item in context.must_dos if item.must_do_id in lesson_md],
        course_blocks=_course_blocks_payload(scope),
        facts=[(fact.fact_key, fact.fact_text, None) for fact in facts],
        allowed_components=list(context.allowed_component_types),
        limits=answer_limits(IdmW3W4ModuleResponseV1),
    )
    accepted: dict[str, IdmLessonDesignV1] = {}
    seen_practices: dict[str, list[IdmPracticeTaskV1]] = {}
    codes: Counter[str] = Counter()
    repair = ""
    thinking: ThinkingLevel = THINKING_MODULE
    # Two content attempts (writer + repair) as before, plus at most ONE schema-only repair that does not use
    # one of them (QC run 8de1c76b, Q1b: a list bound rejected the writer answer, so the repair answer was the
    # only semantic check). Worst case three calls, each inside the task deadline (``admitted_output_tokens``).
    content_attempts = 0
    schema_repairs = IDM_MODULE_SCHEMA_REPAIRS
    call = 0
    while content_attempts < IDM_MODULE_CONTENT_ATTEMPTS:
        call += 1
        kind: InvocationKind = "writer" if call == 1 else "repair"
        try:
            response = await idm_call(
                runtime,
                stage="idm_module",
                prompt=prompt + repair,
                response_model=IdmW3W4ModuleResponseV1,
                max_output_tokens=IDM_MODULE_MAX_OUTPUT_TOKENS,
                thinking_level=thinking,
                invocation_kind=kind,
                prepare=lambda value: trim_module_answer(value, scope),
            )
        except (IdmBudgetError, IdmResponseInvalidError) as error:
            codes[error.code] += 1
            if isinstance(error, IdmBudgetError):
                break
            salvage: _Salvage | None = None
            invalid: dict[str, list[IdmIssue]] = {}
            if error.value is not None:
                parsed, invalid, practices = _parse_rejected_lessons(error.value, scope)
                salvage = _salvage(parsed, scope, allowed, invalid=invalid)
                seen_practices.update({**practices, **salvage.practices})
                accepted.update(salvage.accepted)
                runtime.adjustments.update(salvage.fixes)
                codes.update(issue.code for key, items in salvage.failures.items() if key not in invalid
                             for issue in items)
            _log_attempt(runtime, plan, call=call, kind=kind, outcome=error.code.lower(), salvage=salvage,
                         schema_errors=error.errors)
            if salvage is not None and all(item.lesson_key in accepted for item in scope.lesson_plans):
                break
            content = [] if salvage is None else [issue.as_repair_item() for lesson_key, items
                                                  in salvage.failures.items() if lesson_key not in invalid
                                                  for issue in items]
            repair = (answer_repair(error.code, error.errors, COMPACT_MODULE) if not content
                      else repair_suffix([*({"code": str(item["type"]),
                                             "path": ".".join(str(part) for part in item["loc"])}
                                            for item in error.errors), *content]))
            thinking = repair_thinking(error.code, thinking)
            if error.code == SCHEMA_INVALID_CODE and schema_repairs > 0:
                schema_repairs -= 1
            else:
                content_attempts += 1
            continue
        except IdmProviderError as error:
            if error.terminal:
                raise
            codes[error.code] += 1
            break
        content_attempts += 1
        salvage = _salvage(response.lessons, scope, allowed)
        seen_practices.update(salvage.practices)
        accepted.update(salvage.accepted)
        runtime.adjustments.update(salvage.fixes)
        codes.update(issue.code for items in salvage.failures.values() for issue in items)
        _log_attempt(runtime, plan, call=call, kind=kind, outcome="validated", salvage=salvage)
        missing = [lesson for lesson in scope.lesson_plans if lesson.lesson_key not in accepted]
        if not missing:
            break
        repair = repair_suffix([issue.as_repair_item() for items in salvage.failures.values() for issue in items])
    final: list[IdmLessonDesignV1] = []
    fallback_count = 0
    for lesson_plan in scope.lesson_plans:
        if lesson_plan.lesson_key in accepted:
            final.append(accepted[lesson_plan.lesson_key])
        else:
            fallback_count += 1
            final.append(
                fallback_lesson(lesson_plan, scope, provider_practices=seen_practices.get(lesson_plan.lesson_key, ()))
            )
    if fallback_count:
        record_deterministic_fallback(runtime, stage="idm_module", code="IDM_MODULE_LESSON_FALLBACK")
    final, attached = attach_framework_items(final, scope)
    if attached:
        runtime.adjustments["IDM_W4_FRAMEWORK_ITEMS_ATTACHED"] += attached
    if plan.shard_index == 0:
        # The chapter's practice questions get their key letters in turn (QC run 8de1c76b, Q4); a later shard does
        # not know its lessons' place in the chapter and keeps the plain per-unit letters.
        final, renumbered = encode_key_letters(final, plan.order, context.lesson_index_offset)
        if renumbered:
            runtime.adjustments["IDM_W4_KEY_LETTERS_ENCODED"] += renumbered
    gaps = framework_orientation_gaps(final, scope)
    if gaps:
        codes["IDM_W4_FRAMEWORK_ORIENTATION_MISSING"] += sum(len(phrases) for phrases in gaps.values())
        final = [with_orientation_note(lesson, gaps[lesson.lesson_key], scope.locale)
                 if lesson.lesson_key in gaps else lesson for lesson in final]
    stage_origin: StageOrigin = (
        "provider"
        if fallback_count == 0
        else "deterministic_fallback"
        if fallback_count == len(scope.lesson_plans)
        else "partial_fallback"
    )
    design_payload: dict[str, Any] = {
        "pipeline_version": "idm-1",
        "chapter_key": plan.chapter_key,
        "shard_index": plan.shard_index,
        "lessons": [lesson.model_dump(mode="json") for lesson in final],
        "lesson_index_offset": context.lesson_index_offset,
        "stage_origin": stage_origin,
    }
    design_payload["design_hash"] = design_hash_of(design_payload)
    shard_design = IdmShardDesignV1.model_validate(design_payload)
    shard = project_shard(final, scope.lesson_plans, scope, skeleton=skeleton, plan=plan, shard_design=shard_design)
    usage_complete = runtime.usage.complete and runtime.provider_failure_code is None
    log_stage(
        "idm_stage_completed",
        {
            "correlation_id": runtime.correlation_id,
            "stage": "idm_module",
            "chapter_key": plan.chapter_key,
            "shard_index": plan.shard_index,
            "lesson_count": len(final),
            "fallback_lessons": fallback_count,
            "call_count": runtime.usage.calls,
            "duration_ms": int((time.perf_counter() - started) * _MS),
            "usage": runtime.usage.as_usage(),
            "error_codes": dict(sorted(codes.items())),
            "response_adjustments": dict(sorted(runtime.adjustments.items())),
            "obligation_count": len(shard["assessment_obligations"]),
        },
    )
    return {
        "contract_version": 2,
        "shard": shard,
        "usage": runtime.usage.as_usage(),
        "usage_complete": usage_complete,
        "usage_source": "provider" if usage_complete else "reserved_upper_bound",
        "content_origin": "provider_validated" if stage_origin == "provider" else "structured_fallback",
        "quality_state": "validated" if stage_origin == "provider" else "review_required",
    }
