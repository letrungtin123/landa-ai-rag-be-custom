"""IDM ``chapter_blueprint`` task: W3 treatment + W4 lesson layout (spec §7.6, §8.2).

One call designs every lesson of a shard. Valid lessons are kept; a lesson that
still fails after one repair gets a deterministic layout of its own (partial
salvage), so a single bad lesson never discards the shard.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from typing import Any, Final

from pydantic import ValidationError

from app.idm.contracts import (
    IdmComponentDesignV1,
    IdmLessonDesignV1,
    IdmLessonPlanV1,
    IdmModuleContextV1,
    IdmPracticeTaskV1,
    IdmShardDesignV1,
    IdmW3W4ModuleResponseV1,
    StageOrigin,
    design_hash_of,
)
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
    IDM_MODULE_MAX_OUTPUT_TOKENS,
    IDM_THEORY_RUN_MAX_COMPONENTS,
    THINKING_MODULE,
)
from app.idm.prompts import COMPACT_MODULE, answer_repair, module_prompt, repair_suffix
from app.idm.runtime import (
    IdmBudgetError,
    IdmProviderError,
    IdmResponseInvalidError,
    IdmRuntime,
    ThinkingLevel,
    idm_call,
    log_stage,
    record_deterministic_fallback,
    repair_thinking,
)
from app.idm.signals import idm_has_ordered_steps, idm_relationship_pairs, idm_term_definitions
from app.idm.text import idm_fold, sanitize_author_text, single_line
from app.idm.validation import IdmIssue, errors
from app.lesson_author_orchestration_v2 import (
    ArchitectureComponentAuthorReviewV2,
    CourseSkeletonV2,
    LessonArchitectureV2,
)
from app.lesson_author_orchestration_v2_provider import ChapterShardPlanV2, SourceSnapshotFactV2

PRACTICE_COMPONENT_TYPES: Final = frozenset({"problem", "la_sortable", "la_crossword"})
_MIN_TERM_DEFINITIONS: Final = 3
_MS: Final = 1000
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
            if not _format_has_evidence(component, scope):
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
            else:
                theory_run += 1
                if theory_run > IDM_THEORY_RUN_MAX_COMPONENTS:
                    issues.append(IdmIssue("IDM_W4_THEORY_RUN", component_path, "warning"))
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


def _format_has_evidence(component: IdmComponentDesignV1, scope: ModuleScope) -> bool:
    texts = scope.block_texts([block_id for block_id in component.block_ids if block_id in scope.blocks])
    if component.type == "la_sortable":
        return idm_has_ordered_steps(texts, locale=scope.locale)
    if component.type == "la_crossword":
        return len(idm_term_definitions(texts)) >= _MIN_TERM_DEFINITIONS
    if component.type == "la_diagram":
        return bool(idm_relationship_pairs(texts))
    return True


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
    extra = _warning_notes(warnings, locale)
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


def _warning_notes(warnings: Sequence[str], locale: str) -> str:
    lines = []
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


def _salvage(
    response: IdmW3W4ModuleResponseV1,
    scope: ModuleScope,
    allowed: set[str],
) -> tuple[dict[str, IdmLessonDesignV1], dict[str, list[IdmIssue]]]:
    """Validate each lesson independently; return accepted lessons and per-lesson errors."""

    accepted: dict[str, IdmLessonDesignV1] = {}
    failures: dict[str, list[IdmIssue]] = {}
    by_key = {lesson.lesson_key: lesson for lesson in response.lessons}
    for position, plan in enumerate(scope.lesson_plans):
        lesson = by_key.get(plan.lesson_key)
        path = f"lessons[{position}]"
        if lesson is None:
            failures[plan.lesson_key] = [IdmIssue("IDM_W3_LESSON_SET_MISMATCH", path)]
            continue
        found = validate_lesson(lesson, plan, scope, allowed, path)
        if errors(found):
            failures[plan.lesson_key] = errors(found)
        else:
            normalized = normalize_lesson(
                lesson, plan, scope.locale, [issue.code for issue in found if issue.severity == "warning"]
            )
            try:
                # Normalisation can shorten text below a bound; prove both projections before accepting.
                IdmLessonDesignV1.model_validate(normalized.model_dump(mode="json"))
                LessonArchitectureV2.model_validate(project_lesson(normalized, plan, scope))
            except ValidationError:
                failures[plan.lesson_key] = [IdmIssue("IDM_W4_PROJECTION_INVALID", path)]
            else:
                accepted[plan.lesson_key] = normalized
    if [lesson.lesson_key for lesson in response.lessons] != [plan.lesson_key for plan in scope.lesson_plans]:
        failures.setdefault("__set__", []).append(IdmIssue("IDM_W3_LESSON_SET_MISMATCH", "lessons"))
    return accepted, failures


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
    )
    accepted: dict[str, IdmLessonDesignV1] = {}
    seen_practices: dict[str, list[IdmPracticeTaskV1]] = {}
    codes: Counter[str] = Counter()
    repair = ""
    thinking: ThinkingLevel = THINKING_MODULE
    for attempt in (1, 2):
        try:
            response = await idm_call(
                runtime,
                stage="idm_module",
                prompt=prompt + repair,
                response_model=IdmW3W4ModuleResponseV1,
                max_output_tokens=IDM_MODULE_MAX_OUTPUT_TOKENS,
                thinking_level=thinking,
                invocation_kind="writer" if attempt == 1 else "repair",
            )
        except (IdmBudgetError, IdmResponseInvalidError) as error:
            codes[error.code] += 1
            if isinstance(error, IdmBudgetError):
                break
            repair = answer_repair(error.code, error.errors, COMPACT_MODULE)
            thinking = repair_thinking(error.code, thinking)
            continue
        except IdmProviderError as error:
            if error.terminal:
                raise
            codes[error.code] += 1
            break
        lessons, failures = _salvage(response, scope, allowed)
        for lesson in response.lessons:
            if lesson.practice_tasks:
                seen_practices[lesson.lesson_key] = list(lesson.practice_tasks)
        accepted.update(lessons)
        codes.update(issue.code for items in failures.values() for issue in items)
        missing = [lesson for lesson in scope.lesson_plans if lesson.lesson_key not in accepted]
        if not missing:
            break
        repair = repair_suffix([issue.as_repair_item() for items in failures.values() for issue in items])
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
