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

from app.idm.contracts import (
    IdmLessonDesignV1,
    IdmModuleContextV1,
    IdmPracticeTaskV1,
    IdmShardDesignV1,
    IdmW3W4ModuleResponseV1,
    StageOrigin,
    design_hash_of,
)
from app.idm.faq_plan import FAQ_ADDED_CODE, FAQ_DROPPED_CODE, FAQ_PLANNED_CODE, gate_and_plan_faq
from app.idm.fidelity import TEACHING_SUPPORT_CODE, module_fidelity_rules, plan_teaching_supports, purpose_mode
from app.idm.limits import answer_limits
from app.idm.mcq import assign_key_letters
from app.idm.module_autofix import trim_module_answer
from app.idm.module_design_framework import (
    _taught_before_questions,
    attach_framework_items,
    attach_purpose_framework,
    framework_orientation_gaps,
    with_orientation_note,
)
from app.idm.module_design_lessons import PRACTICE_COMPONENT_TYPES, _brief_text, normalize_lesson, validate_lesson
from app.idm.module_design_salvage import (
    _course_blocks_payload,
    _log_attempt,
    _parse_rejected_lessons,
    _Salvage,
    _salvage,
)
from app.idm.module_layout import build_module_scope, fallback_lesson, project_shard
from app.idm.policy import (
    IDM_MODULE_CONTENT_ATTEMPTS,
    IDM_MODULE_MAX_OUTPUT_TOKENS,
    IDM_MODULE_SCHEMA_REPAIRS,
    THINKING_MODULE,
)
from app.idm.prompts import COMPACT_MODULE, answer_repair, module_prompt, repair_suffix
from app.idm.purpose import module_purpose_rules, purpose_brief
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
from app.idm.titles import TitleCaser, sentence_case_lessons
from app.idm.validation import IdmIssue
from app.lesson_author_orchestration_v2 import CourseSkeletonV2
from app.lesson_author_orchestration_v2_provider import ChapterShardPlanV2, SourceSnapshotFactV2

_MS: Final = 1000
_MAX_ACTIVITIES: Final = 3
_MAX_UNITS_PER_LESSON: Final = 12


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
    # QLT-5 C: the design's content mode (structured for a design without one).
    mode = purpose_mode(context.purpose)
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
        # QLT-5: the design's purpose keeps the lessons on what the course is for (none for a contract-1 design).
        purpose_rules=module_purpose_rules(context.purpose), course_purpose=purpose_brief(context.purpose),
        fidelity_rules=module_fidelity_rules(mode, scope.locale),
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
                parsed, invalid, practices = await runtime.run_cpu(_parse_rejected_lessons, error.value, scope)
                salvage = await runtime.run_cpu(_salvage, parsed, scope, allowed, invalid=invalid)
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
        salvage = await runtime.run_cpu(_salvage, response.lessons, scope, allowed)
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
    final = _taught_before_questions(final, scope, runtime, codes)
    # QLT-5 D: an FAQ only where its unit has something to clarify, and a plan for each of its items.
    faq = await runtime.run_cpu(gate_and_plan_faq, final, scope)
    final = faq.lessons
    for code, count in ((FAQ_DROPPED_CODE, faq.dropped), (FAQ_PLANNED_CODE, faq.planned), (FAQ_ADDED_CODE, faq.added)):
        if count:
            runtime.adjustments[code] += count
    final, attached = attach_framework_items(final, scope)
    if attached:
        runtime.adjustments["IDM_W4_FRAMEWORK_ITEMS_ATTACHED"] += attached
    # QLT-9: a framework the course purpose names is presented whole in the module that numbers its items.
    project = context.project_context
    final, purposed = attach_purpose_framework(final, scope, [text for text in (
        project.course_purpose, project.course_description) if text])
    if purposed:
        runtime.adjustments["IDM_W4_PURPOSE_FRAMEWORK_ATTACHED"] += 1
    # Explain mode: every teaching html plans an example, a comparison or why it matters (AI-drafted when needed).
    final, supports = plan_teaching_supports(final, {block_id: block.name for block_id, block in scope.blocks.items()},
                                             mode, scope.locale)
    if supports:
        runtime.adjustments[TEACHING_SUPPORT_CODE] += supports
    # QLT-7: lesson, unit and component titles in sentence case (a lesson keeps its W4 title when only the case
    # differs), before the design is hashed and projected, so both name every title alike.
    caser = TitleCaser.from_texts([*scope.fact_text.values(), *(block.name for block in scope.blocks.values()),
                                   *(block.summary for block in scope.blocks.values()),
                                   *scope.must_do_statement.values()])
    final, retitled = sentence_case_lessons(final, caser, locale=scope.locale,
                                            plan_titles={item.lesson_key: item.title for item in scope.lesson_plans})
    if retitled:
        runtime.adjustments["IDM_W4_TITLES_SENTENCE_CASE"] += retitled
    # Every single-choice key gets its letter (QC run 8de1c76b, Q4) in a balanced random order seeded by the source
    # (QLT-9, owner 2026-10-16: no A B C D cycle, never all B or C). Shards design in parallel and do not know how many
    # questions came before them: a shard starts where one question per lesson before it would end. Replayed on 5
    # stored courses, each letter keys 17-33% of the questions (starting each shard on a block of its own: 8-35%).
    final, keyed = assign_key_letters(final, skeleton.source_snapshot_hash, context.lesson_index_offset)
    if keyed:
        runtime.adjustments["IDM_W4_KEY_LETTERS_ASSIGNED"] += keyed
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


# The names other modules and the tests import from this path; the lesson checks, the framework steps and the
# salvage moved to module_design_lessons, module_design_framework and module_design_salvage.
__all__ = [
    "PRACTICE_COMPONENT_TYPES", "_brief_text", "attach_framework_items", "attach_purpose_framework", "normalize_lesson",
    "run_idm_module_design", "validate_lesson", "with_orientation_note",
]
