"""W4 course architecture, deterministic assembly and skeleton projection (spec §7.5, §8.1)."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.idm.contracts import (
    IdmAudienceV1,
    IdmAuthorNotesV1,
    IdmBlockLoLinkV1,
    IdmBlockScopeV1,
    IdmBlueprintRowV1,
    IdmContentBlockV1,
    IdmCourseDesignV1,
    IdmDispositionV1,
    IdmHoldItemV1,
    IdmLearningObjectiveV1,
    IdmLessonPlanV1,
    IdmModulePlanV1,
    IdmMustDoV1,
    IdmProjectContextV1,
    IdmW4CourseResponseV1,
    StageOrigin,
    design_hash_of,
)
from app.idm.notes import build_course_notes, build_lesson_notes, build_module_notes
from app.idm.policy import (
    IDM_CONTRACT_VERSION,
    IDM_DURATION_OVER_TARGET_RATIO,
    IDM_FALLBACK_LESSON_MINUTES,
    IDM_FALLBACK_LESSON_SCREENS,
    IDM_MAX_LESSON_BLOCKS,
    IDM_MAX_LESSONS,
    IDM_MAX_LESSONS_PER_MODULE,
    IDM_MAX_MODULES,
    IDM_PIPELINE_VERSION,
    IDM_PROMPT_POLICY_VERSION,
    IDM_SCOPE_KEY_HEX_CHARS,
    IDM_SCOPE_KEY_PREFIX,
    IDM_SHARD_MAX_SOURCE_CHARS,
    IDM_W4_COURSE_MAX_OUTPUT_TOKENS,
    THINKING_W4_COURSE,
)
from app.idm.prompts import COMPACT_W4, answer_repair, repair_suffix, w4_course_prompt
from app.idm.runtime import (
    IdmBudgetError,
    IdmProviderError,
    IdmResponseInvalidError,
    IdmRuntime,
    IdmStageError,
    ThinkingLevel,
    idm_call,
    record_deterministic_fallback,
    repair_thinking,
)
from app.idm.text import is_generic_title, must_do_title, objective_title, single_line
from app.idm.validation import IdmIssue, errors
from app.lesson_author_orchestration_v2 import CourseSkeletonChapterV2, CourseSkeletonV2

_AI_SUFFIX_VI: Final = " (Đề xuất bởi AI — cần xác nhận)"
_AI_SUFFIX_EN: Final = " (Proposed by AI — please confirm)"
_MAX_CHAPTER_OUTCOMES: Final = 12
_MAX_PREREQUISITES: Final = 10
_MAX_SECONDARY_MUST_DOS: Final = 2


@dataclass
class ArchitectureResult:
    plan: IdmW4CourseResponseV1
    origin: StageOrigin
    codes: dict[str, int]
    warnings: list[str]


def block_scope_key(source_snapshot_hash: str, block_id: str, fact_keys: Sequence[str]) -> str:
    material = "\x1e".join([source_snapshot_hash, block_id, ",".join(sorted(fact_keys))])
    return IDM_SCOPE_KEY_PREFIX + hashlib.sha256(material.encode("utf-8")).hexdigest()[:IDM_SCOPE_KEY_HEX_CHARS]


def _course_and_reference_blocks(
    blocks: Sequence[IdmContentBlockV1], rows: Sequence[IdmBlueprintRowV1],
) -> tuple[list[IdmContentBlockV1], list[IdmContentBlockV1]]:
    row_by_id = {row.block_id: row for row in rows}
    course = [block for block in blocks if row_by_id[block.block_id].placement == "course"
              and not row_by_id[block.block_id].hold]
    reference = [block for block in blocks if row_by_id[block.block_id].placement == "reference_job_aid"]
    return course, reference


def validate_w4(
    response: IdmW4CourseResponseV1,
    *,
    course_blocks: Sequence[IdmContentBlockV1],
    reference_blocks: Sequence[IdmContentBlockV1],
    must_dos: Sequence[IdmMustDoV1],
    blocked: Sequence[str],
    objectives: Sequence[IdmLearningObjectiveV1],
    duration_target_minutes: int | None,
    block_chars: Mapping[str, int] | None = None,
) -> list[IdmIssue]:
    issues: list[IdmIssue] = []
    sizes = block_chars or {}
    course_ids = {block.block_id for block in course_blocks}
    reference_ids = {block.block_id for block in reference_blocks}
    lo_ids = {objective.lo_id for objective in objectives}
    md_ids = {must_do.must_do_id for must_do in must_dos}
    lessons = [lesson for module in response.modules for lesson in module.lessons]
    if (len(response.modules) > IDM_MAX_MODULES or len(lessons) > IDM_MAX_LESSONS
            or any(len(module.lessons) > IDM_MAX_LESSONS_PER_MODULE for module in response.modules)):
        issues.append(IdmIssue("IDM_W4_LIMIT", "modules"))
    if ([module.module_key for module in response.modules]
            != [f"mod_{position:02d}" for position in range(1, len(response.modules) + 1)]
            or [lesson.lesson_key for lesson in lessons]
            != [f"lsn_{position:03d}" for position in range(1, len(lessons) + 1)]):
        issues.append(IdmIssue("IDM_W4_KEY_SEQUENCE", "modules"))
    placed: Counter[str] = Counter()
    must_do_use: Counter[str] = Counter()
    for module_position, module in enumerate(response.modules):
        module_path = f"modules[{module_position}]"
        if is_generic_title(module.title):
            issues.append(IdmIssue("IDM_W4_TITLE_GENERIC", f"{module_path}.title"))
        if any(lo_id not in lo_ids for lo_id in module.lo_ids):
            issues.append(IdmIssue("IDM_W4_LO_UNKNOWN", f"{module_path}.lo_ids"))
        for lesson_position, lesson in enumerate(module.lessons):
            path = f"{module_path}.lessons[{lesson_position}]"
            if is_generic_title(lesson.title):
                issues.append(IdmIssue("IDM_W4_TITLE_GENERIC", f"{path}.title"))
            placed.update(lesson.block_ids)
            if sum(sizes.get(block_id, 0) for block_id in lesson.block_ids) > IDM_SHARD_MAX_SOURCE_CHARS:
                issues.append(IdmIssue("IDM_W4_LESSON_TOO_LARGE", f"{path}.block_ids"))
            if any(block_id not in course_ids and block_id not in reference_ids for block_id in lesson.block_ids):
                issues.append(IdmIssue("IDM_W4_BLOCK_UNKNOWN", f"{path}.block_ids"))
            ids = [lesson.primary_must_do_id, *lesson.secondary_must_do_ids] if lesson.primary_must_do_id else list(
                lesson.secondary_must_do_ids)
            must_do_use.update(ids)
            if any(md not in md_ids for md in ids):
                issues.append(IdmIssue("IDM_W4_MUST_DO_UNPLACED", f"{path}.primary_must_do_id"))
            if lesson.kind == "learning":
                if lesson.primary_must_do_id is None:
                    issues.append(IdmIssue("IDM_W4_MUST_DO_UNPLACED", f"{path}.primary_must_do_id"))
                if any(block_id in reference_ids for block_id in lesson.block_ids):
                    issues.append(IdmIssue("IDM_W4_REFERENCE_IN_LEARNING_LESSON", f"{path}.block_ids"))
            elif (lesson.primary_must_do_id is not None or lesson.secondary_must_do_ids
                  or any(block_id in course_ids for block_id in lesson.block_ids)):
                issues.append(IdmIssue("IDM_W4_JOB_AID_LESSON_INVALID", path))
    if any(count > 1 for count in placed.values()):
        issues.append(IdmIssue("IDM_W4_BLOCK_DUPLICATED", "modules"))
    if any(block_id not in placed for block_id in course_ids | reference_ids):
        issues.append(IdmIssue("IDM_W4_BLOCK_UNPLACED", "modules"))
    blocked_set = set(blocked)
    if any(count > 1 for count in must_do_use.values()):
        issues.append(IdmIssue("IDM_W4_MUST_DO_DUPLICATED", "modules"))
    if any(md not in must_do_use for md in md_ids - blocked_set):
        issues.append(IdmIssue("IDM_W4_MUST_DO_UNPLACED", "modules"))
    if any(md in blocked_set for md in must_do_use):
        issues.append(IdmIssue("IDM_W4_MUST_DO_UNPLACED", "modules"))
    total_minutes = sum(lesson.est_minutes for lesson in lessons)
    if duration_target_minutes and total_minutes > IDM_DURATION_OVER_TARGET_RATIO * duration_target_minutes:
        issues.append(IdmIssue("IDM_W4_DURATION_OVER_TARGET", "modules", "warning"))
    return list(dict.fromkeys(issues))


def _normalize_w4(response: IdmW4CourseResponseV1) -> IdmW4CourseResponseV1:
    return response.model_copy(update={
        "course_title": single_line(response.course_title, 500),
        "course_summary": single_line(response.course_summary, 2000),
        "assessment_strategy": single_line(response.assessment_strategy, 2000),
        "prerequisites": [single_line(item, 300) for item in response.prerequisites][:_MAX_PREREQUISITES],
        "modules": [module.model_copy(update={
            "title": single_line(module.title, 500),
            "performance_goal": single_line(module.performance_goal, 2000),
            "lessons": [lesson.model_copy(update={
                "title": single_line(lesson.title, 180),
                "ordering_rationale": single_line(lesson.ordering_rationale, 300),
            }) for lesson in module.lessons],
        }) for module in response.modules],
    })


def fallback_w4(
    *,
    context: IdmProjectContextV1,
    objectives: Sequence[IdmLearningObjectiveV1],
    must_dos: Sequence[IdmMustDoV1],
    rows: Sequence[IdmBlueprintRowV1],
    course_blocks: Sequence[IdmContentBlockV1],
    reference_blocks: Sequence[IdmContentBlockV1],
    blocked: Sequence[str],
    block_chars: Mapping[str, int] | None = None,
) -> IdmW4CourseResponseV1:
    """One module per objective, one lesson per Must Do, Job Aid last (spec §7.5)."""

    vi = context.locale == "vi"
    row_by_id = {row.block_id: row for row in rows}
    blocked_set = set(blocked)
    active = [must_do for must_do in must_dos if must_do.must_do_id not in blocked_set]
    owner: dict[str, str] = {}
    for block in course_blocks:
        candidates = [md for md in row_by_id[block.block_id].must_do_ids if md not in blocked_set]
        owner[block.block_id] = candidates[0] if candidates else (active[0].must_do_id if active else "")
    blocks_by_md: dict[str, list[str]] = {}
    for block in course_blocks:
        blocks_by_md.setdefault(owner[block.block_id], []).append(block.block_id)
    lessons_by_lo: dict[str, list[tuple[IdmMustDoV1, list[str], list[str]]]] = {}
    orphan_mds: list[str] = []
    for must_do in active:
        block_ids = blocks_by_md.get(must_do.must_do_id, [])
        if block_ids:
            lessons_by_lo.setdefault(must_do.lo_id, []).append((must_do, block_ids, []))
        else:
            orphan_mds.append(must_do.must_do_id)
    first_lessons = [lesson for lo_lessons in lessons_by_lo.values() for lesson in lo_lessons]
    for md in orphan_mds:
        # A Must Do whose blocks were claimed by an earlier one shares that lesson.
        target = next((lesson for lesson in first_lessons if len(lesson[2]) < _MAX_SECONDARY_MUST_DOS), None)
        if target is not None:
            target[2].append(md)
    modules: list[IdmModulePlanV1] = []
    lesson_number = 0
    for objective in objectives:
        lo_lessons = lessons_by_lo.get(objective.lo_id, [])
        if not lo_lessons:
            continue
        plans = []
        for must_do, block_ids, secondary in lo_lessons:
            # A lesson holds at most 40 blocks and one shard of source; a larger Must Do continues.
            chunks = _lesson_chunks(block_ids, block_chars or {})
            for part, chunk in enumerate(chunks, start=1):
                lesson_number += 1
                suffix = "" if part == 1 else (f" (phần {part})" if vi else f" (part {part})")
                plans.append(IdmLessonPlanV1(
                    lesson_key=f"lsn_{lesson_number:03d}", kind="learning",
                    title=(single_line(must_do_title(must_do.statement), 180 - len(suffix)) + suffix).ljust(3, "."),
                    primary_must_do_id=must_do.must_do_id, secondary_must_do_ids=secondary if part == 1 else [],
                    block_ids=chunk, est_screens=IDM_FALLBACK_LESSON_SCREENS,
                    est_minutes=IDM_FALLBACK_LESSON_MINUTES,
                    ordering_rationale=("Theo thứ tự mục tiêu học tập." if vi else "Follows the objective order."),
                ))
        for start in range(0, len(plans), IDM_MAX_LESSONS_PER_MODULE):
            modules.append(IdmModulePlanV1(
                module_key=f"mod_{len(modules) + 1:02d}",
                # A module title is a topic, not the objective sentence ("Người học có thể áp dụng …").
                title=single_line(objective_title(objective.statement), 500).ljust(3, "."),
                performance_goal=single_line(objective.statement, 2000).ljust(10, "."),
                lo_ids=[objective.lo_id], lessons=plans[start:start + IDM_MAX_LESSONS_PER_MODULE],
            ))
    reference_chunks = _lesson_chunks([block.block_id for block in reference_blocks], block_chars or {})
    for part, reference_chunk in enumerate(reference_chunks, start=1):
        lesson_number += 1
        title = "Tài liệu tra cứu nhanh" if vi else "Quick reference"
        job_aid = IdmLessonPlanV1(
            lesson_key=f"lsn_{lesson_number:03d}", kind="job_aid",
            title=title if part == 1 else f"{title} ({part})", primary_must_do_id=None,
            secondary_must_do_ids=[], block_ids=reference_chunk,
            est_screens=IDM_FALLBACK_LESSON_SCREENS, est_minutes=IDM_FALLBACK_LESSON_MINUTES,
            ordering_rationale=("Tra cứu khi làm việc, đặt cuối khoá." if vi else "On-the-job lookup, placed last."),
        )
        if modules and len(modules[-1].lessons) < IDM_MAX_LESSONS_PER_MODULE:
            last = modules[-1]
            modules[-1] = last.model_copy(update={"lessons": [*last.lessons, job_aid]})
        else:
            modules.append(IdmModulePlanV1(
                module_key=f"mod_{len(modules) + 1:02d}", title=title, performance_goal=single_line(
                    "Tra cứu đúng thông tin khi làm việc." if vi else "Look up the right information at work.", 2000),
                lo_ids=[objectives[0].lo_id], lessons=[job_aid]))
    if not modules:
        raise IdmStageError("IDM_COURSE_HAS_NO_TEACHABLE_CONTENT")
    topic = context.course_title_hint or context.source_documents[0].name
    return IdmW4CourseResponseV1(
        course_title=single_line(topic, 500).ljust(3, "."),
        course_summary=single_line((f"Khoá học giúp người học thực hiện đúng các công việc trong {topic}." if vi
                                    else f"This course helps learners correctly perform the work in {topic}."),
                                   2000).ljust(20, "."),
        # A promise ("every section has a practice") the fallback cannot check (QC 234653, D14).
        assessment_strategy=("Bài luyện tập của từng mục được thiết kế ở bước thiết kế bài học; mục chưa có bài "
                             "luyện tập được đánh dấu để tác giả bổ sung." if vi
                             else "Each section's practice is designed with its lessons; a section without a "
                                  "practice is flagged for the author to add one."),
        prerequisites=[], modules=modules,
    )


def _lesson_chunks(block_ids: Sequence[str], block_chars: Mapping[str, int]) -> list[list[str]]:
    """Split blocks into lessons of at most 40 blocks and one shard of source characters.

    A single block larger than a shard cannot be placed; Node would reject it after the paid
    design call, so the stage fails here with the same code.
    """

    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for block_id in block_ids:
        size = block_chars.get(block_id, 0)
        if size > IDM_SHARD_MAX_SOURCE_CHARS:
            raise IdmStageError("IDM_LESSON_EXCEEDS_SHARD")
        if current and (len(current) >= IDM_MAX_LESSON_BLOCKS or current_chars + size > IDM_SHARD_MAX_SOURCE_CHARS):
            chunks.append(current)
            current, current_chars = [], 0
        current.append(block_id)
        current_chars += size
    if current:
        chunks.append(current)
    return chunks


async def run_w4(
    runtime: IdmRuntime,
    *,
    context: IdmProjectContextV1,
    audience: IdmAudienceV1,
    objectives: Sequence[IdmLearningObjectiveV1],
    must_dos: Sequence[IdmMustDoV1],
    blocks: Sequence[IdmContentBlockV1],
    rows: Sequence[IdmBlueprintRowV1],
    blocked: Sequence[str],
    block_chars: Mapping[str, int] | None = None,
) -> ArchitectureResult:
    course_blocks, reference_blocks = _course_and_reference_blocks(blocks, rows)
    row_by_id = {row.block_id: row for row in rows}

    def describe(block: IdmContentBlockV1) -> dict[str, Any]:
        row = row_by_id[block.block_id]
        return {"block_id": block.block_id, "name": block.name, "intent": block.intent,
                "content_kind": block.content_kind, "classification": row.classification,
                "detail_level": row.detail_level, "must_do_ids": row.must_do_ids,
                "fact_count": len(block.fact_keys), "content_chars": (block_chars or {}).get(block.block_id, 0)}

    prompt = w4_course_prompt(
        runtime.locale, project_context=context.model_dump(mode="json"), audience=audience.model_dump(mode="json"),
        objectives=[item.model_dump(mode="json") for item in objectives],
        must_dos=[item.model_dump(mode="json") for item in must_dos],
        course_blocks=[describe(block) for block in course_blocks],
        reference_blocks=[describe(block) for block in reference_blocks],
        blocked_must_dos=list(blocked), duration_target_minutes=context.duration_target_minutes,
    )
    codes: Counter[str] = Counter()
    repair = ""
    thinking: ThinkingLevel = THINKING_W4_COURSE
    if course_blocks:
        for attempt in (1, 2):
            try:
                response = await idm_call(
                    runtime, stage="idm_w4", prompt=prompt + repair, response_model=IdmW4CourseResponseV1,
                    max_output_tokens=IDM_W4_COURSE_MAX_OUTPUT_TOKENS, thinking_level=thinking,
                    invocation_kind="writer" if attempt == 1 else "repair",
                )
            except (IdmBudgetError, IdmResponseInvalidError) as error:
                codes[error.code] += 1
                if isinstance(error, IdmBudgetError):
                    break
                repair = answer_repair(error.code, error.errors, COMPACT_W4)
                thinking = repair_thinking(error.code, thinking)
                continue
            except IdmProviderError as error:
                if error.terminal:
                    raise
                codes[error.code] += 1
                break
            found = validate_w4(response, course_blocks=course_blocks, reference_blocks=reference_blocks,
                                must_dos=must_dos, blocked=blocked, objectives=objectives,
                                duration_target_minutes=context.duration_target_minutes,
                                block_chars=block_chars)
            codes.update(issue.code for issue in found)
            if not errors(found):
                return ArchitectureResult(_normalize_w4(response), "provider", dict(codes),
                                          [issue.code for issue in found if issue.severity == "warning"])
            repair = repair_suffix([issue.as_repair_item() for issue in errors(found)])
    record_deterministic_fallback(runtime, stage="idm_w4", code="IDM_W4_FALLBACK")
    plan = fallback_w4(context=context, objectives=objectives, must_dos=must_dos, rows=rows,
                       course_blocks=course_blocks, reference_blocks=reference_blocks, blocked=blocked,
                       block_chars=block_chars)
    return ArchitectureResult(plan, "deterministic_fallback", dict(codes), [])


@dataclass
class AssemblyInput:
    source_snapshot_hash: str
    context: IdmProjectContextV1
    snapshot_fact_keys: Sequence[str]
    fact_text: dict[str, str]
    fact_source_ref: dict[str, str | None]
    audience: IdmAudienceV1
    objectives: Sequence[IdmLearningObjectiveV1]
    must_dos: Sequence[IdmMustDoV1]
    blocks: Sequence[IdmContentBlockV1]
    links: Sequence[IdmBlockLoLinkV1]
    rows: Sequence[IdmBlueprintRowV1]
    blocked: Sequence[str]
    holds: Sequence[IdmHoldItemV1]
    plan: IdmW4CourseResponseV1
    dispositions: Sequence[IdmDispositionV1]
    stage_origins: dict[str, StageOrigin]


def assemble_idm_course_design(data: AssemblyInput) -> IdmCourseDesignV1:
    """Build block scopes, check accounting, write notes and hash the design."""

    by_block = {block.block_id: block for block in data.blocks}
    disposition_keys = [item.fact_key for item in data.dispositions]
    if (len(disposition_keys) != len(set(disposition_keys))
            or set(disposition_keys) != set(data.snapshot_fact_keys)
            or len(disposition_keys) != len(data.snapshot_fact_keys)):
        raise IdmStageError("IDM_ACCOUNTING_INVALID")
    scopes: list[IdmBlockScopeV1] = []
    for module in data.plan.modules:
        for lesson in module.lessons:
            for block_id in lesson.block_ids:
                block = by_block[block_id]
                scopes.append(IdmBlockScopeV1(
                    scope_key=block_scope_key(data.source_snapshot_hash, block.block_id, block.fact_keys),
                    block_id=block.block_id, title=single_line(block.name, 500),
                    source_ref=data.fact_source_ref.get(block.fact_keys[0]),
                    fact_count=len(block.fact_keys),
                    content_chars=max(1, sum(len(data.fact_text[key]) for key in block.fact_keys)),
                    fact_keys=list(block.fact_keys),
                ))
    scoped_facts = [key for scope in scopes for key in scope.fact_keys]
    course_facts = [item.fact_key for item in data.dispositions
                    if item.disposition in {"course", "reference_job_aid"}]
    if len(scoped_facts) != len(set(scoped_facts)) or set(scoped_facts) != set(course_facts):
        raise IdmStageError("IDM_ACCOUNTING_INVALID")
    notes = _build_notes(data)
    payload: dict[str, Any] = {
        "pipeline_version": IDM_PIPELINE_VERSION, "idm_contract_version": IDM_CONTRACT_VERSION,
        "prompt_policy_version": IDM_PROMPT_POLICY_VERSION, "source_snapshot_hash": data.source_snapshot_hash,
        "project_context": data.context.model_dump(mode="json"),
        "target_audience": data.audience.model_dump(mode="json"),
        "learning_objectives": [item.model_dump(mode="json") for item in data.objectives],
        "must_dos": [item.model_dump(mode="json") for item in data.must_dos],
        "blocks": [item.model_dump(mode="json") for item in data.blocks],
        "lo_links": [item.model_dump(mode="json") for item in data.links],
        "blueprint": [item.model_dump(mode="json") for item in data.rows],
        "blocked_must_do_ids": list(data.blocked),
        "hold_items": [item.model_dump(mode="json") for item in data.holds],
        "modules": [item.model_dump(mode="json") for item in data.plan.modules],
        "block_scopes": [item.model_dump(mode="json") for item in scopes],
        "dispositions": [item.model_dump(mode="json") for item in data.dispositions],
        "notes": notes.model_dump(mode="json"),
        "stage_origins": dict(sorted(data.stage_origins.items())),
    }
    payload["design_hash"] = design_hash_of(payload)
    return IdmCourseDesignV1.model_validate(payload)


def _build_notes(data: AssemblyInput) -> IdmAuthorNotesV1:
    locale = data.context.locale
    known_ids = list(data.snapshot_fact_keys)
    lessons = [lesson for module in data.plan.modules for lesson in module.lessons]
    ai_proposed = data.audience.origin == "ai_proposed" or any(
        item.origin == "ai_proposed" for item in data.objectives)
    reference_count = sum(1 for row in data.rows if row.placement == "reference_job_aid")
    excluded_count = sum(1 for row in data.rows if row.placement == "excluded")
    priority = [block for block in data.blocks
                if any(issue.type in {"conflict", "outdated"} for issue in block.issues)]
    others = [block for block in data.blocks if block not in priority]
    hold_ids = {item.block_id for item in data.holds}
    questions = [question for block in [*priority, *others] if block.block_id not in hold_ids
                 for question in block.sme_questions]
    course = build_course_notes(
        locale=locale, ai_proposed=ai_proposed, total_minutes=sum(lesson.est_minutes for lesson in lessons),
        lesson_count=len(lessons), module_count=len(data.plan.modules), reference_block_count=reference_count,
        excluded_block_count=excluded_count,
        holds=[(item.name, item.reason, item.sme_question) for item in data.holds], sme_questions=questions,
        fallback_stages=[stage for stage, origin in data.stage_origins.items() if origin != "provider"],
        known_ids=known_ids,
    )
    statements = {item.lo_id: item.statement for item in data.objectives}
    blooms = {item.must_do_id: item.bloom for item in data.must_dos}
    modules = {module.module_key: build_module_notes(
        locale=locale, performance_goal=module.performance_goal,
        objectives=[statements[lo_id] for lo_id in module.lo_ids if lo_id in statements],
        lesson_count=len(module.lessons), total_minutes=sum(lesson.est_minutes for lesson in module.lessons),
        known_ids=known_ids,
    ) for module in data.plan.modules}
    lesson_notes = {lesson.lesson_key: build_lesson_notes(
        locale=locale, bloom=blooms.get(lesson.primary_must_do_id or ""), est_screens=lesson.est_screens,
        est_minutes=lesson.est_minutes, ordering_rationale=lesson.ordering_rationale,
        job_aid=lesson.kind == "job_aid", known_ids=known_ids,
    ) for lesson in lessons}
    return IdmAuthorNotesV1(course=course, modules=modules, lessons=lesson_notes)


def project_course_skeleton(design: IdmCourseDesignV1, plan: IdmW4CourseResponseV1) -> CourseSkeletonV2:
    """Project the design onto the unchanged ``CourseSkeletonV2`` contract (spec §8.1)."""

    vi = design.project_context.locale == "vi"
    ai_proposed = design.target_audience.origin == "ai_proposed"
    suffix = (_AI_SUFFIX_VI if vi else _AI_SUFFIX_EN) if ai_proposed else ""
    audience = single_line(design.target_audience.description, 2000 - len(suffix)) + suffix
    statements = {item.lo_id: item.statement for item in design.learning_objectives}
    scope_of_block = {scope.block_id: scope.scope_key for scope in design.block_scopes}
    chapters = []
    for position, module in enumerate(design.modules):
        outcomes = [statements[lo_id] for lo_id in module.lo_ids if lo_id in statements][:_MAX_CHAPTER_OUTCOMES]
        chapters.append(CourseSkeletonChapterV2(
            chapter_key=f"chapter-{position + 1}", order=position, title=module.title,
            objective=module.performance_goal,
            learning_outcomes=outcomes or [statements[design.learning_objectives[0].lo_id]],
            source_scope_ids=[scope_of_block[block_id] for lesson in module.lessons for block_id in lesson.block_ids],
        ))
    assumptions = [f"IDM pipeline {IDM_PIPELINE_VERSION}"]
    if any(item.origin == "ai_proposed" for item in design.learning_objectives):
        assumptions.append("LO đề xuất bởi AI — cần xác nhận" if vi else "Objectives proposed by AI — please confirm")
    skeleton = CourseSkeletonV2(
        contract_version=2, source_snapshot_hash=design.source_snapshot_hash, locale=design.project_context.locale,
        title=plan.course_title, summary=plan.course_summary, target_audience=audience,
        prerequisites=list(plan.prerequisites),
        learning_outcomes=[item.statement for item in design.learning_objectives],
        assessment_strategy=plan.assessment_strategy, assumptions=assumptions, chapters=chapters,
    )
    bind_idm_course_skeleton(skeleton, design)
    return skeleton


def bind_idm_course_skeleton(skeleton: CourseSkeletonV2, design: IdmCourseDesignV1) -> None:
    """Every block scope belongs to exactly one chapter, one chapter per module."""

    assigned = [scope for chapter in skeleton.chapters for scope in chapter.source_scope_ids]
    expected = {scope.scope_key for scope in design.block_scopes}
    if (len(assigned) != len(set(assigned)) or set(assigned) != expected
            or len(skeleton.chapters) != len(design.modules)):
        raise IdmStageError("IDM_ACCOUNTING_INVALID")
