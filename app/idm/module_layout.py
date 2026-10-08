"""Shard scope, deterministic lesson layout, held-practice obligations and V2 projection.

Split from ``module_design`` (spec §7.6.4 fallback, §7.6.5, §8.2) so each file stays
within the size budget (STD-5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from app.idm.contracts import (
    IdmBlueprintRowV1,
    IdmComponentDesignV1,
    IdmContentBlockV1,
    IdmFeedbackFocusV1,
    IdmLessonDesignV1,
    IdmLessonPlanV1,
    IdmModuleContextV1,
    IdmPracticeTaskV1,
    IdmShardDesignV1,
    IdmUnitDesignV1,
)
from app.idm.policy import MAX_OBLIGATION_COMPONENT_INDEX
from app.idm.runtime import IdmStageError
from app.idm.text import produces_output, single_line
from app.instructional_quality import build_source_grounded_single_choice
from app.lesson_author_orchestration_v2 import (
    ArchitectureComponentAuthorReviewV2,
    ArchitectureMediaBriefV2,
    ChapterBlueprintShardV2,
    CourseSkeletonV2,
    canonical_hash,
)
from app.lesson_author_orchestration_v2_provider import ChapterShardPlanV2, SourceSnapshotFactV2

_MAX_ACTIVITIES: Final = 3
_MAX_UNITS_PER_LESSON: Final = 12


@dataclass(frozen=True)
class ModuleScope:
    """Server-side lookups for one shard; built from the Node-provided context."""

    blocks: dict[str, IdmContentBlockV1]
    rows: dict[str, IdmBlueprintRowV1]
    scope_of_block: dict[str, str]
    fact_text: dict[str, str]
    lesson_plans: list[IdmLessonPlanV1]
    must_do_statement: dict[str, str]
    locale: str
    # Must Do id -> "do" | "decide": a "do" Must Do needs a practice that produces the output (R5).
    must_do_kind: dict[str, str] = field(default_factory=dict)

    def block_texts(self, block_ids: Sequence[str]) -> list[str]:
        return [
            self.fact_text[key]
            for block_id in block_ids
            for key in self.blocks[block_id].fact_keys
            if key in self.fact_text
        ]

    def lesson_fact_keys(self, plan: IdmLessonPlanV1) -> set[str]:
        return {key for block_id in plan.block_ids for key in self.blocks[block_id].fact_keys}


def build_module_scope(
    context: IdmModuleContextV1,
    plan: ChapterShardPlanV2,
    facts: Sequence[SourceSnapshotFactV2],
) -> ModuleScope:
    """Check the Node context against the shard plan and the remapped facts."""

    blocks = {block.block_id: block for block in context.blocks}
    rows = {row.block_id: row for row in context.blueprint}
    scope_of_block = {scope.block_id: scope.scope_key for scope in context.block_scopes}
    lesson_blocks = [block_id for lesson in context.module.lessons for block_id in lesson.block_ids]
    if (
        len(lesson_blocks) != len(set(lesson_blocks))
        or any(
            block_id not in blocks or block_id not in rows or block_id not in scope_of_block
            for block_id in lesson_blocks
        )
        or [scope_of_block[block_id] for block_id in lesson_blocks] != list(plan.source_scope_ids)
    ):
        raise IdmStageError("IDM_MODULE_CONTEXT_INVALID")
    expected_facts = {key for block_id in lesson_blocks for key in blocks[block_id].fact_keys}
    allowed_scopes = set(plan.source_scope_ids)
    if (
        any(fact.scope_key not in allowed_scopes for fact in facts)
        or {fact.fact_key for fact in facts} != expected_facts
    ):
        raise IdmStageError("IDM_MODULE_CONTEXT_INVALID")
    return ModuleScope(
        blocks=blocks,
        rows=rows,
        scope_of_block=scope_of_block,
        fact_text={fact.fact_key: fact.fact_text for fact in facts},
        lesson_plans=list(context.module.lessons),
        must_do_statement={item.must_do_id: item.statement for item in context.must_dos},
        locale=context.project_context.locale,
        must_do_kind={item.must_do_id: item.kind for item in context.must_dos},
    )


def _vi(locale: str, vi: str, en: str) -> str:
    return vi if locale == "vi" else en


def fallback_lesson(
    plan: IdmLessonPlanV1,
    scope: ModuleScope,
    title_hint: str | None = None,
    provider_practices: Sequence[IdmPracticeTaskV1] = (),
) -> IdmLessonDesignV1:
    """One unit per block with one HTML component; a grounded check when the source allows."""

    locale = scope.locale
    objective = scope.must_do_statement.get(plan.primary_must_do_id or "", plan.title)
    units: list[IdmUnitDesignV1] = []
    # One unit per block, grouped when a lesson has more blocks than units allowed.
    size = -(-len(plan.block_ids) // _MAX_UNITS_PER_LESSON)
    groups = [plan.block_ids[start : start + size] for start in range(0, len(plan.block_ids), size)]
    for position, group in enumerate(groups, start=1):
        block = scope.blocks[group[0]]
        units.append(
            IdmUnitDesignV1(
                unit_index=position,
                segment="job_aid" if plan.kind == "job_aid" else "context_explain",
                title=single_line(block.name, 180).ljust(3, "."),
                purpose=single_line(
                    _vi(
                        locale,
                        f"Người học nắm phần cần thiết để: {objective}",
                        f"The learner gets what is needed to: {objective}",
                    ),
                    500,
                ),
                block_ids=list(group),
                components=[
                    IdmComponentDesignV1(
                        component_index=1,
                        type="html",
                        role="job_aid" if plan.kind == "job_aid" else "explain",
                        title=single_line(block.name, 180).ljust(3, "."),
                        rationale=_vi(
                            locale,
                            "Bố cục dự phòng: trình bày ngắn gọn nội dung cần thiết.",
                            "Fallback layout: a short presentation of the needed content.",
                        ),
                        block_ids=list(group),
                        practice_id=None,
                        support_items=[],
                        author_review=ArchitectureComponentAuthorReviewV2(
                            purpose=single_line(
                                _vi(locale, f"Hỗ trợ Must Do: {objective}", f"Supports the Must Do: {objective}"), 1200
                            ),
                            example_scenario=None,
                            visual_asset=None,
                            user_behavior_navigation=_vi(locale, "Đọc rồi tiếp tục.", "Read, then continue."),
                        ),
                    )
                ],
                media_brief=None,
            )
        )
    practices: list[IdmPracticeTaskV1] = []
    if plan.kind == "learning" and provider_practices:
        # §7.6.4/§7.6.5: the provider practice survives the layout fallback as a hold for the SME.
        practices = _held(provider_practices, scope, plan)
    elif plan.kind == "learning":
        practices, units = _fallback_practice(plan, scope, units, objective)
    activities_hint = title_hint or plan.title
    notes = _vi(locale, f"Bố cục dự phòng tự động cho: {activities_hint}. Cần rà soát.",
                f"Automatic fallback layout for: {activities_hint}. Review needed.")
    primary = plan.primary_must_do_id or ""
    if plan.kind == "learning" and produces_output(scope.must_do_statement.get(primary, ""),
                                                   scope.must_do_kind.get(primary, "")):
        notes += " " + _vi(locale, "Must Do loại làm: cần bổ sung phiếu thực hành (worksheet) để người học tự làm.",
                           "A doing Must Do: add a worksheet practice so the learner produces the output.")
    return IdmLessonDesignV1(
        lesson_key=plan.lesson_key,
        title=single_line(plan.title, 180).ljust(3, "."),
        objective=single_line(objective, 500).ljust(5, "."),
        learning_objectives=[single_line(objective, 500)],
        practice_tasks=practices,
        assessment=single_line(
            _vi(
                locale,
                "Đánh giá theo tiêu chí của Must Do ở cuối mục.",
                "Assessed against the Must Do criterion at the end of the section.",
            ),
            500,
        ),
        units=units,
        notes=single_line(notes, 2000),
    )


def _fallback_practice(
    plan: IdmLessonPlanV1,
    scope: ModuleScope,
    units: list[IdmUnitDesignV1],
    objective: str,
) -> tuple[list[IdmPracticeTaskV1], list[IdmUnitDesignV1]]:
    locale = scope.locale
    primary = plan.primary_must_do_id
    target = next(
        (
            position
            for position, unit in enumerate(units)
            if any(primary in scope.rows[block_id].must_do_ids for block_id in unit.block_ids)
        ),
        len(units) - 1,
    )
    unit_blocks = list(units[target].block_ids)
    texts = scope.block_texts(unit_blocks)
    if build_source_grounded_single_choice(plan.title, texts, locale=locale) is None:
        return [], units
    practice = IdmPracticeTaskV1(
        practice_id="pt_1",
        sentence=single_line(
            _vi(
                locale,
                f"Cho một tình huống công việc, người học chọn cách làm đúng để {objective}",
                f"Given a work situation, the learner chooses the correct action to {objective}",
            ),
            280,
        ),
        context_input=_vi(locale, "Tình huống công việc ngắn", "A short work situation"),
        learner_action=_vi(locale, "Chọn phương án đúng", "Choose the correct option"),
        result=_vi(locale, "Quyết định đúng theo tiêu chí", "A decision that meets the criterion"),
        bloom="apply",
        criteria_fact_keys=[key for block_id in unit_blocks for key in scope.blocks[block_id].fact_keys][:24],
        scenario_origin="source",
        hold=False,
        hold_question=None,
        feedback_focus=IdmFeedbackFocusV1(
            criterion=_vi(locale, "Tiêu chí nêu trong nội dung vừa học", "The criterion stated in the content"),
            rationale=_vi(
                locale,
                "Phương án đúng thoả tiêu chí, các phương án khác thì không",
                "The correct option meets the criterion; the others do not",
            ),
            improvement=_vi(
                locale, "Đối chiếu lại tiêu chí trước khi chọn", "Check the criterion again before choosing"
            ),
        ),
    )
    unit = units[target]
    check = IdmComponentDesignV1(
        component_index=2,
        type="problem",
        role="practice",
        title=_vi(locale, "Luyện tập áp dụng", "Practice"),
        rationale=_vi(
            locale, "Kiểm tra khả năng áp dụng đúng tiêu chí.", "Checks correct application of the criterion."
        ),
        block_ids=unit_blocks[:12],
        practice_id="pt_1",
        support_items=[],
        author_review=ArchitectureComponentAuthorReviewV2(
            purpose=single_line(
                _vi(locale, f"Luyện tập Must Do: {objective}", f"Practice the Must Do: {objective}"), 1200
            ),
            example_scenario=None,
            visual_asset=None,
            user_behavior_navigation=_vi(
                locale, "Chọn một đáp án và xem phản hồi.", "Choose one answer and read the feedback."
            ),
        ),
    )
    units[target] = unit.model_copy(update={"segment": "practice_feedback", "components": [*unit.components, check]})
    return [practice], units


def _held(practices: Sequence[IdmPracticeTaskV1], scope: ModuleScope, plan: IdmLessonPlanV1) -> list[IdmPracticeTaskV1]:
    allowed = scope.lesson_fact_keys(plan)
    question = _vi(
        scope.locale,
        "Bố cục tự động của mục này không hợp lệ; SME xác nhận tiêu chí đúng/sai để dựng lại bài luyện tập.",
        "The automatic layout of this section failed; the SME should confirm the criteria to rebuild the practice.",
    )
    return [
        practice.model_copy(
            update={
                "hold": True,
                "sentence": single_line(practice.sentence, 280),
                "hold_question": single_line(practice.hold_question or question, 400),
                "feedback_focus": clean_feedback(practice.feedback_focus),
                "criteria_fact_keys": [key for key in practice.criteria_fact_keys if key in allowed],
            }
        )
        for practice in practices
    ]


def clean_feedback(focus: IdmFeedbackFocusV1) -> IdmFeedbackFocusV1:
    """Feedback focus reaches the unit writer and notes; keep it editor safe (R10)."""

    return focus.model_copy(update={key: single_line(getattr(focus, key), 400)
                                    for key in ("criterion", "rationale", "improvement")})


def clean_media_brief(brief: ArchitectureMediaBriefV2 | None) -> ArchitectureMediaBriefV2 | None:
    if brief is None:
        return None
    return brief.model_copy(update={
        "title": single_line(brief.title, 180), "rationale": single_line(brief.rationale, 500),
        "context_description": single_line(brief.context_description, 1000),
        "content_points": [single_line(point, 600) for point in brief.content_points],
    })


def held_practice_obligations(
    lessons: Sequence[IdmLessonDesignV1],
    plans: Sequence[IdmLessonPlanV1],
    scope: ModuleScope,
    *,
    skeleton: CourseSkeletonV2,
    plan: ChapterShardPlanV2,
) -> list[dict[str, Any]]:
    """A held practice that would have been a check in slots 1..3 becomes an obligation (§7.6.5)."""

    obligations = []
    for lesson_index, (lesson, lesson_plan) in enumerate(zip(lessons, plans, strict=True), start=1):
        taken: dict[int, int] = {}
        for practice in lesson.practice_tasks:
            if not practice.hold:
                continue
            unit_index = next(
                (
                    position
                    for position, unit in enumerate(lesson.units, start=1)
                    if any(
                        lesson_plan.primary_must_do_id in scope.rows[block_id].must_do_ids
                        for block_id in unit.block_ids
                    )
                ),
                len(lesson.units),
            )
            unit = lesson.units[unit_index - 1]
            # Each held practice of a unit takes its own planned slot after the real components.
            component_index = len(unit.components) + 1 + taken.get(unit_index, 0)
            if component_index > MAX_OBLIGATION_COMPONENT_INDEX:
                continue
            taken[unit_index] = taken.get(unit_index, 0) + 1
            obligations.append(
                {
                    "planned_slot_key": "ao2_"
                    + canonical_hash(
                        {
                            "source_snapshot_hash": skeleton.source_snapshot_hash,
                            "chapter_key": plan.chapter_key,
                            "shard_index": plan.shard_index,
                            "lesson_index": lesson_index,
                            "unit_index": unit_index,
                            "component_index": component_index,
                        }
                    )[:32],
                    "lesson_index": lesson_index,
                    "unit_index": unit_index,
                    "component_index": component_index,
                    "learning_objective_refs": [
                        f"lo_{position}" for position in range(1, len(lesson.learning_objectives) + 1)
                    ],
                    "required_assessment_kind": "single_choice",
                    "relevant_scope_ids": [scope.scope_of_block[block_id] for block_id in unit.block_ids],
                    "relevant_evidence_fact_ids": [
                        key for block_id in unit.block_ids for key in scope.blocks[block_id].fact_keys
                    ],
                    "unresolved_reason": "ASSESSMENT_SOURCE_CHECK_REQUIRED",
                    "status": "open",
                }
            )
    return obligations


# --- projection ---------------------------------------------------------------------------------
def _activities(lesson: IdmLessonDesignV1, plan: IdmLessonPlanV1, locale: str) -> list[str]:
    if plan.kind == "learning" and not lesson.practice_tasks:
        # A learning lesson without practice is a gap to fix, not an on-the-job lookup
        # ("Tra cứu Thực hiện đúng … khi thực hiện công việc").
        return [
            single_line(
                _vi(
                    locale,
                    f"Chưa có bài luyện tập — cần thiết kế hoạt động thực hành cho: {lesson.title}",
                    f"No practice yet — design a practice activity for: {lesson.title}",
                ),
                280,
            )
        ]
    if plan.kind == "job_aid" or not lesson.practice_tasks:
        return [
            single_line(
                _vi(
                    locale,
                    f"Tra cứu {lesson.title} khi thực hiện công việc",
                    f"Use {lesson.title} as an on-the-job reference",
                ),
                280,
            )
        ]
    return [single_line(practice.sentence, 280) for practice in lesson.practice_tasks][:_MAX_ACTIVITIES]


def project_lesson(lesson: IdmLessonDesignV1, lesson_plan: IdmLessonPlanV1, scope: ModuleScope) -> dict[str, Any]:
    """Project one IDM lesson onto ``LessonArchitectureV2`` JSON (spec §8.2)."""

    refs = [f"lo_{position}" for position in range(1, len(lesson.learning_objectives) + 1)]
    assessment = (
        lesson.assessment
        if lesson_plan.kind == "learning"
        else _vi(scope.locale, "Không đánh giá — tài liệu hỗ trợ khi làm việc", "Not assessed — on-the-job support")
    )
    return {
        "title": lesson.title,
        "objective": lesson.objective,
        "learning_objectives": list(lesson.learning_objectives),
        "learning_activities": _activities(lesson, lesson_plan, scope.locale),
        "assessment": assessment,
        "units": [
            {
                "title": unit.title,
                "purpose": unit.purpose,
                "learning_objective_refs": refs,
                "source_scope_ids": [scope.scope_of_block[block_id] for block_id in unit.block_ids],
                "component_plan": [
                    {
                        "type": component.type,
                        "title": component.title,
                        "rationale": component.rationale,
                        "author_review": component.author_review.model_dump(mode="json"),
                        "source_scope_ids": [scope.scope_of_block[block_id] for block_id in component.block_ids],
                    }
                    for component in unit.components
                ],
                "media_brief": unit.media_brief.model_dump(mode="json") if unit.media_brief else None,
            }
            for unit in lesson.units
        ],
    }


def project_shard(
    lessons: Sequence[IdmLessonDesignV1],
    plans: Sequence[IdmLessonPlanV1],
    scope: ModuleScope,
    *,
    skeleton: CourseSkeletonV2,
    plan: ChapterShardPlanV2,
    shard_design: IdmShardDesignV1,
) -> dict[str, Any]:
    """Return ``ChapterBlueprintShardV2`` JSON plus ``idm_design`` (§8.2)."""

    chapter = next(item for item in skeleton.chapters if item.chapter_key == plan.chapter_key)
    projected_lessons = [project_lesson(lesson, lesson_plan, scope) for lesson, lesson_plan in
                         zip(lessons, plans, strict=True)]
    shard = ChapterBlueprintShardV2.model_validate(
        {
            "contract_version": 2,
            "source_snapshot_hash": skeleton.source_snapshot_hash,
            "chapter_key": chapter.chapter_key,
            "order": chapter.order,
            "shard_index": plan.shard_index,
            "shard_count": plan.shard_count,
            "source_scope_ids": list(plan.source_scope_ids),
            "title": chapter.title,
            "objective": chapter.objective,
            "lessons": projected_lessons,
            "assessment_obligations": held_practice_obligations(lessons, plans, scope, skeleton=skeleton, plan=plan),
        }
    )
    return {**shard.model_dump(mode="json"), "idm_design": shard_design.model_dump(mode="json")}
