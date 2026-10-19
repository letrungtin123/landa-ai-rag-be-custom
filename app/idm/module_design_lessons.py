"""W3/W4 lesson checks of the module design (spec §7.6, §8.2): ``validate_lesson`` lists what a provider lesson
breaks, ``normalize_lesson`` makes the safe deterministic fixes and turns the lesson into its stored form.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from app.contracts.orchestration_v2.blueprint import ArchitectureComponentAuthorReviewV2
from app.idm.artifacts import ROW_LABEL_ANY_RE, plain_row_refs
from app.idm.contracts import IdmLessonDesignV1, IdmLessonPlanV1
from app.idm.module_autofix import format_has_evidence
from app.idm.module_layout import ModuleScope, clean_feedback, clean_media_brief
from app.idm.policy import (
    AI_DRAFTED_MARKER_EN,
    AI_DRAFTED_MARKER_VI,
    DOING_PRACTICE_TYPES,
    GRADED_PRACTICE_TYPES,
    IDM_THEORY_RUN_MAX_COMPONENTS,
    WORKSHEET_COMPONENT_TYPE,
)
from app.idm.text import idm_fold, produces_output, sanitize_author_text, single_line
from app.idm.validation import IdmIssue

# Graded practice types plus the worksheet html of a "do" Must Do (spec §10.1, QC 234653 R5).
PRACTICE_COMPONENT_TYPES: Final = GRADED_PRACTICE_TYPES | {WORKSHEET_COMPONENT_TYPE}
# Lower bound of IdmSupportItemV1.brief (characters).
_MIN_BRIEF_CHARS: Final = 3


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


def _brief_text(brief: str) -> str:
    """A support-item brief on one line, without the "Row N:" labels the PDF reader puts before table rows (QC run
    ab8d67e1, R5: the writer copied "Row 1: 01 Năng Lực …" from the brief into the learner's table)."""

    cleaned = single_line(ROW_LABEL_ANY_RE.sub("", brief), 300)
    return cleaned if len(cleaned) >= _MIN_BRIEF_CHARS else single_line(brief, 300)


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
                            item.model_copy(update={"brief": _brief_text(item.brief)})
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
                        "hold_question": single_line(plain_row_refs(practice.hold_question), 400)
                        if practice.hold_question else None,
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
