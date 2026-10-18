"""Course design and module stage of the "keep the document as it is" mode (QLT-6 ``preserve_source``).

The course skeleton task removes page furniture, segments the document with ONE provider call
(``app.idm.preserve_segment``) and cuts lessons into units (``app.idm.preserve_outline``). The result is an ordinary
IDM course design, so Node's checks, the chapter fan-out, the unit inventory, the receipts and Apply are unchanged:

* one Content Block per unit (its lines, in order); blocks are kept as they are (``treatment`` "keep");
* chapters and lessons carry the document's heading lines verbatim; the course keeps its CMS name (Apply never
  renames it);
* the pedagogy of the design mode does not run: no purpose profile, objectives, audience proposal, SME questions or
  judge. The contract still needs an objective, a Must Do, an audience and a purpose; they are fixed placeholders that
  start with "Không áp dụng" / "Not applicable", have origin "client" (never "AI proposed") and reach only author-only
  fields (design, skeleton and notes), never a learner page. ``project_context.design_mode`` tells every later stage
  that the design is a kept document.

The module stage is deterministic (no provider call): each unit gets exactly one html teaching slot that owns all
of its lines; no practice, FAQ or diagram, and therefore no assessment obligation.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from typing import Any, Final

from app.contracts.orchestration_v2.blueprint import (
    ArchitectureComponentAuthorReviewV2,
    ChapterBlueprintShardV2,
    CourseSkeletonV2,
)
from app.contracts.orchestration_v2.provider import ChapterShardPlanV2, SourceSnapshotFactV2
from app.idm.architecture import block_scope_key, project_course_skeleton
from app.idm.contracts import (
    IdmAudienceV1,
    IdmBlockScopeV1,
    IdmBlueprintRowV1,
    IdmComponentDesignV1,
    IdmContentBlockV1,
    IdmCourseDesignV1,
    IdmCourseSkeletonRequestV1,
    IdmDispositionV1,
    IdmLearningObjectiveV1,
    IdmLessonDesignV1,
    IdmLessonPlanV1,
    IdmModuleContextV1,
    IdmModulePlanV1,
    IdmMustDoV1,
    IdmProjectContextV1,
    IdmPurposeV1,
    IdmShardDesignV1,
    IdmUnitDesignV1,
    IdmW4CourseResponseV1,
    design_hash_of,
)
from app.idm.module_layout import ModuleScope, build_module_scope, project_lesson
from app.idm.policy import (
    IDM_CONTRACT_VERSION,
    IDM_PIPELINE_VERSION,
    IDM_PRESERVE_WORDS_PER_MINUTE,
    IDM_PROMPT_POLICY_VERSION,
)
from app.idm.preserve_lines import Furniture, page_furniture, visible_text
from app.idm.preserve_outline import Outline, build_outline
from app.idm.preserve_segment import Segmentation, segment_document, segmentation_log
from app.idm.runtime import IdmRuntime, IdmStageError, log_stage
from app.idm.text import single_line

PRESERVE_MODE: Final = "preserve_source"
EMPTY_CODE: Final = "IDM_PRESERVE_SOURCE_EMPTY"
HTML_NOT_ALLOWED_CODE: Final = "IDM_PRESERVE_HTML_NOT_ALLOWED"
MODE_MISMATCH_CODE: Final = "IDM_PRESERVE_MODE_MISMATCH"
COVERAGE_INVALID_CODE: Final = "IDM_PRESERVE_COVERAGE_INVALID"
SEGMENT_STAGE_ORIGIN_KEY: Final = "preserve_segmentation"
_MS: Final = 1000
_SUMMARY_CHARS: Final = 300
_MIN_SCREENS, _MAX_SCREENS = 2, 30
_MIN_MINUTES, _MAX_MINUTES = 2, 60

# Author-only placeholders the contract requires (never shown to learners), and the fixed author texts.
_TEXTS: Final[dict[str, dict[str, str]]] = {
    "vi": {
        "objective": "Không áp dụng — chế độ giữ nguyên nội dung tài liệu không đặt mục tiêu học tập.",
        "must_do": "Không áp dụng — giữ nguyên nội dung tài liệu.",
        "audience": "Không áp dụng — chế độ giữ nguyên nội dung tài liệu.",
        "purpose": "Giữ nguyên nội dung tài liệu: AI chỉ chia chương và bài theo đúng tài liệu.",
        "summary": ("Khoá học giữ nguyên nội dung tài liệu: nội dung được giữ nguyên từng chữ, AI chỉ chia chương "
                    "và bài."),
        "assessment": "Không áp dụng — chế độ giữ nguyên nội dung tài liệu không thêm câu hỏi hay bài kiểm tra.",
        "detail": "Giữ nguyên từng chữ, đúng thứ tự trong tài liệu.",
        "rationale": "Chế độ giữ nguyên nội dung tài liệu.",
        "ordering": "Theo đúng thứ tự trong tài liệu.",
        "unit_purpose": "Nội dung giữ nguyên theo tài liệu.",
        "slot_rationale": "Giữ nguyên nội dung tài liệu, từng chữ và đúng thứ tự.",
        "navigation": "Đọc nội dung.",
        "activity": "Đọc nội dung giữ nguyên theo tài liệu.",
        "page_number": "Số trang",
        "repeated_page_line": "Dòng đầu trang hoặc chân trang lặp lại",
    },
    "en": {
        "objective": "Not applicable — keeping the document as it is sets no learning objectives.",
        "must_do": "Not applicable — the document is kept as it is.",
        "audience": "Not applicable — the document is kept as it is.",
        "purpose": "Keep the document as it is: AI only splits it into chapters and lessons as the document does.",
        "summary": ("A course that keeps the document as it is: every word is kept, AI only splits chapters and "
                    "lessons."),
        "assessment": "Not applicable — keeping the document as it is adds no questions or tests.",
        "detail": "Kept word for word, in the document's order.",
        "rationale": "The document is kept as it is.",
        "ordering": "In the document's order.",
        "unit_purpose": "Content kept as it is in the document.",
        "slot_rationale": "The document's content, word for word and in order.",
        "navigation": "Read the content.",
        "activity": "Read the content kept as it is in the document.",
        "page_number": "Page number",
        "repeated_page_line": "Repeated page header or footer line",
    },
}


def is_preserve_context(context: IdmProjectContextV1) -> bool:
    return context.design_mode == PRESERVE_MODE


def _text(locale: str, key: str) -> str:
    return _TEXTS["vi" if locale == "vi" else "en"][key]


# --- author notes ---------------------------------------------------------------------------------------------
def course_note(locale: str, segmentation: Segmentation, outline: Outline, furniture: Furniture) -> str:
    """The course note in plain words: the mode, where the structure came from and what was removed."""

    if locale == "vi":
        lines = ["Chế độ: Giữ nguyên nội dung tài liệu — nội dung được giữ nguyên từng chữ, AI chỉ chia chương và bài."]
        lines.append({
            None: "• Cấu trúc chương, bài: AI chia theo các tiêu đề có trong tài liệu; tên chương, bài là đúng dòng "
                  "tiêu đề của tài liệu.",
            "headings": "• Cấu trúc chương, bài: AI chưa chia được nên hệ thống tự chia theo các tiêu đề tìm thấy "
                        "trong tài liệu. Hãy kiểm tra lại trước khi áp dụng.",
            "size": "• Cấu trúc chương, bài: tài liệu không có tiêu đề rõ ràng nên hệ thống chia theo trang và độ dài. "
                    "Phần nào không bắt đầu bằng một dòng tiêu đề được đặt tên tạm \"Phần 1\", \"Phần 2\"… Hãy kiểm "
                    "tra và đặt lại tên.",
        }[segmentation.fallback])
        if furniture.page_numbers or furniture.repeated_lines:
            lines.append(f"• Đã bỏ {furniture.page_numbers} dòng số trang và {furniture.repeated_lines} dòng đầu trang "
                         "hoặc chân trang lặp lại. Không bỏ dòng nào khác.")
        else:
            lines.append("• Không bỏ dòng nào của tài liệu.")
        if outline.split_lessons or outline.split_chapters:
            lines.append(f"• {outline.split_lessons} bài và {outline.split_chapters} chương quá dài được chia thành "
                         "nhiều phần liên tiếp (tên có thêm \"(phần 2)\"…).")
        lines.append("• Không thêm câu hỏi, câu hỏi thường gặp, ví dụ, bài luyện tập hay tóm tắt nào.")
        return "\n".join(lines)
    lines = ["Mode: Keep the document as it is — every word is kept, AI only splits chapters and lessons."]
    lines.append({
        None: "• Chapters and lessons: AI split the document at its own headings; every chapter and lesson title is "
              "the document's heading line.",
        "headings": "• Chapters and lessons: AI could not split the document, so the system split it at the "
                    "headings it found. Please check before applying.",
        "size": "• Chapters and lessons: the document has no clear headings, so the system split it by pages and "
                "length. A part that does not start with a heading line has a temporary name \"Part 1\", \"Part 2\"… "
                "Please check and rename.",
    }[segmentation.fallback])
    if furniture.page_numbers or furniture.repeated_lines:
        lines.append(f"• Removed {furniture.page_numbers} page number lines and {furniture.repeated_lines} repeated "
                     "page header or footer lines. Nothing else was removed.")
    else:
        lines.append("• No line of the document was removed.")
    if outline.split_lessons or outline.split_chapters:
        lines.append(f"• {outline.split_lessons} lessons and {outline.split_chapters} chapters that were too long "
                     "continue in consecutive parts (titles end with \"(part 2)\"…).")
    lines.append("• No questions, FAQ, examples, practice or summaries were added.")
    return "\n".join(lines)


def _lesson_note(locale: str, units: int, minutes: int) -> str:
    if locale == "vi":
        return f"Bài giữ nguyên theo tài liệu: {units} phần, khoảng {minutes} phút đọc."
    return f"Lesson kept as it is in the document: {units} parts, about {minutes} minutes of reading."


def _module_note(locale: str, lessons: int) -> str:
    if locale == "vi":
        return f"Chương giữ nguyên theo tài liệu: {lessons} bài."
    return f"Chapter kept as it is in the document: {lessons} lessons."


# --- course design --------------------------------------------------------------------------------------------
def _lesson_minutes(facts: Sequence[SourceSnapshotFactV2]) -> int:
    words = sum(len(visible_text(fact.fact_text).split()) for fact in facts)
    return max(_MIN_MINUTES, min(_MAX_MINUTES, math.ceil(words / IDM_PRESERVE_WORDS_PER_MINUTE)))


def _blocks_and_modules(
    outline: Outline, content: Sequence[SourceSnapshotFactV2], segmentation: Segmentation, locale: str,
) -> tuple[list[IdmContentBlockV1], list[IdmModulePlanV1], dict[str, str]]:
    blocks: list[IdmContentBlockV1] = []
    modules: list[IdmModulePlanV1] = []
    lesson_notes: dict[str, str] = {}
    lesson_number = 0
    for chapter_index, chapter in enumerate(outline.chapters, start=1):
        lessons: list[IdmLessonPlanV1] = []
        for lesson in chapter.lessons:
            lesson_number += 1
            block_ids: list[str] = []
            lesson_facts: list[SourceSnapshotFactV2] = []
            for unit in lesson.units:
                facts = [content[line - 1] for line in unit.lines]
                lesson_facts.extend(facts)
                block_id = f"cb_{len(blocks) + 1:04d}"
                block_ids.append(block_id)
                preview = " ".join(visible_text(fact.fact_text) for fact in facts[:2]).strip()
                blocks.append(IdmContentBlockV1(
                    block_id=block_id, section_id=f"sec_{chapter_index:03d}", name=unit.title,
                    summary=single_line(preview or unit.title, _SUMMARY_CHARS), intent="know", support_role=None,
                    content_kind="reference", fact_keys=[fact.fact_key for fact in facts], issues=[], gaps=[],
                    sme_questions=[], origin=("provider" if segmentation.origin == "provider"
                                              else "deterministic_fallback")))
            key = f"lsn_{lesson_number:03d}"
            minutes = _lesson_minutes(lesson_facts)
            lessons.append(IdmLessonPlanV1(
                lesson_key=key, kind="learning", title=lesson.title, primary_must_do_id=None, secondary_must_do_ids=[],
                block_ids=block_ids, est_screens=max(_MIN_SCREENS, min(_MAX_SCREENS, len(lesson.units))),
                est_minutes=minutes, ordering_rationale=_text(locale, "ordering")))
            lesson_notes[key] = _lesson_note(locale, len(lesson.units), minutes)
        modules.append(IdmModulePlanV1(
            module_key=f"mod_{chapter_index:02d}", title=chapter.title, performance_goal=_text(locale, "objective"),
            lo_ids=["lo_1"], lessons=lessons))
    return blocks, modules, lesson_notes


def assemble_preserve_design(
    *, source_snapshot_hash: str, context: IdmProjectContextV1, facts: Sequence[SourceSnapshotFactV2],
    content: Sequence[SourceSnapshotFactV2], furniture: Furniture, segmentation: Segmentation, outline: Outline,
) -> tuple[IdmCourseDesignV1, IdmW4CourseResponseV1]:
    """The IDM course design of a kept document and the course plan its skeleton is projected from."""

    locale = context.locale
    blocks, modules, lesson_notes = _blocks_and_modules(outline, content, segmentation, locale)
    # Coverage receipt: every line that is not page furniture is in exactly one unit, in document order.
    if [key for block in blocks for key in block.fact_keys] != [fact.fact_key for fact in content]:
        raise IdmStageError(COVERAGE_INVALID_CODE)
    text_of = {fact.fact_key: fact.fact_text for fact in facts}
    ref_of = {fact.fact_key: fact.source_ref for fact in facts}
    block_of = {key: block.block_id for block in blocks for key in block.fact_keys}
    scopes = [IdmBlockScopeV1(
        scope_key=block_scope_key(source_snapshot_hash, block.block_id, block.fact_keys), block_id=block.block_id,
        title=block.name, source_ref=ref_of[block.fact_keys[0]],
        fact_count=len(block.fact_keys), content_chars=max(1, sum(len(text_of[key]) for key in block.fact_keys)),
        fact_keys=list(block.fact_keys)) for block in blocks]
    dispositions = [
        IdmDispositionV1(fact_key=fact.fact_key, disposition="course", block_id=block_of[fact.fact_key], reason=None)
        if fact.fact_key in block_of else
        IdmDispositionV1(fact_key=fact.fact_key, disposition="noise", block_id=None,
                         reason=_text(locale, furniture.reasons[fact.fact_key]))
        for fact in facts]
    rows = [IdmBlueprintRowV1(
        block_id=block.block_id, lo_id="lo_1", must_do_ids=[], classification="must_know", placement="course",
        treatment="keep", detail_level=_text(locale, "detail"), hold=False, hold_reason=None, sme_question=None,
        combine_into=None, separate_into=[], rationale=_text(locale, "rationale")) for block in blocks]
    payload: dict[str, Any] = {
        "pipeline_version": IDM_PIPELINE_VERSION, "idm_contract_version": IDM_CONTRACT_VERSION,
        "prompt_policy_version": IDM_PROMPT_POLICY_VERSION, "source_snapshot_hash": source_snapshot_hash,
        "project_context": context.model_dump(mode="json"),
        "purpose": IdmPurposeV1(profile="general", summary=_text(locale, "purpose"), confidence="high",
                                origin="author_intent", evidence=[], signals=[], template_gaps=[],
                                fidelity_mode="keep").model_dump(mode="json"),
        "target_audience": IdmAudienceV1(description=_text(locale, "audience"),
                                         origin="client").model_dump(mode="json"),
        "learning_objectives": [IdmLearningObjectiveV1(lo_id="lo_1", statement=_text(locale, "objective"),
                                                       bloom="remember", origin="client").model_dump(mode="json")],
        "must_dos": [IdmMustDoV1(must_do_id="md_1", lo_id="lo_1", statement=_text(locale, "must_do"), kind="do",
                                 bloom="remember").model_dump(mode="json")],
        "blocks": [block.model_dump(mode="json") for block in blocks],
        "lo_links": [],
        "blueprint": [row.model_dump(mode="json") for row in rows],
        "blocked_must_do_ids": [], "hold_items": [],
        "modules": [module.model_dump(mode="json") for module in modules],
        "block_scopes": [scope.model_dump(mode="json") for scope in scopes],
        "dispositions": [item.model_dump(mode="json") for item in dispositions],
        "notes": {"course": course_note(locale, segmentation, outline, furniture),
                  "modules": {module.module_key: _module_note(locale, len(module.lessons)) for module in modules},
                  "lessons": lesson_notes},
        "stage_origins": {SEGMENT_STAGE_ORIGIN_KEY: segmentation.origin},
    }
    payload["design_hash"] = design_hash_of(payload)
    design = IdmCourseDesignV1.model_validate(payload)
    topic = context.course_title_hint or context.source_documents[0].name
    plan = IdmW4CourseResponseV1(
        course_title=single_line(topic, 500).ljust(3, "."), course_summary=_text(locale, "summary"),
        assessment_strategy=_text(locale, "assessment"), prerequisites=[], modules=modules, pattern_sections=[])
    return design, plan


async def run_preserve_course_design(
    request: IdmCourseSkeletonRequestV1, *, source_snapshot_hash: str, runtime: IdmRuntime,
) -> dict[str, Any]:
    """The ``/course-skeleton`` answer of a kept document: one segmentation call, everything else deterministic."""

    started = time.perf_counter()
    context = request.project_context
    if not is_preserve_context(context):
        raise IdmStageError(MODE_MISMATCH_CODE)
    facts = request.source_facts
    furniture = page_furniture(facts)
    content = [fact for fact in facts if fact.fact_key not in furniture.reasons]
    if not content:
        raise IdmStageError(EMPTY_CODE)
    texts = [fact.fact_text for fact in content]
    segmentation = await segment_document(runtime, texts, [fact.source_page for fact in content])
    outline = await asyncio.to_thread(build_outline, segmentation.chapters, texts, context.locale)
    design, plan = await asyncio.to_thread(
        lambda: assemble_preserve_design(source_snapshot_hash=source_snapshot_hash, context=context, facts=facts,
                                         content=content, furniture=furniture, segmentation=segmentation,
                                         outline=outline))
    skeleton: CourseSkeletonV2 = project_course_skeleton(design, plan)
    usage_complete = runtime.usage.complete and runtime.provider_failure_code is None
    log_stage("idm_stage_completed", {
        "correlation_id": runtime.correlation_id, "stage": "idm_preserve_course_design", "fact_count": len(facts),
        "page_numbers_removed": furniture.page_numbers, "repeated_lines_removed": furniture.repeated_lines,
        "segmentation": segmentation_log(segmentation), "call_count": runtime.usage.calls,
        "module_count": len(design.modules), "lesson_count": sum(len(module.lessons) for module in design.modules),
        "unit_count": len(design.blocks), "split_lessons": outline.split_lessons,
        "split_chapters": outline.split_chapters, "label_titles": outline.label_titles,
        "duration_ms": int((time.perf_counter() - started) * _MS), "usage": runtime.usage.as_usage(),
    })
    provider = segmentation.origin == "provider"
    return {
        "contract_version": 2, "skeleton": skeleton.model_dump(mode="json"), "idm": design.model_dump(mode="json"),
        "usage": runtime.usage.as_usage(), "usage_complete": usage_complete,
        "usage_source": "provider" if usage_complete else "reserved_upper_bound",
        "content_origin": "provider_validated" if provider else "structured_fallback",
        "quality_state": "validated" if provider else "review_required",
    }


# --- module stage (deterministic) -------------------------------------------------------------------------------
def _preserve_lesson(plan: IdmLessonPlanV1, scope: ModuleScope) -> IdmLessonDesignV1:
    locale = scope.locale
    units = []
    for position, block_id in enumerate(plan.block_ids, start=1):
        title = scope.blocks[block_id].name
        units.append(IdmUnitDesignV1(
            unit_index=position, segment="context_explain", title=title, purpose=_text(locale, "unit_purpose"),
            block_ids=[block_id], media_brief=None,
            components=[IdmComponentDesignV1(
                component_index=1, type="html", role="explain", title=title,
                rationale=_text(locale, "slot_rationale"), block_ids=[block_id], practice_id=None, support_items=[],
                author_review=ArchitectureComponentAuthorReviewV2(
                    purpose=_text(locale, "unit_purpose"), example_scenario=None, visual_asset=None,
                    user_behavior_navigation=_text(locale, "navigation")))]))
    return IdmLessonDesignV1(
        lesson_key=plan.lesson_key, title=plan.title, objective=_text(locale, "objective"),
        learning_objectives=[_text(locale, "objective")], practice_tasks=[], assessment=_text(locale, "assessment"),
        units=units, notes=_lesson_note(locale, len(units), plan.est_minutes))


def run_preserve_module_design(
    *, context: IdmModuleContextV1, skeleton: CourseSkeletonV2, plan: ChapterShardPlanV2,
    facts: Sequence[SourceSnapshotFactV2],
) -> dict[str, Any]:
    """The ``/chapter-shard`` answer of a kept document, without a provider call."""

    if not is_preserve_context(context.project_context):
        raise IdmStageError(MODE_MISMATCH_CODE)
    if "html" not in context.allowed_component_types:
        raise IdmStageError(HTML_NOT_ALLOWED_CODE)
    scope = build_module_scope(context, plan, facts)
    lessons = [_preserve_lesson(lesson_plan, scope) for lesson_plan in scope.lesson_plans]
    design_payload: dict[str, Any] = {
        "pipeline_version": IDM_PIPELINE_VERSION, "chapter_key": plan.chapter_key, "shard_index": plan.shard_index,
        "lessons": [lesson.model_dump(mode="json") for lesson in lessons],
        "lesson_index_offset": context.lesson_index_offset, "stage_origin": "deterministic_fallback",
    }
    design_payload["design_hash"] = design_hash_of(design_payload)
    shard_design = IdmShardDesignV1.model_validate(design_payload)
    chapter = next(item for item in skeleton.chapters if item.chapter_key == plan.chapter_key)
    activity = [_text(scope.locale, "activity")]
    shard = ChapterBlueprintShardV2.model_validate({
        "contract_version": 2, "source_snapshot_hash": skeleton.source_snapshot_hash,
        "chapter_key": chapter.chapter_key, "order": chapter.order, "shard_index": plan.shard_index,
        "shard_count": plan.shard_count, "source_scope_ids": list(plan.source_scope_ids), "title": chapter.title,
        "objective": chapter.objective,
        "lessons": [{**project_lesson(lesson, lesson_plan, scope), "learning_activities": activity}
                    for lesson, lesson_plan in zip(lessons, scope.lesson_plans, strict=True)],
        "assessment_obligations": [],
    })
    log_stage("idm_stage_completed", {"stage": "idm_preserve_module", "chapter_key": plan.chapter_key,
                                      "shard_index": plan.shard_index, "lesson_count": len(lessons),
                                      "unit_count": sum(len(lesson.units) for lesson in lessons), "call_count": 0})
    return {
        "contract_version": 2, "shard": {**shard.model_dump(mode="json"), "idm_design": shard_design.model_dump(
            mode="json")},
        "usage": {"inputTokens": 0, "outputTokens": 0, "embeddingTokens": 0, "totalTokens": 0},
        # Deterministic layout (no provider call): Node's planning envelope pairs it with "review_required"; the author
        # reviews the kept course before Apply anyway. Zero provider usage settles the dispatch fence Node sets.
        "usage_complete": True, "usage_source": "provider",
        "content_origin": "structured_fallback", "quality_state": "review_required",
    }
