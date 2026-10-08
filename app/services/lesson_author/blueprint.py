"""Course-architect blueprint generation with source-structure enforcement and a source-locked fallback."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal

from fastapi import HTTPException
from google.genai import types

from app.core.config import settings
from app.lesson_author_blueprint import (
    LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA,
    LessonAuthorBlueprintValidationError,
    ensure_lesson_author_blueprint_faqs,
    parse_and_validate_lesson_author_blueprint,
    parse_lesson_author_blueprint_candidate,
    validate_lesson_author_blueprint,
)
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorBlueprintRequest
from app.services import provider
from app.services.lesson_author.proposal_validation import _normalize_structural_title
from app.services.lesson_author.source_context import V5ImmutableSourceContext
from app.services.provider import combine_usage, emit_safe_provider_telemetry
from app.source_chapter_policy import bind_source_chapters
from app.workflows.contracts import WorkflowIssue, WorkflowValidationResult, safe_workflow_issue_summary


class LessonAuthorBlueprintGenerationError(RuntimeError):
    def __init__(self, code: str, usage: AiUsage, reason: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.usage = usage
        self.reason = reason


class CourseArchitectSemanticScopeError(RuntimeError):
    """A bounded Architect regeneration exhausted semantic-scope acceptance.

    The candidate was valid JSON/schema, but it never became an accepted
    Course Architect result because canonical primary ownership was incomplete
    or ambiguous. Only safe, server-derived metadata is retained.
    """

    def __init__(
        self,
        issues: list[WorkflowIssue],
        usage: AiUsage,
        attempt_count: int,
    ) -> None:
        super().__init__("ARCHITECTURE_SCOPE_INCOMPLETE")
        self.code = "ARCHITECTURE_SCOPE_INCOMPLETE"
        self.issues = issues
        self.usage = usage
        self.attempt_count = attempt_count


def _safe_course_architect_semantic_scope_findings(
    issues: list[WorkflowIssue],
) -> list[dict[str, Any]]:
    """Return IDs, structural paths, enums, and cardinalities only."""

    return [
        safe_workflow_issue_summary(issue, repairable=False)
        for issue in issues[:32]
    ]


def format_course_architect_semantic_scope_feedback(
    issues: list[WorkflowIssue],
) -> str:
    """Create deterministic provider feedback without source or Blueprint text."""

    return json.dumps(
        {
            "semantic_scope_findings": _safe_course_architect_semantic_scope_findings(issues),
            "required_action": "Regenerate the complete Course Architect Blueprint from the supplied SOURCE_MAP. Do not emit a patch or canonical source fact fields.",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_lesson_author_source_refs(
    blueprint: dict[str, Any],
    allowed_source_refs: set[str] | None,
) -> None:
    if allowed_source_refs is None:
        return
    invalid_refs: set[str] = set()
    for chapter in blueprint.get("chapters", []):
        for ref in chapter.get("source_refs", []) or []:
            if ref not in allowed_source_refs:
                invalid_refs.add(ref)
        for lesson in chapter.get("lessons", []) or []:
            for ref in lesson.get("source_refs", []) or []:
                if ref not in allowed_source_refs:
                    invalid_refs.add(ref)
            for unit in lesson.get("units", []) or []:
                if not isinstance(unit, dict):
                    continue
                for ref in unit.get("source_refs", []) or []:
                    if ref not in allowed_source_refs:
                        invalid_refs.add(ref)
                for component_plan in unit.get("component_plan", []) or []:
                    if not isinstance(component_plan, dict):
                        continue
                    for ref in component_plan.get("source_refs", []) or []:
                        if ref not in allowed_source_refs:
                            invalid_refs.add(ref)
    if invalid_refs:
        refs = ", ".join(sorted(invalid_refs)[:5])
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SOURCE_REF",
            f"Blueprint contains source references outside the supplied source outline: {refs}",
        )


def drop_invalid_lesson_author_source_refs(
    blueprint: dict[str, Any],
    allowed_source_refs: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Remove untrusted lesson-level refs without inventing replacement evidence."""
    dropped: list[str] = []

    def keep_allowed_refs(value: Any) -> list[str] | None:
        if not isinstance(value, list):
            return None
        kept: list[str] = []
        for ref in value:
            if isinstance(ref, str) and ref in allowed_source_refs:
                kept.append(ref)
            else:
                dropped.append(str(ref)[:32])
        return kept

    next_chapters: list[dict[str, Any]] = []
    for chapter_value in blueprint.get("chapters", []):
        if not isinstance(chapter_value, dict):
            next_chapters.append(chapter_value)
            continue
        chapter = dict(chapter_value)
        chapter_refs = keep_allowed_refs(chapter.get("source_refs"))
        if chapter_refs is not None:
            chapter["source_refs"] = chapter_refs
        next_lessons: list[dict[str, Any]] = []
        for lesson_value in chapter.get("lessons", []):
            if not isinstance(lesson_value, dict):
                next_lessons.append(lesson_value)
                continue
            lesson = dict(lesson_value)
            lesson_refs = keep_allowed_refs(lesson.get("source_refs"))
            if lesson_refs is not None:
                lesson["source_refs"] = lesson_refs
            next_units: list[dict[str, Any]] = []
            for unit_value in lesson.get("units", []):
                if not isinstance(unit_value, dict):
                    continue
                unit = dict(unit_value)
                unit_refs = keep_allowed_refs(unit.get("source_refs"))
                if unit_refs is not None:
                    unit["source_refs"] = unit_refs
                next_plan: list[dict[str, Any]] = []
                for plan_value in unit.get("component_plan", []):
                    if not isinstance(plan_value, dict):
                        continue
                    plan = dict(plan_value)
                    plan_refs = keep_allowed_refs(plan.get("source_refs"))
                    if plan_refs is not None:
                        plan["source_refs"] = plan_refs
                    next_plan.append(plan)
                if isinstance(unit.get("component_plan"), list):
                    unit["component_plan"] = next_plan
                next_units.append(unit)
            if isinstance(lesson.get("units"), list):
                lesson["units"] = next_units
            next_lessons.append(lesson)
        if isinstance(chapter.get("lessons"), list):
            chapter["lessons"] = next_lessons
        next_chapters.append(chapter)
    return {**blueprint, "chapters": next_chapters}, sorted(set(dropped))


def enforce_lesson_author_source_structure(
    blueprint: dict[str, Any],
    *,
    structure_source: str | None,
    authoritative_source_nodes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Make a TOC-derived outline deterministic before it is persisted.

    The model still designs objectives, lessons and activities. It must not
    rename, reorder, merge or invent top-level chapters when the parser found
    an authoritative table of contents. A mismatch is retried by the caller
    instead of being silently rewritten into a misleading course structure.
    """
    if structure_source != "toc" or not authoritative_source_nodes:
        return blueprint

    source_chapters = [
        node
        for node in authoritative_source_nodes
        if str(node.get("title") or "").strip()
        and str(node.get("source_ref") or "").strip()
    ]
    chapters = blueprint.get("chapters")
    if not isinstance(chapters, list) or len(chapters) != len(source_chapters):
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_SOURCE_STRUCTURE_MISMATCH",
            "Blueprint chapter count does not match the authoritative source table of contents.",
        )

    for chapter_index, (chapter, source_node) in enumerate(zip(chapters, source_chapters)):
        if not isinstance(chapter, dict):
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_SOURCE_STRUCTURE_MISMATCH",
                "Blueprint contains an invalid chapter entry.",
            )
        expected_title = _normalize_structural_title(
            source_node.get("title"),
            f"Chương {chapter_index + 1}",
        )
        actual_title = _normalize_structural_title(chapter.get("title"), "")
        expected_refs = [str(source_node["source_ref"]).strip()]
        actual_refs = [
            str(value).strip()
            for value in (chapter.get("source_refs") or [])
            if isinstance(value, str) and value.strip()
        ]
        if actual_title != expected_title or actual_refs != expected_refs:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_SOURCE_STRUCTURE_MISMATCH",
                "Blueprint chapter does not match the authoritative source table of contents.",
                path=f"chapters[{chapter_index}]",
                constraint="VERIFIED_TOC_CHAPTER_IDENTITY",
                expected_type="exact TOC chapter title and source reference",
                actual_type="mismatched chapter identity",
            )

    return blueprint


def lesson_author_blueprint_failure_message(locale: Literal["vi", "en"]) -> str:
    if locale == "en":
        return "The AI could not produce a valid course blueprint after an automatic retry. Please narrow the course goal or review the selected source material."
    return "AI chưa thể tạo Bản thiết kế khóa học hợp lệ sau khi đã thử lại tự động. Hãy thu hẹp mục tiêu khóa học hoặc kiểm tra tài liệu nguồn đã chọn."


def course_title_from_context(course_context: str | None, locale: Literal["vi", "en"]) -> str:
    """Extract the authoritative root title supplied by the backend course outline."""
    context = str(course_context or "")
    match = re.search(r"(?mi)^\s*(?:course|khóa học|khoa hoc)\s*:\s*(.+?)\s*$", context)
    if match:
        return match.group(1).strip()[:180]
    return "Course blueprint" if locale == "en" else "Bản thiết kế khóa học"


def build_source_locked_blueprint_fallback(
    request: RagLessonAuthorBlueprintRequest,
    source_structure_nodes: list[dict[str, Any]] | None,
    allowed_source_refs: set[str] | None,
    *,
    structure_source: str | None,
) -> dict[str, Any] | None:
    """Create a compact review plan from trusted source headings after model truncation.

    The fallback is intentionally architecture-only. It never tries to infer
    detailed lesson facts, and later drafting still receives the raw source
    facts as its authoritative input.
    """
    trusted_refs = allowed_source_refs or set()
    candidates: list[dict[str, Any]] = []
    for node in source_structure_nodes or []:
        title = _normalize_structural_title(node.get("title"), "")
        source_ref = str(node.get("source_ref") or "").strip()
        if not title or (trusted_refs and source_ref not in trusted_refs):
            continue
        try:
            level = int(node.get("level") or 1)
        except (TypeError, ValueError):
            level = 1
        try:
            order = int(node.get("order") or len(candidates))
        except (TypeError, ValueError):
            order = len(candidates)
        candidates.append({"title": title, "source_ref": source_ref, "level": level, "order": order})

    if not candidates:
        return None

    candidates.sort(key=lambda node: (node["order"], node["title"].casefold()))
    top_level = [node for node in candidates if node["level"] == 1]
    selected = top_level if top_level else candidates
    if structure_source == "toc" and not top_level:
        return None

    chapters: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    seen_titles: set[str] = set()
    is_english = request.locale == "en"
    for node in selected:
        source_ref = node["source_ref"]
        title = node["title"]
        title_key = title.casefold()
        if source_ref in seen_refs or title_key in seen_titles:
            continue
        seen_refs.add(source_ref)
        seen_titles.add(title_key)
        chapter_refs = [source_ref] if source_ref else []
        objective = (
            f"Understand and apply the source-grounded content of {title}."
            if is_english
            else f"Hiểu và vận dụng nội dung dựa trên tài liệu nguồn về {title}."
        )
        chapters.append({
            "title": title,
            "objective": objective,
            "source_refs": chapter_refs,
            "lessons": [{
                "title": title,
                "objective": objective,
                "learning_activities": [
                    "Review the source-grounded material for this section."
                    if is_english
                    else "Rà soát nội dung dựa trên tài liệu nguồn của phần này.",
                ],
                "assessment": (
                    "Check understanding against the source material."
                    if is_english
                    else "Kiểm tra mức độ hiểu theo tài liệu nguồn."
                ),
                "source_refs": chapter_refs,
                "units": [{
                    "title": title,
                    "source_refs": chapter_refs,
                    "component_plan": [{
                        "type": "html",
                        "title": title,
                        "rationale": (
                            "Explains the complete source-grounded content for this section."
                            if is_english
                            else "Trình bày đầy đủ nội dung dựa trên tài liệu nguồn của phần này."
                        ),
                    }],
                }],
            }],
        })
        if len(chapters) == 12:
            break

    if not chapters:
        return None

    return validate_lesson_author_blueprint({
        "title": course_title_from_context(request.course_context, request.locale),
        "summary": (
            "A compact course framework reconstructed from trusted source headings after the provider response was incomplete."
            if is_english
            else "Khung khóa học gọn được dựng lại từ các tiêu đề nguồn đáng tin cậy sau khi phản hồi từ nhà cung cấp chưa hoàn chỉnh."
        ),
        "target_audience": (
            "Learners confirmed by the course administrator."
            if is_english
            else "Người học được quản trị khóa học xác nhận."
        ),
        "prerequisites": [],
        "learning_outcomes": [
            "Identify the main source-grounded topics."
            if is_english
            else "Nhận diện các chủ đề chính có trong tài liệu nguồn.",
            "Explain the source-grounded principles and procedures."
            if is_english
            else "Giải thích nguyên tắc và quy trình dựa trên tài liệu nguồn.",
            "Apply the source-grounded knowledge in the relevant course context."
            if is_english
            else "Vận dụng kiến thức dựa trên tài liệu nguồn trong bối cảnh khóa học phù hợp.",
        ],
        "assessment_strategy": (
            "Use source-grounded knowledge checks and applied review."
            if is_english
            else "Dùng kiểm tra kiến thức và rà soát vận dụng dựa trên tài liệu nguồn."
        ),
        "assumptions": [
            "Confirm learner profile and delivery constraints before publication."
            if is_english
            else "Cần xác nhận hồ sơ người học và điều kiện triển khai trước khi xuất bản."
        ],
        "chapters": chapters,
    })


async def generate_validated_lesson_author_blueprint(
    request: RagLessonAuthorBlueprintRequest,
    prompt: str,
    allowed_source_refs: set[str] | None = None,
    *,
    structure_source: str | None = None,
    authoritative_source_nodes: list[dict[str, Any]] | None = None,
    source_chapter_policy: dict[str, Any] | None = None,
    source_structure_nodes: list[dict[str, Any]] | None = None,
    allow_server_fact_allocation: bool = False,
    semantic_scope_validator: Callable[[dict[str, Any]], WorkflowValidationResult] | None = None,
    v5_immutable_source_context: V5ImmutableSourceContext | None = None,
    defer_semantic_scope_validation: bool = False,
    emit_diagnostic: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], AiUsage]:
    total_usage = AiUsage()
    last_error: LessonAuthorBlueprintValidationError | None = None
    last_validation_feedback: str | None = None
    last_semantic_scope_issues: list[WorkflowIssue] = []
    # The terminal error must describe the final provider attempt. Earlier
    # failures are already logged safely and may inform retry feedback, but
    # must never overwrite a later schema/parser failure after the budget is
    # exhausted.
    terminal_failure_kind: Literal["semantic_scope", "schema"] | None = None

    for attempt in range(request.max_attempts):
        validation_feedback = (
            last_validation_feedback
            if last_validation_feedback is not None
            else "The prior candidate did not satisfy the server validation contract."
        )
        attempt_prompt = prompt if attempt == 0 else "\n\n".join(
            [
                prompt,
                "<SERVER_VALIDATION_FEEDBACK>\n"
                f"{validation_feedback}\n"
                "</SERVER_VALIDATION_FEEDBACK>",
                "SERVER ARCHITECT REGENERATION: The prior response was not accepted. Return a complete replacement Course Architect Blueprint from the authoritative SOURCE_MAP, not a scoped repair patch. Preserve every required semantic field, source-backed chapter, lesson, unit, semantic learning block, concept ID, source reference, and assessment signal. Never return source_fact_ids, covered_source_fact_ids, or source_fact_allocation: canonical facts are allocated only by the server after architecture design. The full provider output budget is available: do not compress, omit, or collapse source-backed learning architecture merely to save tokens. The validation feedback is server-generated and is the only retry instruction.",
            ]
        )
        try:
            text, usage = await provider.generate_content(
                request.api_key,
                request.model,
                attempt_prompt,
                max_output_tokens=request.max_output_tokens,
                json_mode=True,
                response_schema=LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA,
                thinking_config=types.ThinkingConfig(include_thoughts=False),
                request_timeout_ms=settings.blueprint_provider_request_timeout_ms,
                on_provider_telemetry=(
                    lambda telemetry, attempt_number=attempt + 1: emit_safe_provider_telemetry(
                        emit_diagnostic,
                        {
                            "stage": "course_architect_provider",
                            "event": "completed",
                            "architect_attempt": attempt_number,
                            "repair_pass_number": 0,
                            **telemetry,
                        },
                    )
                ),
            )
        except HTTPException as error:
            emit_safe_provider_telemetry(
                emit_diagnostic,
                {
                    "stage": "course_architect_provider",
                    "event": "failed",
                    "architect_attempt": attempt + 1,
                    "repair_pass_number": 0,
                    "provider_http_status": None if isinstance(error.detail, dict) and error.detail.get("code") == "AI_PROVIDER_TIMEOUT" else error.status_code,
                    "application_http_status": error.status_code,
                    "provider_finish_reason": None,
                    "provider_finish_reason_available": False,
                    "usage_source": "unavailable",
                },
            )
            raise
        total_usage = combine_usage(total_usage, usage)
        try:
            blueprint = parse_and_validate_lesson_author_blueprint(
                text,
                require_source_fact_ownership=not allow_server_fact_allocation,
                forbid_provider_fact_ownership=allow_server_fact_allocation,
            )
            try:
                validate_lesson_author_source_refs(blueprint, allowed_source_refs)
            except LessonAuthorBlueprintValidationError:
                # A model-generated lesson ref is never evidence by itself. In
                # TOC mode, drop unknown lesson refs and let the deterministic
                # chapter canonicalization below restore only the real source
                # reference. Other structure modes keep the strict failure.
                if source_chapter_policy is not None or structure_source != "toc" or allowed_source_refs is None:
                    raise
                blueprint, dropped_refs = drop_invalid_lesson_author_source_refs(
                    blueprint,
                    allowed_source_refs,
                )
                emit_safe_provider_telemetry(
                    emit_diagnostic,
                    {
                        "stage": "course_architect_output_validation",
                        "event": "source_refs_normalized",
                        "architect_attempt": attempt + 1,
                        "repair_pass_number": 0,
                        "dropped_source_ref_count": len(dropped_refs),
                    },
                )
                validate_lesson_author_source_refs(blueprint, allowed_source_refs)
            blueprint = bind_source_chapters(blueprint, source_chapter_policy) if source_chapter_policy is not None else enforce_lesson_author_source_structure(
                blueprint,
                structure_source=structure_source,
                authoritative_source_nodes=authoritative_source_nodes,
            )
            blueprint = ensure_lesson_author_blueprint_faqs(blueprint, request.locale)
            emit_safe_provider_telemetry(
                emit_diagnostic,
                {
                    "stage": "course_architect_output_validation",
                    "event": "structural_schema_passed",
                    "architect_attempt": attempt + 1,
                    "repair_pass_number": 0,
                    "response_chars": len(text),
                    "chapter_count": len(blueprint["chapters"]),
                },
            )
            if semantic_scope_validator is not None and not defer_semantic_scope_validation:
                semantic_validation = semantic_scope_validator(blueprint)
                semantic_errors = [
                    issue
                    for issue in semantic_validation.issues
                    if issue.get("severity") == "error"
                ]
                if semantic_errors:
                    last_semantic_scope_issues = semantic_errors
                    terminal_failure_kind = "semantic_scope"
                    last_validation_feedback = format_course_architect_semantic_scope_feedback(semantic_errors)
                    emit_safe_provider_telemetry(
                        emit_diagnostic,
                        {
                            "stage": "course_architect_semantic_scope_validation",
                            "event": "failed",
                            "architect_attempt": attempt + 1,
                            "repair_pass_number": 0,
                            "validation_finding_count": len(semantic_validation.issues),
                            "blocking_finding_count": len(semantic_errors),
                            "findings": _safe_course_architect_semantic_scope_findings(semantic_errors),
                        },
                    )
                    continue
                emit_safe_provider_telemetry(
                    emit_diagnostic,
                    {
                        "stage": "course_architect_semantic_scope_validation",
                        "event": "passed",
                        "architect_attempt": attempt + 1,
                        "repair_pass_number": 0,
                        "validation_finding_count": len(semantic_validation.issues),
                    },
                )
            return blueprint, total_usage
        except LessonAuthorBlueprintValidationError as error:
            last_error = error
            terminal_failure_kind = "schema"
            last_validation_feedback = re.sub(r"\s+", " ", str(error)).strip()[:240]
            emit_safe_provider_telemetry(
                emit_diagnostic,
                {
                    "stage": "course_architect_output_validation",
                    "event": "failed",
                    "architect_attempt": attempt + 1,
                    "repair_pass_number": 0,
                    "validation_code": error.code,
                    "response_chars": len(text),
                    **error.safe_diagnostic(),
                },
            )
            # A complete V5 JSON object with only a lesson-local schema
            # violation belongs to the graph's bounded patch repair, not a
            # second full Architect generation and never the legacy fallback.
            if v5_immutable_source_context is not None:
                try:
                    candidate = parse_lesson_author_blueprint_candidate(
                        text,
                        forbid_provider_fact_ownership=True,
                    )
                except LessonAuthorBlueprintValidationError:
                    candidate = None
                if (
                    isinstance(candidate, dict)
                    and candidate.get("architecture_contract_version") == 5
                    and error.code == "BLUEPRINT_INVALID_SCHEMA"
                    and re.fullmatch(r"chapters\[\d+\]\.lessons\[\d+\]\.units", error.path or "")
                ):
                    emit_safe_provider_telemetry(
                        emit_diagnostic,
                        {
                            "stage": "course_architect_output_validation",
                            "event": "local_schema_repair_deferred",
                            "architect_attempt": attempt + 1,
                            "repair_pass_number": 0,
                            "validation_code": error.code,
                            "schema_path": error.path,
                            "canonical_fact_count": v5_immutable_source_context.canonical_fact_count,
                            "evidence_scope_count": v5_immutable_source_context.evidence_scope_count,
                        },
                    )
                    return candidate, total_usage

    if terminal_failure_kind == "semantic_scope":
        raise CourseArchitectSemanticScopeError(
            last_semantic_scope_issues,
            total_usage,
            request.max_attempts,
        )

    if v5_immutable_source_context is not None:
        # `source_locked_fallback` is a legacy V3/V4 heading scaffold. It has
        # no V5 semantic blocks/scope ownership and must never enter a V5
        # allocation or repair flow with a zeroed canonical manifest.
        emit_safe_provider_telemetry(
            emit_diagnostic,
            {
                "stage": "course_architect_output_validation",
                "event": "v5_source_locked_fallback_rejected",
                "architect_attempt": request.max_attempts,
                "repair_pass_number": 0,
                "canonical_fact_count": v5_immutable_source_context.canonical_fact_count,
                "evidence_scope_count": v5_immutable_source_context.evidence_scope_count,
            },
        )
        raise LessonAuthorBlueprintGenerationError(
            "V5_SOURCE_CONTEXT_FALLBACK_UNSAFE",
            total_usage,
            last_error.code if last_error is not None else "BLUEPRINT_INVALID_SCHEMA",
        )

    fallback = build_source_locked_blueprint_fallback(
        request,
        source_structure_nodes,
        allowed_source_refs,
        structure_source=structure_source,
    )
    if fallback is not None:
        validate_lesson_author_source_refs(fallback, allowed_source_refs)
        fallback = enforce_lesson_author_source_structure(
            fallback,
            structure_source=structure_source,
            authoritative_source_nodes=authoritative_source_nodes,
        )
        fallback = ensure_lesson_author_blueprint_faqs(fallback, request.locale)
        emit_safe_provider_telemetry(
            emit_diagnostic,
            {
                "stage": "course_architect_output_validation",
                "event": "source_locked_fallback",
                "architect_attempt": request.max_attempts,
                "repair_pass_number": 0,
                "last_validation_code": last_error.code if last_error else "unknown",
                "chapter_count": len(fallback["chapters"]),
            },
        )
        return fallback, total_usage

    raise LessonAuthorBlueprintGenerationError(
        last_error.code if last_error else "BLUEPRINT_INVALID_SCHEMA",
        total_usage,
        re.sub(r"\s+", " ", str(last_error)).strip()[:240] if last_error else None,
    )
