"""Lesson proposal JSON parsing, normalization and shape validation (legacy lesson writer contract)."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException

from app.ordered_learning_content import flatten_ordered_content
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.source_structure import strip_source_range_suffix
from app.workflows.contracts import WorkflowFailure


def parse_lesson_author_json(text: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise HTTPException(status_code=502, detail=f"AI không trả JSON {label} hợp lệ.")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail=f"AI không trả JSON {label} hợp lệ.") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail=f"AI không trả JSON {label} dạng object hợp lệ.")
    return parsed


def parse_course_architecture_repair_payload(text: str) -> dict[str, Any]:
    """Preserve the existing JSON acceptance rules with typed repair failures."""

    candidate = text.strip()
    diagnostics = {
        "response_chars": len(text),
        "response_bytes": len(text.encode("utf-8")),
        "json_candidate_found": False,
    }
    if not candidate:
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_INVALID",
            "Provider returned an empty scoped architecture repair.",
            internal_code="ARCH_REPAIR_EMPTY_RESPONSE",
            failure_stage="architecture_repair_json_parser",
            diagnostics=diagnostics,
        )
    try:
        parsed = json.loads(candidate)
        diagnostics["json_candidate_found"] = True
    except json.JSONDecodeError as first_error:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Provider returned invalid JSON for scoped architecture repair.",
                internal_code="ARCH_REPAIR_JSON_INVALID",
                failure_stage="architecture_repair_json_parser",
                diagnostics={
                    **diagnostics,
                    "json_error_position": first_error.pos,
                    "json_error_kind": "decode_error",
                },
            ) from first_error
        try:
            parsed = json.loads(match.group(0))
            diagnostics["json_candidate_found"] = True
        except json.JSONDecodeError as error:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Provider returned invalid JSON for scoped architecture repair.",
                internal_code="ARCH_REPAIR_JSON_INVALID",
                failure_stage="architecture_repair_json_parser",
                diagnostics={
                    **diagnostics,
                    "json_error_position": error.pos,
                    "json_error_kind": "embedded_decode_error",
                },
            ) from error
    if not isinstance(parsed, dict):
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_INVALID",
            "Provider returned a non-object scoped architecture repair.",
            internal_code="ARCH_REPAIR_RESPONSE_NOT_OBJECT",
            failure_stage="architecture_repair_json_parser",
            diagnostics=diagnostics,
        )
    return parsed


def _normalize_structural_title(value: Any, fallback: str) -> str:
    raw = value.strip() if isinstance(value, str) else ""
    return (strip_source_range_suffix(raw) or fallback).strip()[:180]


def normalize_lesson_author_proposal_tree(proposal: dict[str, Any]) -> dict[str, Any]:
    """Repair common flattened lesson shapes before strict proposal validation."""
    normalized = dict(proposal)
    chapters = normalized.get("chapters")
    if not isinstance(chapters, list):
        return normalized

    next_chapters: list[Any] = []
    for chapter_index, chapter_value in enumerate(chapters, start=1):
        if not isinstance(chapter_value, dict):
            next_chapters.append(chapter_value)
            continue
        chapter = dict(chapter_value)
        chapter["title"] = _normalize_structural_title(
            chapter.get("title"),
            f"Chương {chapter_index}",
        )
        lessons = chapter.get("lessons")
        if not isinstance(lessons, list):
            next_chapters.append(chapter)
            continue

        next_lessons: list[Any] = []
        for lesson_index, lesson_value in enumerate(lessons, start=1):
            if not isinstance(lesson_value, dict):
                next_lessons.append(lesson_value)
                continue
            lesson = dict(lesson_value)
            lesson["title"] = _normalize_structural_title(
                lesson.get("title"),
                f"Mục {lesson_index}",
            )
            nested_content = lesson.get("content") if isinstance(lesson.get("content"), dict) else {}
            units = lesson.get("units")
            if isinstance(units, dict):
                units = [units]
            if not isinstance(units, list) or not units:
                flattened_components = (
                    lesson.get("components")
                    or lesson.get("blocks")
                    or nested_content.get("components")
                    or nested_content.get("blocks")
                )
                if isinstance(flattened_components, list) and flattened_components:
                    units = [{
                        "title": lesson["title"],
                        "components": flattened_components,
                        "source_refs": lesson.get("source_refs", []),
                    }]
                else:
                    flattened_html = (
                        lesson.get("html")
                        or nested_content.get("html")
                        or (lesson.get("content") if isinstance(lesson.get("content"), str) else "")
                    )
                    if isinstance(flattened_html, str) and flattened_html.strip():
                        units = [{
                            "title": lesson["title"],
                            "components": [{
                                "type": "html",
                                "title": lesson["title"],
                                "html": flattened_html,
                            }],
                            "source_refs": lesson.get("source_refs", []),
                        }]
            if isinstance(units, list):
                units = [
                    {
                        **unit,
                        "title": _normalize_structural_title(
                            unit.get("title"),
                            f"Bài học {unit_index + 1}",
                        ),
                    }
                    if isinstance(unit, dict) else unit
                    for unit_index, unit in enumerate(units, start=1)
                ]
            lesson["units"] = units
            next_lessons.append(lesson)
        chapter["lessons"] = next_lessons
        next_chapters.append(chapter)

    normalized["chapters"] = next_chapters
    return normalized


MIN_LESSON_AUTHOR_HTML_TEXT_CHARS = 180
MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS = 320


SEMANTIC_LEARNING_HTML_LIMITS = {
    "heading": 240,
    "paragraphs": (12, 2_000),
    "bullet_points": (20, 800),
    "ordered_steps": (20, 1_000),
    "warnings": (8, 1_000),
    "comparison_rows": (30, 500, 1_000),
}


def semantic_learning_visible_text(value: Any) -> tuple[str, str | None]:
    """Validate the shared semantic payload and extract renderer-visible text.

    Node owns HTML rendering. Python only verifies the same bounded semantic
    vocabulary before accepting provider output, so an oversize value cannot
    be silently clipped after this workflow has declared source coverage.
    """
    if not isinstance(value, dict) or not value:
        return "", "Semantic explanatory content must be a non-empty object."
    try:
        value = flatten_ordered_content(value)
    except ValueError as exc:
        return "", str(exc)
    visible: list[str] = []
    heading = value.get("heading")
    if heading is not None:
        if not isinstance(heading, str) or not heading.strip():
            return "", "Semantic heading must be non-empty text."
        if len(heading.strip()) > SEMANTIC_LEARNING_HTML_LIMITS["heading"]:
            return "", "Semantic heading exceeds the renderer character limit."
        visible.append(heading.strip())
    fields: list[tuple[str, Any]] = [
        ("paragraphs", value.get("paragraphs")),
        ("bullet_points", value.get("bullet_points", value.get("bullets"))),
        ("ordered_steps", value.get("ordered_steps", value.get("steps"))),
        ("warnings", value.get("warnings", value.get("warning"))),
    ]
    for field_name, raw_items in fields:
        if raw_items is None:
            continue
        if not isinstance(raw_items, list):
            return "", f"Semantic {field_name} must be an array."
        max_items, max_characters = SEMANTIC_LEARNING_HTML_LIMITS[field_name]
        if len(raw_items) > max_items:
            return "", f"Semantic {field_name} exceeds the {max_items}-item renderer limit."
        for raw_item in raw_items:
            if not isinstance(raw_item, str):
                return "", f"Semantic {field_name} contains a non-string item."
            if not raw_item.strip():
                return "", f"Semantic {field_name} contains an empty text value."
            item = raw_item.strip()
            if len(item) > max_characters:
                return "", f"Semantic {field_name} contains text exceeding the renderer character limit."
            visible.append(item)
    raw_rows = value.get("comparison_rows", value.get("table_rows"))
    if raw_rows is not None:
        if not isinstance(raw_rows, list):
            return "", "Semantic comparison_rows must be an array."
        max_rows, max_label_characters, max_value_characters = SEMANTIC_LEARNING_HTML_LIMITS["comparison_rows"]
        if len(raw_rows) > max_rows:
            return "", f"Semantic comparison_rows exceeds the {max_rows}-row renderer limit."
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                return "", "Semantic comparison_rows contains an invalid row."
            label = raw_row.get("label")
            row_value = raw_row.get("value")
            if not isinstance(label, str) or not label.strip() or not isinstance(row_value, str) or not row_value.strip():
                return "", "Semantic comparison_rows contains an incomplete row."
            if len(label.strip()) > max_label_characters or len(row_value.strip()) > max_value_characters:
                return "", "Semantic comparison_rows contains text exceeding the renderer limit."
            visible.extend([label.strip(), row_value.strip()])
    if not visible:
        return "", "Semantic explanatory content has no renderer-visible text."
    return " ".join(visible), None


def _merged_lesson_author_component(component: dict[str, Any]) -> dict[str, Any]:
    """Expose nested RAG component payloads to the same validation rules."""
    nested_content = component.get("content")
    merged = dict(nested_content) if isinstance(nested_content, dict) else {}
    merged.update(component)
    return merged


def _non_empty_component_items(value: Any, *, kind: str) -> list[Any]:
    if not isinstance(value, list):
        return []
    valid: list[Any] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            valid.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if kind == "sortable":
            text = item.get("text") or item.get("label") or item.get("title")
        elif kind == "faq":
            text = item.get("question")
            answer = item.get("answer") or item.get("a") or item.get("content")
            if isinstance(text, str) and text.strip() and isinstance(answer, str) and answer.strip():
                valid.append(item)
            continue
        else:
            text = item.get("answer") or item.get("term") or item.get("text")
        if isinstance(text, str) and text.strip():
            valid.append(item)
    return valid


def validate_lesson_author_proposal_shape(proposal: dict[str, Any]) -> None:
    """Reject incomplete proposal trees before they reach CMS block creation."""
    chapters = proposal.get("chapters")
    # Keep compatibility with the historical changes-based prompt. The
    # backend still performs the final lossless conversion and validation.
    if not isinstance(chapters, list) or not chapters:
        changes = proposal.get("changes")
        if isinstance(changes, list) and changes:
            return
        raise LessonAuthorProposalValidationError(
            "Proposal phải có ít nhất một chương hoặc danh sách thay đổi hợp lệ.",
        )
    if len(chapters) > 1:
        raise LessonAuthorProposalValidationError(
            "Proposal soạn chi tiết chỉ được chứa một chương trong mỗi lần xử lý.",
        )

    for chapter_index, chapter_value in enumerate(chapters, start=1):
        if not isinstance(chapter_value, dict):
            raise LessonAuthorProposalValidationError(f"Chương {chapter_index} không hợp lệ.")
        lessons = chapter_value.get("lessons")
        if not isinstance(lessons, list) or not lessons:
            raise LessonAuthorProposalValidationError(f"Chương {chapter_index} chưa có bài học.")
        for lesson_index, lesson_value in enumerate(lessons, start=1):
            if not isinstance(lesson_value, dict):
                raise LessonAuthorProposalValidationError(
                    f"Bài học {lesson_index} trong chương {chapter_index} không hợp lệ.",
                )
            units = lesson_value.get("units")
            if not isinstance(units, list) or not units:
                raise LessonAuthorProposalValidationError(
                    f"Bài học {lesson_index} trong chương {chapter_index} chưa có unit.",
                )
            for unit_index, unit_value in enumerate(units, start=1):
                if not isinstance(unit_value, dict):
                    raise LessonAuthorProposalValidationError(
                        f"Unit {unit_index} trong bài học {lesson_index} không hợp lệ.",
                    )
                components = unit_value.get("components")
                if not isinstance(components, list) or not components:
                    components = unit_value.get("blocks")
                if isinstance(components, list) and components:
                    if not all(isinstance(component, dict) for component in components):
                        raise LessonAuthorProposalValidationError(
                            f"Unit {unit_index} trong bài học {lesson_index} có component không hợp lệ.",
                        )
                    for component_index, component in enumerate(components):
                        normalized_component = _merged_lesson_author_component(component)
                        nested_content = component.get("content") if isinstance(component.get("content"), dict) else {}
                        component_type = str(
                            normalized_component.get("type")
                            or normalized_component.get("block_type")
                            or nested_content.get("type")
                            or nested_content.get("block_type")
                            or ""
                        ).strip().casefold()
                        if component_type in {"la_sortable", "sortable", "ordering"}:
                            sortable_items = _non_empty_component_items(
                                normalized_component.get("items")
                                or normalized_component.get("ordered_items")
                                or normalized_component.get("steps"),
                                kind="sortable",
                            )
                            if len(sortable_items) < 3:
                                raise LessonAuthorProposalValidationError(
                                    "Sortable component requires at least 3 ordered items.",
                                    code="SORTABLE_ITEM_COUNT_INVALID", path=f"components[{component_index}].items", repairable=True,
                                )
                        elif component_type in {"la_faq", "faq"}:
                            faq_items = _non_empty_component_items(
                                normalized_component.get("items"),
                                kind="faq",
                            )
                            if len(faq_items) < 2:
                                raise LessonAuthorProposalValidationError(
                                    "FAQ component requires at least 2 Q&A items.",
                                    code="FAQ_ITEM_COUNT_INVALID", path=f"components[{component_index}].items", repairable=True,
                                )
                        elif component_type in {
                            "problem",
                            "question",
                            "quiz",
                            "la_problem",
                            "multiple_choice",
                            "multiple-select",
                            "multiple_select",
                            "multi_choice",
                            "multi_select",
                            "mcq",
                            "single_choice",
                            "dropdown",
                            "select",
                            "numerical",
                            "numeric",
                            "short_text",
                            "short-answer",
                            "short_answer",
                        }:
                            raw_problem_type = str(
                                normalized_component.get("problem_type")
                                or normalized_component.get("subtype")
                                or normalized_component.get("response_type")
                                or component_type
                                or "multiple_choice"
                            ).strip().casefold()
                            problem_type = {
                                "mcq": "multiple_choice",
                                "single_choice": "multiple_choice",
                                "multi_choice": "multiple_select",
                                "multi_select": "multiple_select",
                                "checkbox": "multiple_select",
                                "checkboxes": "multiple_select",
                                "select": "dropdown",
                                "option": "dropdown",
                                "numeric": "numerical",
                                "short_answer": "short_text",
                                "string": "short_text",
                            }.get(raw_problem_type, raw_problem_type)
                            if problem_type in {"numerical", "short_text"}:
                                answer = normalized_component.get("answer")
                                if not str(answer or "").strip():
                                    raise LessonAuthorProposalValidationError(
                                        "Problem component requires an answer.",
                                        code="PROBLEM_ANSWER_REQUIRED", path=f"components[{component_index}].answer", repairable=True,
                                    )
                            else:
                                option_key = "options" if problem_type == "dropdown" else "choices"
                                options = normalized_component.get(option_key)
                                if not isinstance(options, list) or len(_non_empty_component_items(options, kind="choice")) < 2:
                                    raise LessonAuthorProposalValidationError(
                                        "Problem component requires at least 2 answer choices.",
                                        code="PROBLEM_CHOICES_INVALID", path=f"components[{component_index}].choices", repairable=True,
                                    )
                        if component_type in {"html", "text", "content"}:
                            semantic_content = normalized_component.get("semantic_content")
                            if semantic_content is not None:
                                visible_text, semantic_failure = semantic_learning_visible_text(semantic_content)
                                if semantic_failure:
                                    raise LessonAuthorProposalValidationError(semantic_failure, code="HTML_SEMANTIC_INVALID", path=f"components[{component_index}].semantic_content", repairable=True)
                            else:
                                html_value = (
                                    component.get("html")
                                    or component.get("data")
                                    or (component.get("content") if isinstance(component.get("content"), str) else "")
                                    or nested_content.get("html")
                                    or nested_content.get("data")
                                    or nested_content.get("content")
                                    or ""
                                )
                                visible_text = re.sub(r"<[^>]+>", " ", str(html_value))
                                visible_text = re.sub(r"\s+", " ", visible_text).strip()
                            # A deterministic source-locked baseline is a
                            # review surface, not a claim that the model
                            # produced a complete lesson. Keep readable source
                            # visible instead of turning a sparse-but-valid
                            # document into a blank terminal failure.
                            source_locked = (
                                component.get("source_locked_fallback") is True
                                or normalized_component.get("source_locked_fallback") is True
                            )
                            minimum_html_chars = 1 if source_locked else MIN_LESSON_AUTHOR_HTML_TEXT_CHARS
                            if len(visible_text) < minimum_html_chars:
                                raise LessonAuthorProposalValidationError(
                                    f"HTML component trong Unit {unit_index} phải có ít nhất {minimum_html_chars} ký tự nội dung hiển thị.",
                                    code="HTML_INSUFFICIENT_DEPTH", path=f"components[{component_index}].semantic_content", repairable=True,
                                )
                    continue
                if isinstance(unit_value.get("html"), str) and unit_value["html"].strip():
                    visible_text = re.sub(r"<[^>]+>", " ", unit_value["html"])
                    visible_text = re.sub(r"\s+", " ", visible_text).strip()
                    if len(visible_text) < MIN_LESSON_AUTHOR_HTML_TEXT_CHARS:
                        raise LessonAuthorProposalValidationError(
                            f"HTML trong Unit {unit_index} phải có ít nhất {MIN_LESSON_AUTHOR_HTML_TEXT_CHARS} ký tự nội dung hiển thị.",
                        )
                    continue
                if isinstance(unit_value.get("content"), str) and unit_value["content"].strip():
                    visible_text = re.sub(r"<[^>]+>", " ", unit_value["content"])
                    visible_text = re.sub(r"\s+", " ", visible_text).strip()
                    if len(visible_text) < MIN_LESSON_AUTHOR_HTML_TEXT_CHARS:
                        raise LessonAuthorProposalValidationError(
                            f"Nội dung trong Unit {unit_index} phải có ít nhất {MIN_LESSON_AUTHOR_HTML_TEXT_CHARS} ký tự hiển thị.",
                        )
                    continue
                raise LessonAuthorProposalValidationError(
                    f"Unit {unit_index} trong bài học {lesson_index} chưa có nội dung.",
                )
