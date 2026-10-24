"""Lesson proposal JSON parsing, normalization and shape validation (legacy lesson writer contract)."""

from __future__ import annotations

import json
import re
from typing import Any

from app.content.source_structure import strip_source_range_suffix
# Moved in refactor wave 3b; re-exported here for the legacy pipeline and its tests.
from app.services.staged_unit.proposal_shape import (
    MIN_LESSON_AUTHOR_HTML_TEXT_CHARS as MIN_LESSON_AUTHOR_HTML_TEXT_CHARS,
    MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS as MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS,
    SEMANTIC_LEARNING_HTML_LIMITS as SEMANTIC_LEARNING_HTML_LIMITS,
    semantic_learning_visible_text as semantic_learning_visible_text,
    validate_lesson_author_proposal_shape as validate_lesson_author_proposal_shape,
)
from app.workflows.contracts import WorkflowFailure


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
