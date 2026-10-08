"""Source-ref validation and blueprint source coverage of lesson proposals."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def update_blueprint_source_coverage(
    retrieval: dict[str, Any],
    blueprint: dict[str, Any],
    structure_context: dict[str, Any],
) -> dict[str, Any]:
    authoritative_nodes = structure_context.get("authoritative_source_nodes") or []
    if structure_context.get("structure_source") == "toc" and authoritative_nodes:
        # A Blueprint represents the course-level structure. Coverage is
        # therefore measured against every top-level TOC chapter, while
        # lesson-level evidence is checked later during content drafting.
        known_refs = {
            str(node.get("source_ref"))
            for node in authoritative_nodes
            if str(node.get("source_ref") or "").strip()
        }
    else:
        known_refs = set(structure_context.get("known_source_refs", set()))
    generated_refs: set[str] = set()
    for chapter in blueprint.get("chapters", []):
        generated_refs.update(str(ref) for ref in chapter.get("source_refs", []) or [])
        for lesson in chapter.get("lessons", []) or []:
            generated_refs.update(str(ref) for ref in lesson.get("source_refs", []) or [])
    covered_refs = generated_refs & known_refs
    ratio = len(covered_refs) / len(known_refs) if known_refs else None
    retrieval["known_source_ref_count"] = len(known_refs)
    retrieval["covered_source_ref_count"] = len(covered_refs)
    retrieval["source_coverage_ratio"] = round(ratio, 4) if ratio is not None else None
    return retrieval


def validate_lesson_author_proposal_source_refs(
    proposal: dict[str, Any],
    allowed_source_refs: set[str],
) -> None:
    invalid_refs: set[str] = set()
    for chapter_value in proposal.get("chapters", []):
        if not isinstance(chapter_value, dict):
            continue
        chapter = chapter_value
        for ref in chapter.get("source_refs", []) or []:
            if ref not in allowed_source_refs:
                invalid_refs.add(str(ref))
        for lesson_value in chapter.get("lessons", []) or []:
            if not isinstance(lesson_value, dict):
                continue
            lesson = lesson_value
            for ref in lesson.get("source_refs", []) or []:
                if ref not in allowed_source_refs:
                    invalid_refs.add(str(ref))
            for unit_value in lesson.get("units", []) or []:
                if not isinstance(unit_value, dict):
                    continue
                unit = unit_value
                for ref in unit.get("source_refs", []) or []:
                    if ref not in allowed_source_refs:
                        invalid_refs.add(str(ref))
    if invalid_refs:
        refs = ", ".join(sorted(invalid_refs)[:5])
        raise HTTPException(
            status_code=502,
            detail=f"AI trả mã nguồn không tồn tại trong cấu trúc tài liệu: {refs}",
        )


def drop_invalid_lesson_author_proposal_source_refs(
    proposal: dict[str, Any],
    allowed_source_refs: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Remove hallucinated source references without changing generated content."""
    dropped: list[str] = []

    def keep_allowed_refs(value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            dropped.append(str(value)[:32])
            return []
        kept: list[str] = []
        for ref in value:
            if isinstance(ref, str) and ref in allowed_source_refs:
                kept.append(ref)
            else:
                dropped.append(str(ref)[:32])
        return kept

    def sanitize_scope(value: Any) -> dict[str, Any] | Any:
        if not isinstance(value, dict):
            return value
        scope = dict(value)
        refs = keep_allowed_refs(scope.get("source_refs"))
        if refs is not None:
            scope["source_refs"] = refs
        return scope

    next_chapters: list[Any] = []
    for chapter_value in proposal.get("chapters", []):
        chapter = sanitize_scope(chapter_value)
        if not isinstance(chapter, dict):
            next_chapters.append(chapter)
            continue
        next_lessons: list[Any] = []
        for lesson_value in chapter.get("lessons", []) or []:
            lesson = sanitize_scope(lesson_value)
            if not isinstance(lesson, dict):
                next_lessons.append(lesson)
                continue
            if isinstance(lesson.get("units"), list):
                lesson["units"] = [sanitize_scope(unit) for unit in lesson["units"]]
            next_lessons.append(lesson)
        if isinstance(chapter.get("lessons"), list):
            chapter["lessons"] = next_lessons
        next_chapters.append(chapter)

    return {**proposal, "chapters": next_chapters}, sorted(set(dropped))
