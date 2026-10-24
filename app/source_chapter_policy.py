"""Bounded server-owned chapter authority; never redistributes source facts."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

# Moved in refactor wave 3b; re-exported here for the legacy pipeline and its tests.
from app.content.source_chapter_policy import (
    resolve_source_chapter_policy as resolve_source_chapter_policy,
)
from app.lesson_author_blueprint import LessonAuthorBlueprintValidationError


def bind_source_chapters(blueprint: dict[str, Any], policy: dict[str, Any], source_map: dict[str, Any] | None = None) -> dict[str, Any]:
    """Exact source-ref binding, never positional relabeling or fuzzy titles."""
    if not policy.get("complete"):
        raise LessonAuthorBlueprintValidationError("SOURCE_STRUCTURE_REVIEW_REQUIRED", "Source chapter authority requires review.")
    result = deepcopy(blueprint)
    if policy["mode"].startswith("SOURCE_LOCKED"):
        expected = policy["chapters"]
        chapters = result.get("chapters", [])
        if len(chapters) != len(expected):
            raise LessonAuthorBlueprintValidationError("BLUEPRINT_SOURCE_STRUCTURE_MISMATCH", "Chapter bindings do not cover the source inventory.")
        for index, (chapter, binding) in enumerate(zip(chapters, expected)):
            # Both exact identity and authoritative order are required. Title
            # injection is legal only AFTER this evidence binding succeeds.
            if chapter.get("source_refs") != [binding["source_ref"]]:
                raise LessonAuthorBlueprintValidationError(
                    "BLUEPRINT_SOURCE_STRUCTURE_MISMATCH", "Chapter has an invalid or reordered source binding.",
                    path=f"chapters[{index}]", constraint="EXACT_SOURCE_CHAPTER_BINDING")
            chapter["title"] = binding["title"]
            if source_map is not None:
                sections = source_map.get("sections", [])
                roots = [section for section in sections if section.get("document_id") == binding["document_id"]
                         and section.get("source_ref") == binding["source_ref"]]
                if len(roots) != 1:
                    raise LessonAuthorBlueprintValidationError("BLUEPRINT_SOURCE_STRUCTURE_MISMATCH", "Source chapter binding is absent or ambiguous in the Source Map.")
                owned_sections = {roots[0]["id"]}
                for _ in range(len(sections)):
                    expanded = owned_sections | {s["id"] for s in sections if s.get("parent_id") in owned_sections
                                                 and s.get("document_id") == binding["document_id"]}
                    if expanded == owned_sections:
                        break
                    owned_sections = expanded
                scopes = {s["id"]: s for s in source_map.get("source_evidence_scopes", [])}
                for lesson in chapter.get("lessons", []):
                    for unit in lesson.get("units", []):
                        for block in unit.get("learning_blocks", []):
                            if any(scopes.get(scope, {}).get("section_id") not in owned_sections
                                   for scope in block.get("primary_evidence_scope_ids", [])):
                                raise LessonAuthorBlueprintValidationError(
                                    "BLUEPRINT_SOURCE_STRUCTURE_MISMATCH", "Primary evidence crosses the source chapter boundary.",
                                    path=f"chapters[{index}]", constraint="SOURCE_CHAPTER_PRIMARY_SCOPE")
    result["source_chapter_policy"] = deepcopy(policy)
    return result
