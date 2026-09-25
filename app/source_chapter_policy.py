"""Bounded server-owned chapter authority; never redistributes source facts."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.source_structure import MAX_NODES, strip_source_range_suffix
from app.lesson_author_blueprint import LessonAuthorBlueprintValidationError, MAX_BLUEPRINT_CHAPTERS


def resolve_source_chapter_policy(documents: list[dict[str, Any]], *, outline_complete: bool = True) -> dict[str, Any]:
    policy: dict[str, Any] = {"version": 1, "mode": "MODEL_DESIGNED", "complete": True,
                              "reason_codes": [], "chapters": []}

    def review(reason: str) -> None:
        policy["mode"] = "NEEDS_STRUCTURE_REVIEW"
        policy["complete"] = False
        if reason not in policy["reason_codes"]:
            policy["reason_codes"].append(reason)

    if not outline_complete:
        review("SOURCE_OUTLINE_INCOMPLETE")
    modes: list[str] = []
    for document in documents:
        structure = document.get("structure") or {}
        if not isinstance(structure, dict):
            review("SOURCE_STRUCTURE_UNAVAILABLE")
            continue
        nodes = structure.get("nodes") or []
        source = structure.get("structure_source")
        authority = structure.get("chapter_authority") or {}
        if not isinstance(nodes, list) or any(not isinstance(n, dict) or type(n.get("level", 1)) is not int for n in nodes):
            review("SOURCE_STRUCTURE_HIERARCHY_INVALID")
            continue
        if not isinstance(authority, dict):
            review("SOURCE_HEADING_AUTHORITY_INVALID")
            authority = {}
        if len(nodes) >= MAX_NODES or authority.get("complete") is False or any(
            code in {"SOURCE_STRUCTURE_CAPACITY_REACHED", "SOURCE_STRUCTURE_REBUILD_INCOMPLETE"}
            for code in structure.get("warnings", [])
        ):
            review("SOURCE_STRUCTURE_INCOMPLETE")
        if not nodes:
            review("SOURCE_STRUCTURE_UNAVAILABLE")
            continue
        refs = [n.get("source_ref") for n in nodes]
        if any(not isinstance(ref, str) or not ref for ref in refs) or len(set(refs)) != len(refs):
            review("SOURCE_STRUCTURE_REFERENCE_INVALID")
            continue
        by_ref = {n["source_ref"]: n for n in nodes}
        if any(n.get("parent_source_ref") and (
            n["parent_source_ref"] not in by_ref
            or by_ref[n["parent_source_ref"]].get("level", 1) >= n.get("level", 1)
        ) for n in nodes):
            review("SOURCE_STRUCTURE_PARENT_INVALID")
        roots = [n for n in nodes if n.get("level", 1) == 1]
        if source == "toc":
            confidence = structure.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0.9 <= confidence <= 1:
                review("SOURCE_TOC_AUTHORITY_UNVERIFIED")
            mode = "SOURCE_LOCKED_TOC"
        elif source == "heading_inferred" and authority.get("heading_refs"):
            if authority["heading_refs"] != [n["source_ref"] for n in roots]:
                review("SOURCE_HEADING_AUTHORITY_INVALID")
            mode = "SOURCE_LOCKED_HEADINGS"
        elif source == "heading_inferred" and any(n.get("number_label") for n in nodes):
            # Legacy numbering cannot distinguish a body list from a chapter
            # hierarchy. Do not turn lost parser evidence into model freedom.
            review("SOURCE_HEADING_AUTHORITY_UNVERIFIED")
            mode = "NEEDS_STRUCTURE_REVIEW"
        elif source in {"semantic_inferred", "heading_inferred"}:
            mode = "MODEL_DESIGNED"
        else:
            review("SOURCE_STRUCTURE_UNAVAILABLE")
            mode = "NEEDS_STRUCTURE_REVIEW"
        modes.append(mode)
        if mode.startswith("SOURCE_LOCKED"):
            if not roots:
                review("SOURCE_STRUCTURE_EMPTY_ROOTS")
            for root in roots:
                title = strip_source_range_suffix(str(root.get("title") or ""))
                if not title or not document.get("document_id"):
                    review("SOURCE_STRUCTURE_IDENTITY_INVALID")
                policy["chapters"].append({
                    "document_id": document.get("document_id"), "source_ref": root["source_ref"],
                    "title": title, "source_title": root.get("title"),
                    "basis": mode, "parser_version": structure.get("parser_version"),
                })
    # There is currently no approved cross-document chapter grouping contract.
    # Resolve each inventory, but do not merge documents by positional order or
    # conflate colliding src-001 identifiers. Human scope selection is needed.
    if len(documents) > 1 and any(mode != "MODEL_DESIGNED" for mode in modes):
        review("MULTI_DOCUMENT_CHAPTER_GROUPING_UNRESOLVED")
    if len(policy["chapters"]) > MAX_BLUEPRINT_CHAPTERS:
        review("SOURCE_CHAPTER_CAPACITY_EXCEEDED")
    if policy["mode"] != "NEEDS_STRUCTURE_REVIEW" and policy["chapters"]:
        policy["mode"] = modes[0]
    return policy


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
