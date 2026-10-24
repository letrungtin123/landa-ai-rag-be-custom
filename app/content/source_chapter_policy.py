"""Bounded server-owned chapter authority of a source inventory; never redistributes source facts.

Moved verbatim from ``app.source_chapter_policy`` (refactor wave 3b) with the chapter limit it applies, which the
legacy Blueprint contract (``app.lesson_author_blueprint``) imports from here; binding a Blueprint to the chapters
stays in the legacy module.
"""

from __future__ import annotations

from typing import Any

from app.content.source_structure import MAX_NODES, PARSER_VERSION, strip_source_range_suffix

# Keep the generation contract aligned with the backend (the legacy Blueprint's chapter limit).
MAX_BLUEPRINT_CHAPTERS = 12


def resolve_source_chapter_policy(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve authority from the full inventory, never its display summary.

    Current parser proof of *no* chapter authority permits pedagogical design;
    absent legacy proof or an incomplete/invalid inventory does not.
    """
    policy: dict[str, Any] = {"version": 1, "mode": "MODEL_DESIGNED", "complete": True,
                              "reason_codes": [], "chapters": []}

    def review(reason: str) -> None:
        policy["mode"] = "NEEDS_STRUCTURE_REVIEW"
        policy["complete"] = False
        if reason not in policy["reason_codes"]:
            policy["reason_codes"].append(reason)

    modes: list[str] = []
    for document in documents:
        structure = document.get("structure") or {}
        if not isinstance(structure, dict):
            review("SOURCE_STRUCTURE_UNAVAILABLE")
            continue
        nodes = structure.get("nodes") or []
        source = structure.get("structure_source")
        authority = structure.get("chapter_authority") or {}
        if not isinstance(nodes, list) or any(not isinstance(n, dict) or type(n.get("level", 1)) is not int
                                              for n in nodes):
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
        elif (source == "heading_inferred"
              and structure.get("parser_version") == PARSER_VERSION
              and type(authority.get("version")) is int and authority["version"] == 1
              and authority.get("complete") is True
              and authority.get("basis") == "NONE"
              and authority.get("heading_refs") == []):
            # Body numbering, slide headings and repeated headers are evidence
            # hints, not verified chapter bindings. The current parser has
            # explicitly completed authority detection; no provenance is lost
            # by leaving chapter grouping to the evidence-grounded Architect.
            mode = "MODEL_DESIGNED"
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
