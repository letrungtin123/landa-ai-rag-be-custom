from __future__ import annotations

"""Deterministic, provenance-preserving source representation for course design.

The map deliberately derives from the persisted source structure and fact
manifest. It does not ask a model to invent concepts or dependencies: every
concept is anchored to a parsed source section and every mapped fact retains
its document/page/chunk locator.
"""

import hashlib
import json
from collections import deque
from typing import Any, Iterable


SOURCE_MAP_VERSION = "source-map-v2"
MAX_SOURCE_MAP_ARCHITECT_CONTEXT_CHARS = 48_000
# These are provenance-partition bounds, not instructional-design targets.
# They are deliberately derived from the existing 3,200-character chunk
# contract and the current Course Architect's 24-unit / 48,000-character
# boundaries.  A hard context failure is safer than silently removing a
# source-evidence scope from the Architect.
FALLBACK_SCOPE_TARGET_EVIDENCE_CHARS = 4_800
FALLBACK_SCOPE_MAX_EVIDENCE_CHARS = 6_400
FALLBACK_SCOPE_MAX_FACTS = 32
MAX_ARCHITECT_EVIDENCE_SCOPES = 72
# Representative evidence is intentionally concise because each scope must
# receive at least one signal within the fixed global Architect context.  The
# full canonical fact text never enters this prompt.
MAX_SCOPE_REPRESENTATIVE_CHARS = 56


def _text(value: Any, maximum: int = 220) -> str:
    return " ".join(str(value or "").split())[:maximum].strip()


def _document_key(value: Any) -> str:
    return _text(value, 96) or "unknown-document"


def _section_id(document_id: str, source_ref: str) -> str:
    digest = hashlib.sha1(f"{document_id}:{source_ref}".encode("utf-8")).hexdigest()[:16]
    return f"src_section_{digest}"


def _concept_id(section_id: str) -> str:
    return f"concept_{section_id.removeprefix('src_section_')}"


def _page_number(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _scope_id(
    *,
    document_id: str,
    section_id: str,
    derivation_basis: str,
    source_ref: str,
    fact_ids: list[str],
) -> str:
    """Return a stable server-owned identifier for an exact fact partition."""

    membership = "\x1f".join(fact_ids)
    digest = hashlib.sha256(
        f"{document_id}\x1e{section_id}\x1e{derivation_basis}\x1e{source_ref}\x1e{membership}".encode("utf-8"),
    ).hexdigest()[:20]
    return f"evidence_scope_{digest}"


def _scope_evidence_size(facts: list[dict[str, Any]]) -> tuple[int, int]:
    """Return deterministic character and conservative token estimates."""

    characters = sum(len(_text(fact.get("text"), 4_000)) for fact in facts)
    # This is an operational upper-bound estimate only; it is not provider
    # usage telemetry and is never represented as Gemini token usage.
    return characters, max(0, (characters + 3) // 4)


def _split_fallback_scope_facts(facts: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split only contiguous provenance atoms using evidence weight first.

    Fact count is a secondary guard.  The caller already guarantees that all
    input facts belong to the same document/section and no trusted heading
    boundary is crossed.
    """

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for fact in facts:
        fact_chars = len(_text(fact.get("text"), 4_000))
        exceeds = current and (
            current_chars + fact_chars > FALLBACK_SCOPE_MAX_EVIDENCE_CHARS
            or len(current) >= FALLBACK_SCOPE_MAX_FACTS
        )
        if exceeds:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(fact)
        current_chars += fact_chars
    if current:
        groups.append(current)
    return groups


def _partition_provenance_atoms(atoms: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Merge contiguous page/chunk atoms only within their source section.

    Page and chunk boundaries are provenance locators, not semantic heading
    boundaries.  Keeping every atom as a separate scope would turn a document
    with many small chunks into hundreds of model-facing decisions.  This
    deterministic partition retains each locator on the scope while grouping
    adjacent atoms by evidence weight first and fact count second.
    """

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for atom in atoms:
        # A single physical atom can exceed a scope bound; split only that
        # contiguous atom before considering it for adjacent aggregation.
        for atom_part in _split_fallback_scope_facts(atom):
            part_chars, _part_tokens = _scope_evidence_size(atom_part)
            exceeds = current and (
                current_chars + part_chars > FALLBACK_SCOPE_TARGET_EVIDENCE_CHARS
                or len(current) + len(atom_part) > FALLBACK_SCOPE_MAX_FACTS
            )
            if exceeds:
                groups.append(current)
                current = []
                current_chars = 0
            current.extend(atom_part)
            current_chars += part_chars
    if current:
        groups.append(current)
    return groups


def _evidence_scopes(
    *,
    sections: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create exact server-owned provenance partitions without semantic claims."""

    manifest_by_id = {
        _text(value.get("fact_id"), 96): value
        for value in manifest.get("facts", []) if isinstance(value, dict) and _text(value.get("fact_id"), 96)
    }
    facts_by_section: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        fact_id = _text(fact.get("id"), 96)
        detail = manifest_by_id.get(fact_id)
        if not fact_id or detail is None:
            continue
        facts_by_section.setdefault(str(fact.get("section_id") or ""), []).append({
            **fact,
            "text": detail.get("text") or "",
        })
    concept_ids_by_section = {
        str(section.get("id") or ""): [str(concept.get("id") or "")]
        for section in sections
        for concept in concepts
        if str(section.get("id") or "") in {str(value) for value in concept.get("source_section_ids", [])}
    }
    scopes: list[dict[str, Any]] = []
    warnings: list[str] = []
    for section in sections:
        section_id = str(section.get("id") or "")
        section_facts = facts_by_section.get(section_id, [])
        if not section_facts:
            continue
        # Existing facts are already in canonical manifest order.  Group by a
        # physical page when present, otherwise by source chunk.  These are
        # provenance atoms only, never semantic labels.
        atoms: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_key: tuple[str, int | None] | None = None
        for fact in section_facts:
            page = _page_number(fact.get("page"))
            chunk = fact.get("chunk") if isinstance(fact.get("chunk"), int) else None
            key = ("page", page) if page is not None else ("chunk", chunk)
            if current and key != current_key:
                atoms.append(current)
                current = []
            current.append(fact)
            current_key = key
        if current:
            atoms.append(current)

        groups = _partition_provenance_atoms(atoms)
        for group_index, group in enumerate(groups, start=1):
            fact_ids = [str(fact["id"]) for fact in group]
            char_count, token_estimate = _scope_evidence_size(group)
            pages = [value for value in (_page_number(fact.get("page")) for fact in group) if value is not None]
            chunks = [fact.get("chunk") for fact in group if isinstance(fact.get("chunk"), int)]
            has_page = bool(pages)
            has_chunk = bool(chunks)
            fallback_basis = (
                "FALLBACK_PAGE_RANGE" if has_page and len(set(pages)) > 1 else
                "FALLBACK_PAGE_FACT_ORDINAL_RANGE" if has_page else
                "FALLBACK_CHUNK_RANGE" if has_chunk and len(set(chunks)) > 1 else
                "FALLBACK_CHUNK_FACT_ORDINAL_RANGE" if has_chunk else
                "FALLBACK_FACT_ORDINAL_RANGE"
            )
            trusted_descendant = (
                bool(section.get("parent_id"))
                and str(section.get("structure_source") or "") == "toc"
            )
            basis = (
                "TRUSTED_DESCENDANT_HEADING"
                if trusted_descendant and len(groups) == 1
                else f"TRUSTED_DESCENDANT_HEADING_{fallback_basis.removeprefix('FALLBACK_')}"
                if trusted_descendant
                else fallback_basis
            )
            source_ref = str(section.get("source_ref") or "")
            scopes.append({
                "id": _scope_id(
                    document_id=str(section.get("document_id") or ""),
                    section_id=section_id,
                    derivation_basis=basis,
                    source_ref=source_ref,
                    fact_ids=fact_ids,
                ),
                "document_id": str(section.get("document_id") or ""),
                "section_id": section_id,
                "concept_ids": [item for item in concept_ids_by_section.get(section_id, []) if item],
                "source_ref": source_ref,
                "parent_source_ref": section.get("parent_source_ref"),
                "heading_path": [str(section.get("title") or "")] if section.get("title") else [],
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "chunk_start": min(chunks) if chunks else None,
                "chunk_end": max(chunks) if chunks else None,
                "fact_count": len(fact_ids),
                "evidence_char_count": char_count,
                "evidence_token_estimate": token_estimate,
                "source_fact_ids": fact_ids,
                "derivation_basis": basis,
                "provenance_complete": True,
                "ordinal": group_index,
            })
    return scopes, list(dict.fromkeys(warnings))


def _section_for_fact(
    fact: dict[str, Any],
    sections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    document_id = _document_key(fact.get("document_id"))
    source_ref = _text(fact.get("source_ref"), 64).casefold()
    candidates = [section for section in sections if section["document_id"] == document_id]
    if source_ref:
        direct = next((section for section in candidates if section["source_ref"].casefold() == source_ref), None)
        if direct is not None:
            return direct
    page = _page_number(fact.get("source_page"))
    if page is not None:
        paged = [
            section for section in candidates
            if _page_number(section.get("page_start")) is not None
        ]
        # A parsed heading applies until the following parsed heading. Use the
        # nearest preceding section; this is deterministic provenance, not a
        # semantic inference.
        preceding = [section for section in paged if int(section["page_start"]) <= page]
        if preceding:
            return max(preceding, key=lambda section: (int(section["page_start"]), section["order"]))
        if paged:
            return min(paged, key=lambda section: (int(section["page_start"]), section["order"]))
    return candidates[0] if len(candidates) == 1 else None


def build_source_map(
    source_structure_nodes: Iterable[dict[str, Any]],
    source_coverage_manifest: dict[str, Any] | None,
    *,
    locale: str = "vi",
) -> dict[str, Any]:
    """Build a compact global map without losing section/fact provenance.

    `source_structure_nodes` is the existing normalized
    ``rag_document_structure_nodes`` representation (or the compatible chunk
    metadata fallback). The function does not persist or mutate it.
    """
    raw_nodes = [node for node in source_structure_nodes if isinstance(node, dict)]
    sections: list[dict[str, Any]] = []
    source_ref_to_section: dict[tuple[str, str], str] = {}
    # The full-source scope is bounded before Source Map construction. Do not
    # add a second flat section cap here: it would make canonical coverage
    # dependent on a presentation limit instead of actual source extraction.
    for index, node in enumerate(raw_nodes):
        document_id = _document_key(node.get("document_id"))
        source_ref = _text(node.get("source_ref"), 64)
        title = _text(node.get("title"))
        if not source_ref or not title:
            continue
        section_id = _section_id(document_id, source_ref)
        parent_ref = _text(node.get("parent_source_ref"), 64)
        parent_id = source_ref_to_section.get((document_id, parent_ref.casefold())) if parent_ref else None
        try:
            level = max(1, min(6, int(node.get("level") or 1)))
        except (TypeError, ValueError):
            level = 1
        try:
            order = max(0, int(node.get("order") if node.get("order") is not None else index))
        except (TypeError, ValueError):
            order = index
        page = _page_number(node.get("logical_page")) or _page_number(node.get("page"))
        section = {
            "id": section_id,
            "document_id": document_id,
            "document_name": _text(node.get("document_name"), 180) or "Source document",
            "source_ref": source_ref,
            "title": title,
            "level": level,
            "parent_id": parent_id,
            "parent_source_ref": parent_ref or None,
            "order": order,
            "page_start": page,
            "page_end": page,
            "node_type": _text(node.get("node_type"), 40) or "section",
            "structure_source": _text(node.get("structure_source"), 40) or "legacy_unknown",
            "structure_confidence": node.get("confidence") if isinstance(node.get("confidence"), (float, int)) else None,
            "source_fact_ids": [],
        }
        sections.append(section)
        source_ref_to_section[(document_id, source_ref.casefold())] = section_id

    # Keep document provenance even when an older index has no headings.
    if not sections:
        document_ids = sorted({_document_key(fact.get("document_id")) for fact in (source_coverage_manifest or {}).get("facts", []) if isinstance(fact, dict)})
        for index, document_id in enumerate(document_ids):
            source_ref = f"document-{index + 1:03d}"
            sections.append({
                "id": _section_id(document_id, source_ref),
                "document_id": document_id,
                "document_name": "Source document",
                "source_ref": source_ref,
                "title": "Source document",
                "level": 1,
                "parent_id": None,
                "parent_source_ref": None,
                "order": index,
                "page_start": None,
                "page_end": None,
                "node_type": "document",
                "structure_source": "synthetic",
                "structure_confidence": 0.0,
                "source_fact_ids": [],
            })

    facts: list[dict[str, Any]] = []
    unmapped_fact_ids: list[str] = []
    # Facts are canonical provenance records. Prompt detail is bounded later
    # by build_course_architect_context, never by this representation.
    for fact in (source_coverage_manifest or {}).get("facts", []):
        if not isinstance(fact, dict):
            continue
        fact_id = _text(fact.get("fact_id"), 96)
        if not fact_id:
            continue
        section = _section_for_fact(fact, sections)
        if section is None:
            unmapped_fact_ids.append(fact_id)
            continue
        section["source_fact_ids"].append(fact_id)
        facts.append({
            "id": fact_id,
            "section_id": section["id"],
            "document_id": _document_key(fact.get("document_id")) or section["document_id"],
            "source_ref": _text(fact.get("source_ref"), 64) or section["source_ref"],
            "page": _page_number(fact.get("source_page")),
            "chunk": fact.get("source_chunk") if isinstance(fact.get("source_chunk"), int) else None,
        })

    sections_by_id = {section["id"]: section for section in sections}
    concepts: list[dict[str, Any]] = []
    for section in sections:
        parent_concepts = []
        if section["parent_id"] and section["parent_id"] in sections_by_id:
            parent_concepts = [_concept_id(section["parent_id"])]
        is_core = section["level"] == 1 or section["node_type"] in {"chapter", "document"}
        concepts.append({
            "id": _concept_id(section["id"]),
            "name": section["title"],
            # Description intentionally repeats only the section title; no
            # model-derived explanation is treated as source knowledge.
            "description": section["title"],
            "source_section_ids": [section["id"]],
            "source_fact_ids": list(section["source_fact_ids"]),
            "prerequisite_concept_ids": parent_concepts,
            "importance": "core" if is_core else "supporting",
            "relationship_evidence": "source_hierarchy" if parent_concepts else "none",
        })

    documents: list[dict[str, Any]] = []
    for document_id in sorted({section["document_id"] for section in sections}):
        document_sections = [section for section in sections if section["document_id"] == document_id]
        documents.append({
            "id": document_id,
            "title": document_sections[0]["document_name"] if document_sections else "Source document",
            "language": locale,
            "source_section_ids": [section["id"] for section in document_sections],
        })

    manifest = source_coverage_manifest or {}
    manifest_fact_count = len([
        fact for fact in manifest.get("facts", [])
        if isinstance(fact, dict) and _text(fact.get("fact_id"), 96)
    ])
    total_fact_count = manifest.get("total_fact_count")
    if not isinstance(total_fact_count, int) or total_fact_count < 0:
        total_fact_count = manifest_fact_count
    manifest_complete = bool(manifest.get("fact_scope_complete", not manifest.get("truncated")))
    incomplete_reason = _text(manifest.get("incomplete_reason"), 96) or None
    section_incomplete = bool(manifest.get("section_scope_incomplete"))
    warnings: list[str] = []
    fact_scope_complete = (
        manifest_complete
        and total_fact_count == manifest_fact_count
        and total_fact_count == len(facts)
        and not unmapped_fact_ids
    )
    if not fact_scope_complete:
        warnings.append("SOURCE_MAP_FACT_SCOPE_INCOMPLETE")
    if section_incomplete:
        warnings.append("SOURCE_MAP_SECTION_SCOPE_INCOMPLETE")
    if unmapped_fact_ids:
        warnings.append("SOURCE_MAP_UNMAPPED_FACTS")
    fact_partitions = [
        {
            "section_id": section["id"],
            "document_id": section["document_id"],
            "source_ref": section["source_ref"],
            "fact_count": len(section["source_fact_ids"]),
            "source_fact_ids": list(section["source_fact_ids"]),
        }
        for section in sections
    ]
    evidence_scopes, scope_warnings = _evidence_scopes(
        sections=sections,
        concepts=concepts,
        facts=facts,
        manifest=manifest,
    )
    scope_fact_ids = [
        fact_id
        for scope in evidence_scopes
        for fact_id in scope.get("source_fact_ids", [])
        if isinstance(fact_id, str) and fact_id
    ]
    evidence_scope_complete = (
        len(scope_fact_ids) == len(facts)
        and len(set(scope_fact_ids)) == len(scope_fact_ids)
        and set(scope_fact_ids) == {str(fact.get("id") or "") for fact in facts}
    )
    if not evidence_scope_complete:
        warnings.append("SOURCE_MAP_EVIDENCE_SCOPE_INCOMPLETE")
    warnings.extend(scope_warnings)
    return {
        "version": SOURCE_MAP_VERSION,
        "documents": documents,
        "sections": sections,
        "concepts": concepts,
        "facts": facts,
        "fact_partitions": fact_partitions,
        "source_evidence_scopes": evidence_scopes,
        "coverage": {
            "section_count": len(sections),
            "source_fact_count": len(facts),
            "mapped_fact_count": len(facts),
            "total_fact_count": total_fact_count,
            "represented_fact_count": len(facts),
            "concept_count": len(concepts),
            "evidence_scope_count": len(evidence_scopes),
            "evidence_scope_complete": evidence_scope_complete,
            "unmapped_fact_ids": unmapped_fact_ids[:80],
            "incomplete_reason": incomplete_reason,
            "fact_scope_complete": fact_scope_complete,
            "section_scope_complete": not section_incomplete,
        },
        "warnings": warnings,
    }


def _compact_global_hierarchy(source_map: dict[str, Any]) -> dict[str, Any]:
    """Return the complete global hierarchy without atomic fact text."""
    sections = source_map.get("sections") if isinstance(source_map.get("sections"), list) else []
    concepts = source_map.get("concepts") if isinstance(source_map.get("concepts"), list) else []
    coverage = source_map.get("coverage") if isinstance(source_map.get("coverage"), dict) else {}
    return {
        "source_map_version": source_map.get("version"),
        "coverage": coverage,
        "documents": [
            {
                "id": document.get("id"),
                "title": _text(document.get("title"), 120),
                "source_section_ids": document.get("source_section_ids", []),
            }
            for document in source_map.get("documents", []) if isinstance(document, dict)
        ],
        "sections": [
            {
                "id": section.get("id"), "source_ref": section.get("source_ref"),
                "title": _text(section.get("title"), 120), "level": section.get("level"),
                "parent_id": section.get("parent_id"), "document_id": section.get("document_id"),
                "page_start": section.get("page_start"),
                "source_fact_count": len(section.get("source_fact_ids") or []),
            }
            for section in sections if isinstance(section, dict)
        ],
        "concepts": [
            {
                "id": concept.get("id"), "name": _text(concept.get("name"), 120),
                "source_section_ids": concept.get("source_section_ids", []),
                "prerequisite_concept_ids": concept.get("prerequisite_concept_ids", []),
                "importance": concept.get("importance"),
                "source_fact_count": len(concept.get("source_fact_ids") or []),
            }
            for concept in concepts if isinstance(concept, dict)
        ],
        "source_evidence_scope_count": len(source_map.get("source_evidence_scopes") or []),
    }


def _fact_detail_buckets(
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
) -> list[deque[dict[str, Any]]]:
    """Partition detail by section so early long sections cannot starve later ones."""
    details_by_id = {
        _text(fact.get("fact_id"), 96): fact
        for fact in (source_coverage_manifest or {}).get("facts", [])
        if isinstance(fact, dict) and _text(fact.get("fact_id"), 96)
    }
    sections_by_id = {
        str(section.get("id")): section
        for section in source_map.get("sections", []) if isinstance(section, dict)
    }
    buckets: list[deque[dict[str, Any]]] = []
    for partition in source_map.get("fact_partitions", []):
        if not isinstance(partition, dict):
            continue
        section = sections_by_id.get(str(partition.get("section_id") or ""))
        if section is None:
            continue
        bucket: deque[dict[str, Any]] = deque()
        for raw_fact_id in partition.get("source_fact_ids", []):
            fact_id = _text(raw_fact_id, 96)
            fact = details_by_id.get(fact_id)
            if fact is None:
                continue
            bucket.append({
                "section_id": section["id"],
                "source_ref": section["source_ref"],
                "fact_id": fact_id,
                "page": _page_number(fact.get("source_page")),
                "chunk": fact.get("source_chunk") if isinstance(fact.get("source_chunk"), int) else None,
                "text": _text(fact.get("text"), 420),
            })
        if bucket:
            buckets.append(bucket)
    return buckets


def _scope_representative_samples(
    scope: dict[str, Any],
    details_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """Return bounded beginning/middle/end evidence without exposing fact IDs."""

    fact_ids = [str(value) for value in scope.get("source_fact_ids", []) if isinstance(value, str) and value]
    if not fact_ids:
        return []
    indexes = sorted({0, len(fact_ids) // 2, len(fact_ids) - 1})
    samples: list[str] = []
    for index in indexes:
        detail = details_by_id.get(fact_ids[index])
        text = _text((detail or {}).get("text"), MAX_SCOPE_REPRESENTATIVE_CHARS)
        if text and text not in samples:
            samples.append(text)
    return samples


def _scope_context_descriptors(
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    details_by_id = {
        _text(fact.get("fact_id"), 96): fact
        for fact in (source_coverage_manifest or {}).get("facts", [])
        if isinstance(fact, dict) and _text(fact.get("fact_id"), 96)
    }
    descriptors: list[dict[str, Any]] = []
    for scope in source_map.get("source_evidence_scopes", []):
        if not isinstance(scope, dict):
            continue
        # The Architect receives this compact transport shape only. The
        # canonical map retains full metric/provenance fields server-side;
        # repeating them for every scope would crowd out representative signal
        # without helping the model choose scope IDs.
        descriptor = {
            "i": scope.get("id"),
            # The global hierarchy already maps source_ref -> section ->
            # canonical concept ID. Avoid repeating those long identifiers for
            # every bounded scope descriptor.
            "r": scope.get("source_ref"),
            "h": scope.get("heading_path", []),
            # The Architect may claim only concepts compatible with this
            # server-owned evidence scope. This compact allowlist contains no
            # canonical fact IDs or allocation metadata.
            "c": sorted({
                _text(value, 96)
                for value in scope.get("concept_ids", [])
                if _text(value, 96)
            }),
            # Empty initially. The context builder adds representative samples
            # only after it proves every descriptor fits in the fixed budget.
            "e": [],
        }
        descriptor["_samples"] = _scope_representative_samples(scope, details_by_id)
        descriptors.append(descriptor)
    return descriptors


def build_course_architect_context(
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
    *,
    max_chars: int = MAX_SOURCE_MAP_ARCHITECT_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Return bounded global course-design evidence without losing canonical facts.

    The full document hierarchy and concept inventory always remain present.
    Atomic fact detail is selected round-robin by source section under an
    explicit request budget. Facts omitted from this *prompt* stay represented
    in the canonical map and are allocated server-side after architecture.
    """
    coverage = source_map.get("coverage") if isinstance(source_map.get("coverage"), dict) else {}
    hierarchy = _compact_global_hierarchy(source_map)
    total_fact_count = coverage.get("total_fact_count")
    if not isinstance(total_fact_count, int) or total_fact_count < 0:
        total_fact_count = int(coverage.get("source_fact_count") or 0)
    is_v2 = source_map.get("version") == SOURCE_MAP_VERSION
    scope_descriptors = _scope_context_descriptors(source_map, source_coverage_manifest) if is_v2 else []
    base = {
        **hierarchy,
        "architect_context_mode": "evidence_scope" if is_v2 else "hierarchical",
        "architect_detail_fact_count": 0,
        "architect_fact_details": [],
        **({"source_evidence_scopes": [
            {key: value for key, value in descriptor.items() if key != "_samples"}
            for descriptor in scope_descriptors
        ]} if is_v2 else {}),
    }
    base_text = json.dumps(base, ensure_ascii=False, separators=(",", ":"))
    diagnostics = {
        "source_total_facts": total_fact_count,
        "source_total_sections": int(coverage.get("section_count") or 0),
        "source_total_concepts": int(coverage.get("concept_count") or len(source_map.get("concepts", []))),
        "source_map_complete": bool(coverage.get("fact_scope_complete")) and bool(coverage.get("section_scope_complete")),
        "architect_context_mode": "evidence_scope" if is_v2 else "hierarchical",
        "architect_context_size": len(base_text),
        "architect_detail_fact_count": 0,
        "source_evidence_scope_count": len(scope_descriptors),
        "architect_scope_descriptor_count": len(scope_descriptors),
        "architect_scope_sample_count": 0,
    }
    if is_v2 and (len(scope_descriptors) > MAX_ARCHITECT_EVIDENCE_SCOPES or not coverage.get("evidence_scope_complete")):
        return {
            "context": "",
            "context_complete": False,
            "error_code": "EVIDENCE_SCOPE_CONTEXT_INCOMPLETE",
            "diagnostics": diagnostics,
        }
    if len(base_text) > max_chars:
        return {
            "context": "",
            "context_complete": False,
            # V5 must distinguish a complete global hierarchy that cannot fit
            # from an evidence-scope inventory that cannot be represented.
            # The latter includes the compact server-owned compatible-concept
            # allowlists and must fail with the scope-specific contract code;
            # omitting descriptors or their allowlists would weaken provenance
            # and repair authority.
            "error_code": "EVIDENCE_SCOPE_CONTEXT_INCOMPLETE" if is_v2 else "ARCHITECT_CONTEXT_CANNOT_REPRESENT_SOURCE",
            "diagnostics": diagnostics,
        }

    if is_v2:
        # First give every scope one bounded representative signal.  Then add
        # middle/end samples round-robin.  Scope descriptors are never removed
        # to make room, so the Architect either receives the complete inventory
        # or the request fails closed.
        used_chars = len(base_text)
        sample_count = 0
        for sample_index in range(3):
            for descriptor in scope_descriptors:
                samples = descriptor.get("_samples", [])
                if sample_index >= len(samples):
                    continue
                candidate = list(descriptor["e"]) + [samples[sample_index]]
                prior = descriptor["e"]
                descriptor["e"] = candidate
                next_text = json.dumps({
                    **hierarchy,
                    "architect_context_mode": "evidence_scope",
                    "architect_detail_fact_count": 0,
                    "architect_fact_details": [],
                    "source_evidence_scopes": [
                        {key: value for key, value in value.items() if key != "_samples"}
                        for value in scope_descriptors
                    ],
                }, ensure_ascii=False, separators=(",", ":"))
                if len(next_text) > max_chars:
                    descriptor["e"] = prior
                    if sample_index == 0:
                        return {
                            "context": "",
                            "context_complete": False,
                            "error_code": "EVIDENCE_SCOPE_CONTEXT_INCOMPLETE",
                            "diagnostics": diagnostics,
                        }
                    continue
                used_chars = len(next_text)
                sample_count += 1
        context = {
            **hierarchy,
            "architect_context_mode": "evidence_scope",
            "architect_detail_fact_count": 0,
            "architect_fact_details": [],
            "source_evidence_scopes": [
                {key: value for key, value in descriptor.items() if key != "_samples"}
                for descriptor in scope_descriptors
            ],
        }
        text = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        diagnostics.update({
            "architect_context_size": len(text),
            "architect_scope_sample_count": sample_count,
        })
        return {"context": text, "context_complete": True, "error_code": None, "diagnostics": diagnostics}

    details: list[dict[str, Any]] = []
    used_chars = len(base_text)
    buckets = _fact_detail_buckets(source_map, source_coverage_manifest)
    while buckets:
        next_buckets: list[deque[dict[str, Any]]] = []
        progress = False
        for bucket in buckets:
            detail = bucket.popleft()
            detail_size = len(json.dumps(detail, ensure_ascii=False, separators=(",", ":"))) + 1
            if used_chars + detail_size <= max_chars:
                details.append(detail)
                used_chars += detail_size
                progress = True
            if bucket:
                next_buckets.append(bucket)
        if not progress:
            break
        buckets = next_buckets

    context = {
        **hierarchy,
        "architect_context_mode": "detailed" if len(details) == total_fact_count else "hierarchical",
        "architect_detail_fact_count": len(details),
        "architect_fact_details": details,
    }
    text = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    # Defend against JSON escaping estimation differences without silently
    # dropping global hierarchy. Only representative detail may be reduced.
    while details and len(text) > max_chars:
        details.pop()
        context["architect_context_mode"] = "hierarchical"
        context["architect_detail_fact_count"] = len(details)
        context["architect_fact_details"] = details
        text = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    if len(text) > max_chars:
        return {
            "context": "",
            "context_complete": False,
            "error_code": "ARCHITECT_CONTEXT_CANNOT_REPRESENT_SOURCE",
            "diagnostics": diagnostics,
        }
    diagnostics.update({
        "architect_context_mode": context["architect_context_mode"],
        "architect_context_size": len(text),
        "architect_detail_fact_count": len(details),
    })
    return {
        "context": text,
        "context_complete": True,
        "error_code": None,
        "diagnostics": diagnostics,
    }


def format_source_map_for_course_architect(
    source_map: dict[str, Any],
    *,
    max_chars: int = MAX_SOURCE_MAP_ARCHITECT_CONTEXT_CHARS,
) -> tuple[str, bool]:
    """Return a global hierarchy/concept inventory or an explicit truncation.

    The function never silently removes trailing sections. The caller must
    reject/ask for a narrower source scope if the compact global map still
    exceeds the configured budget.
    """
    text = json.dumps(_compact_global_hierarchy(source_map), ensure_ascii=False, separators=(",", ":"))
    return text, len(text) > max_chars
