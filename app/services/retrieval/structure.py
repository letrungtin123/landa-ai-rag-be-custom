"""Stored source-structure (outline) context, rebuilt from chunks when stale."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import asyncpg

from app.schemas.chat import RagChatRequest
from app.services.ingestion.extract import ExtractedSection
from app.services.retrieval.query import decode_json_list, decode_json_object
from app.source_chapter_policy import resolve_source_chapter_policy
from app.source_structure import PARSER_VERSION, analyze_source_structure, structure_outline

MAX_SOURCE_STRUCTURE_DOCUMENTS = 40
MAX_SOURCE_OUTLINE_CHARS = 16000


def build_source_structure_context(
    documents: list[dict[str, Any]],
    *,
    locale: Literal["vi", "en"],
) -> dict[str, Any]:
    known_refs: set[str] = set()
    warnings: list[str] = []
    sources: set[str] = set()
    confidences: list[float] = []
    outline_parts: list[str] = []
    authoritative_source_nodes: list[dict[str, Any]] = []
    source_structure_nodes: list[dict[str, Any]] = []
    outline_chars = 0
    outline_display_truncated = False
    node_count = 0
    for document in documents:
        structure = document.get("structure") or {}
        nodes = [node for node in structure.get("nodes", []) if isinstance(node, dict)]
        node_count += len(nodes)
        known_refs.update(
            str(node.get("source_ref"))
            for node in nodes
            if str(node.get("source_ref") or "").strip()
        )
        source = str(structure.get("structure_source") or "semantic_inferred")
        sources.add(source)
        document_id = str(document.get("document_id") or "")
        document_name = str(document.get("document_name") or "Tài liệu nguồn")
        source_structure_nodes.extend(
            {
                **node,
                "document_id": document_id,
                "document_name": document_name,
                "structure_source": source,
            }
            for node in nodes
            if str(node.get("source_ref") or "").strip()
        )
        if source == "toc":
            for node in nodes:
                try:
                    level = int(node.get("level") or 1)
                except (TypeError, ValueError):
                    level = 1
                title = str(node.get("title") or "").strip()
                source_ref = str(node.get("source_ref") or "").strip()
                if level == 1 and title and source_ref:
                    authoritative_source_nodes.append(
                        {
                            **node,
                            "document_id": document_id,
                            "document_name": document_name,
                        },
                    )
        try:
            confidences.append(float(structure.get("confidence") or 0))
        except (TypeError, ValueError):
            confidences.append(0)
        warnings.extend(str(item) for item in structure.get("warnings", []) if str(item).strip())
        outline = structure_outline(structure, max_chars=5000, locale=locale)
        if "... source structure truncated ..." in outline or "... cấu trúc nguồn đã được rút gọn ..." in outline:
            outline_display_truncated = True
            warnings.append("SOURCE_OUTLINE_TRUNCATED")
        if outline:
            name = str(document.get("document_name") or "Tài liệu nguồn")
            part = f"Tài liệu: {name}\n{outline}" if locale != "en" else f"Document: {name}\n{outline}"
            if outline_chars + len(part) > MAX_SOURCE_OUTLINE_CHARS:
                warnings.append("SOURCE_OUTLINE_TRUNCATED")
                outline_display_truncated = True
                continue  # Keep inventory for every document even if display is full.
            outline_parts.append(part)
            outline_chars += len(part)

    if len(sources) == 1:
        structure_source = next(iter(sources))
    elif sources:
        structure_source = "mixed"
    else:
        structure_source = None
    confidence = min(confidences) if confidences else None
    return {
        "outline": "\n\n".join(outline_parts),
        "source_outline_display_truncated": outline_display_truncated,
        "structure_source": structure_source,
        "structure_confidence": round(confidence, 4) if confidence is not None else None,
        "structure_node_count": node_count,
        "known_source_refs": known_refs,
        "covered_source_refs": set(),
        "source_structure_warnings": list(dict.fromkeys(warnings))[:12],
        "authoritative_source_nodes": authoritative_source_nodes,
        "source_structure_nodes": source_structure_nodes,
        # All nodes above remain available to the global Source Map. A bounded
        # display string is not the authority for source inventory completeness.
        "source_chapter_policy": resolve_source_chapter_policy(documents),
    }


async def rebuild_stale_source_structures_from_chunks(
    pool: asyncpg.Pool,
    request: RagChatRequest,
    document_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Reparse bounded stored chunks for indexes created by an older parser."""
    if not document_ids:
        return {}
    repair_rows = await pool.fetch(
        """
        SELECT c.document_id::text AS document_id,
               d.name AS document_name,
               c.content,
               c.source_page,
               c.chunk_no
        FROM rag_chunks c
        JOIN rag_document_indexes r ON r.id = c.index_id
        JOIN kb_documents d ON d.id = c.document_id
        WHERE c.tenant_id = $1::uuid
          AND c.kb_id = $2::uuid
          AND r.engine = 'self_built_rag'
          AND r.status = 'learned'
          AND r.is_active = true
          AND c.document_id = ANY($3::uuid[])
        ORDER BY c.document_id, c.chunk_no ASC
        LIMIT 480
        """,
        request.tenant_id,
        request.kb_id,
        document_ids,
    )
    repair_sections: dict[str, list[ExtractedSection]] = {}
    repair_names: dict[str, str] = {}
    for row in repair_rows:
        document_id = str(row["document_id"])
        repair_names[document_id] = str(row["document_name"] or "Tài liệu nguồn")
        repair_sections.setdefault(document_id, []).append(
            ExtractedSection(
                text=str(row["content"] or ""),
                page=row["source_page"],
            ),
        )

    repaired: dict[str, dict[str, Any]] = {}
    for document_id, sections in repair_sections.items():
        structure = await asyncio.to_thread(analyze_source_structure, sections)
        if len(repair_rows) >= 480:
            structure["warnings"].append("SOURCE_STRUCTURE_REBUILD_INCOMPLETE")
        repaired[document_id] = {
            "document_id": document_id,
            "document_name": repair_names.get(document_id, "Tài liệu nguồn"),
            "structure": structure,
        }
    return repaired


async def load_source_structure_context(
    pool: asyncpg.Pool,
    request: RagChatRequest,
) -> dict[str, Any]:
    if not request.kb_id or request.target != "lesson_author":
        return build_source_structure_context([], locale=request.locale)
    doc_ids = [doc.document_id for doc in request.source_documents] or None
    documents_by_id: dict[str, dict[str, Any]] = {}
    stale_document_ids: set[str] = set()

    # Prefer the normalized table when the optional production migration is
    # present. The chunk metadata query below keeps rollout backward compatible.
    try:
        normalized_rows = await pool.fetch(
            """
            SELECT n.document_id::text AS document_id,
                   d.name AS document_name,
                   MAX(n.confidence)::float AS confidence,
                   jsonb_agg(
                     jsonb_build_object(
                       'source_ref', n.source_ref,
                       'title', n.title,
                       'level', n.level,
                       'order', n.sort_order,
                       'page', n.page_start,
                       'logical_page', n.logical_page,
                       'number_label', n.number_label,
                       'parent_source_ref', n.parent_source_ref,
                       'node_type', n.node_type,
                       'confidence', n.confidence
                     ) ORDER BY n.sort_order
                   ) AS nodes,
                   ((array_agg(n.metadata ORDER BY n.sort_order))[1])::text AS metadata
            FROM rag_document_structure_nodes n
            JOIN rag_document_indexes r ON r.id = n.index_id
            JOIN kb_documents d ON d.id = n.document_id
            WHERE n.tenant_id = $1::uuid
              AND n.kb_id = $2::uuid
              AND r.engine = 'self_built_rag'
              AND r.status = 'learned'
              AND r.is_active = true
              AND ($3::uuid[] IS NULL OR n.document_id = ANY($3::uuid[]))
            GROUP BY n.document_id, d.name
            ORDER BY n.document_id
            LIMIT 40
            """,
            request.tenant_id,
            request.kb_id,
            doc_ids,
        )
        for row in normalized_rows:
            metadata = decode_json_object(row["metadata"]) or {}
            documents_by_id[str(row["document_id"])] = {
                "document_id": str(row["document_id"]),
                "document_name": row["document_name"],
                "structure": {
                    # Rows created before parser versioning may not contain
                    # this field. Treat them as legacy so they are re-read
                    # from stored chunks instead of trusted as current.
                    "parser_version": metadata.get("parser_version", "source-structure-v1"),
                    "structure_source": metadata.get("structure_source", "semantic_inferred"),
                    "confidence": float(row["confidence"] or 0),
                    "warnings": decode_json_list(metadata.get("warnings")),
                    "chapter_authority": metadata.get("chapter_authority"),
                    "nodes": decode_json_list(row["nodes"]),
                },
            }
            if metadata.get("parser_version") != PARSER_VERSION:
                stale_document_ids.add(str(row["document_id"]))
    except asyncpg.exceptions.UndefinedTableError:
        pass

    fallback_rows = await pool.fetch(
        """
        SELECT DISTINCT ON (c.document_id)
               c.document_id::text AS document_id,
               d.name AS document_name,
               c.metadata->'source_structure' AS source_structure
        FROM rag_chunks c
        JOIN rag_document_indexes r ON r.id = c.index_id
        JOIN kb_documents d ON d.id = c.document_id
        WHERE c.tenant_id = $1::uuid
          AND c.kb_id = $2::uuid
          AND r.engine = 'self_built_rag'
          AND r.status = 'learned'
          AND r.is_active = true
          AND c.metadata ? 'source_structure'
          AND ($3::uuid[] IS NULL OR c.document_id = ANY($3::uuid[]))
        ORDER BY c.document_id, c.chunk_no ASC
        LIMIT 40
        """,
        request.tenant_id,
        request.kb_id,
        doc_ids,
    )
    for row in fallback_rows:
        document_id = str(row["document_id"])
        if document_id in documents_by_id:
            continue
        structure = decode_json_object(row["source_structure"])
        if structure:
            if str(structure.get("parser_version") or "source-structure-v1") != PARSER_VERSION:
                stale_document_ids.add(document_id)
            documents_by_id[document_id] = {
                "document_id": document_id,
                "document_name": row["document_name"],
                "structure": structure,
            }

    # Existing indexes may contain structure metadata from an older parser.
    # Re-read a bounded set of already stored chunks so Blueprint generation
    # can benefit from the current parser without re-embedding or mutating DB
    # state inside a chat request. The normal background reindex remains the
    # durable repair path for all chunk metadata.
    repaired_documents = await rebuild_stale_source_structures_from_chunks(
        pool,
        request,
        sorted(stale_document_ids),
    )
    documents_by_id.update(repaired_documents)

    return build_source_structure_context(list(documents_by_id.values()), locale=request.locale)
