"""Vector and keyword retrieval of chunks for chat and lesson authoring."""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from app.core.config import settings
from app.core.logging import SERVICE_LOGGER_NAME
from app.repositories.pgvector import vector_literal
from app.schemas.chat import RagChatRequest
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorBlueprintRequest
from app.schemas.orchestration_v2 import RagLessonAuthorSourceSnapshotV2Request
from app.services import provider
from app.services.provider import normalize_embedding_model
from app.services.retrieval.query import (
    apply_retrieval_quality_controls,
    build_keyword_patterns,
    build_retrieval_query_texts,
    build_target_source_scopes,
    decode_json_object,
    merge_retrieval_rows,
    normalize_query_text,
    retrieval_candidate_limit,
    retrieval_limits,
    retrieval_text_signature,
)
from app.services.retrieval.source_coverage import (
    build_source_coverage_manifest,
    lesson_author_blueprint_draft_fact_ids,
    lesson_author_blueprint_draft_supporting_fact_ids,
    lesson_author_target_source_refs,
    restrict_blueprint_draft_source_manifest,
)
from app.services.retrieval.structure import load_source_structure_context
from app.services.text import clean_text

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def load_target_source_scope_chunks(
    pool: asyncpg.Pool,
    request: RagChatRequest,
    scopes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not scopes or not request.kb_id:
        return [], {"candidate_count": 0, "truncated": False, "pages": []}
    rows: list[dict[str, Any]] = []
    remaining = max(1, settings.lesson_author_scope_max_chunks)
    candidate_count = 0
    truncated = False
    for scope in scopes:
        if remaining <= 0:
            break
        if not scope.get("document_id"):
            continue
        scoped_rows = await pool.fetch(
            """
            SELECT c.content,
                   c.source_page,
                   c.source_section,
                   c.metadata,
                   c.chunk_no,
                   d.id::text AS document_id,
                   d.name AS document_name,
                   1.0::float AS score,
                   0.0::float AS vector_score,
                   0.0::float AS keyword_score,
                   'source_scope' AS method
            FROM rag_chunks c
            JOIN rag_document_indexes r ON r.id = c.index_id
            JOIN kb_documents d ON d.id = c.document_id
            WHERE c.tenant_id = $1::uuid
              AND c.kb_id = $2::uuid
              AND c.document_id = $3::uuid
              AND r.engine = 'self_built_rag'
              AND r.status = 'learned'
              AND r.is_active = true
              AND r.embedding_model = $6
              AND r.embedding_dimensions = $7::int
              AND c.source_page BETWEEN $4::int AND $5::int
            ORDER BY c.source_page ASC NULLS LAST, c.chunk_no ASC
            LIMIT ($8::int + 1)
            """,
            request.tenant_id,
            request.kb_id,
            scope["document_id"],
            scope["start_page"],
            scope["end_page"],
            normalize_embedding_model(request.embedding_model),
            request.embedding_dimensions,
            remaining,
        )
        candidate_count += len(scoped_rows)
        if len(scoped_rows) > remaining:
            truncated = True
            scoped_rows = scoped_rows[:remaining]
        for row in scoped_rows:
            item = dict(row)
            metadata = decode_json_object(item.get("metadata")) or {}
            if scope.get("source_ref"):
                metadata["source_ref"] = scope["source_ref"]
                metadata["heading_path"] = scope.get("title")
            item["metadata"] = json.dumps(metadata, ensure_ascii=False)
            rows.append(item)
        remaining -= len(scoped_rows)
    return rows, {
        "candidate_count": candidate_count,
        "truncated": truncated,
        "pages": sorted({
            int(row["source_page"])
            for row in rows
            if isinstance(row.get("source_page"), int) and row["source_page"] > 0
        }),
    }


async def load_lesson_author_blueprint_source_chunks(
    pool: asyncpg.Pool,
    request: RagChatRequest,
) -> tuple[list[dict[str, Any]], bool]:
    """Load selected source documents in source order for a Blueprint contract.

    A course Blueprint cannot safely be planned from only the top semantic
    retrieval hits: doing so lets one slide become the architecture for an
    entire source chapter. A Blueprint-locked chapter draft needs this same
    snapshot before it narrows the coverage manifest to its persisted fact IDs.
    """
    if not request.kb_id or not request.source_documents:
        return [], False
    document_ids = [document.document_id for document in request.source_documents]
    limit = max(1, settings.lesson_author_scope_max_chunks)
    fetched = await pool.fetch(
        """
        SELECT c.content,
               c.source_page,
               c.source_section,
               c.metadata,
               c.chunk_no,
               d.id::text AS document_id,
               d.name AS document_name,
               1.0::float AS score,
               1.0::float AS vector_score,
               1.0::float AS keyword_score,
               'blueprint_source_scope' AS method
        FROM rag_chunks c
        JOIN rag_document_indexes r ON r.id = c.index_id
        JOIN kb_documents d ON d.id = c.document_id
        WHERE c.tenant_id = $1::uuid
          AND c.kb_id = $2::uuid
          AND c.document_id = ANY($3::uuid[])
          AND r.engine = 'self_built_rag'
          AND r.status = 'learned'
          AND r.is_active = true
          AND r.embedding_model = $4
          AND r.embedding_dimensions = $5::int
        ORDER BY d.id, c.source_page ASC NULLS LAST, c.chunk_no ASC
        LIMIT ($6::int + 1)
        """,
        request.tenant_id,
        request.kb_id,
        document_ids,
        normalize_embedding_model(request.embedding_model),
        request.embedding_dimensions,
        limit,
    )
    truncated = len(fetched) > limit
    return [dict(row) for row in fetched[:limit]], truncated


async def retrieve_chunks(
    pool: asyncpg.Pool,
    request: RagChatRequest,
) -> tuple[list[dict[str, Any]], AiUsage, dict[str, Any]]:
    structure_context = await load_source_structure_context(pool, request)
    if not request.kb_id:
        return [], AiUsage(), structure_context

    limits = retrieval_limits(request)
    query_texts = build_retrieval_query_texts(request)
    embeddings, usage = await provider.embed_texts(
        request.api_key,
        request.embedding_model,
        query_texts,
        task_type="QUESTION_ANSWERING",
        output_dimensionality=request.embedding_dimensions,
    )
    doc_ids = [doc.document_id for doc in request.source_documents] or None
    candidate_limit = retrieval_candidate_limit(request)
    vector_rows: list[asyncpg.Record] = []
    for embedding in embeddings:
        vector_rows.extend(
            await pool.fetch(
                """
                SELECT c.content,
                       c.source_page,
                       c.source_section,
                       c.metadata,
                       c.chunk_no,
                       d.id::text AS document_id,
                       d.name AS document_name,
                       1 - (c.embedding <=> $4::vector) AS score,
                       1 - (c.embedding <=> $4::vector) AS vector_score,
                       0::float AS keyword_score,
                       'vector' AS method
                FROM rag_chunks c
                JOIN rag_document_indexes r ON r.id = c.index_id
                JOIN kb_documents d ON d.id = c.document_id
                WHERE c.tenant_id = $1::uuid
                  AND c.kb_id = $2::uuid
                  AND r.engine = 'self_built_rag'
                  AND r.status = 'learned'
                  AND r.is_active = true
                  AND r.embedding_model = $6
                  AND r.embedding_dimensions = $7::int
                  AND ($3::uuid[] IS NULL OR c.document_id = ANY($3::uuid[]))
                ORDER BY c.embedding <=> $4::vector
                LIMIT $5
                """,
                request.tenant_id,
                request.kb_id,
                doc_ids,
                vector_literal(embedding),
                candidate_limit,
                normalize_embedding_model(request.embedding_model),
                request.embedding_dimensions,
            )
        )

    phrase_patterns, term_patterns, terms = build_keyword_patterns(request.user_message)
    keyword_rows: list[asyncpg.Record] = []
    if phrase_patterns or term_patterns:
        keyword_rows = await pool.fetch(
            """
            SELECT c.content,
                   c.source_page,
                   c.source_section,
                   c.metadata,
                   c.chunk_no,
                   d.id::text AS document_id,
                   d.name AS document_name,
                   LEAST(1.0, GREATEST(
                     CASE WHEN phrase_hits.hit_count > 0 THEN 0.99 ELSE 0 END,
                     CASE WHEN term_hits.hit_count > 0
                       THEN LEAST(0.94, 0.58 + (term_hits.hit_count::float / GREATEST($6::int, 1)) * 0.36)
                       ELSE 0
                     END,
                     similarity(d.name, $7),
                     LEAST(0.90, similarity(c.content, $7))
                   )) AS score,
                   0::float AS vector_score,
                   LEAST(1.0, GREATEST(
                     CASE WHEN phrase_hits.hit_count > 0 THEN 0.99 ELSE 0 END,
                     CASE WHEN term_hits.hit_count > 0
                       THEN LEAST(0.94, 0.58 + (term_hits.hit_count::float / GREATEST($6::int, 1)) * 0.36)
                       ELSE 0
                     END,
                     similarity(d.name, $7),
                     LEAST(0.90, similarity(c.content, $7))
                   )) AS keyword_score,
                   'keyword' AS method
            FROM rag_chunks c
            JOIN rag_document_indexes r ON r.id = c.index_id
            JOIN kb_documents d ON d.id = c.document_id
            LEFT JOIN LATERAL (
              SELECT COUNT(*)::int AS hit_count
              FROM unnest($4::text[]) AS pattern(value)
              WHERE c.content ILIKE pattern.value OR d.name ILIKE pattern.value
            ) phrase_hits ON true
            LEFT JOIN LATERAL (
              SELECT COUNT(*)::int AS hit_count
              FROM unnest($5::text[]) AS pattern(value)
              WHERE c.content ILIKE pattern.value OR d.name ILIKE pattern.value
            ) term_hits ON true
            WHERE c.tenant_id = $1::uuid
              AND c.kb_id = $2::uuid
              AND r.engine = 'self_built_rag'
              AND r.status = 'learned'
              AND r.is_active = true
              AND r.embedding_model = $9
              AND r.embedding_dimensions = $10::int
              AND ($3::uuid[] IS NULL OR c.document_id = ANY($3::uuid[]))
              AND (
                c.content ILIKE ANY($4::text[])
                OR d.name ILIKE ANY($4::text[])
                OR c.content ILIKE ANY($5::text[])
                OR d.name ILIKE ANY($5::text[])
              )
            ORDER BY keyword_score DESC, c.chunk_no ASC
            LIMIT $8
            """,
            request.tenant_id,
            request.kb_id,
            doc_ids,
            phrase_patterns or ["__landa_no_phrase_match__"],
            term_patterns or ["__landa_no_term_match__"],
            len(terms),
            normalize_query_text(request.user_message),
            candidate_limit,
            normalize_embedding_model(request.embedding_model),
            request.embedding_dimensions,
        )

    target_scopes = build_target_source_scopes(request, structure_context)
    structure_context["target_source_scopes"] = target_scopes
    scope_rows, scope_diagnostics = await load_target_source_scope_chunks(pool, request, target_scopes)
    structure_context["target_source_scope_candidate_count"] = scope_diagnostics["candidate_count"]
    structure_context["target_source_scope_truncated"] = scope_diagnostics["truncated"]
    structure_context["target_source_scope_pages"] = scope_diagnostics["pages"]
    expected_scope_pages = sorted({
        page
        for scope in target_scopes
        for page in range(
            int(scope.get("start_page") or 0),
            int(scope.get("end_page") or 0) + 1,
        )
        if page > 0
    })
    structure_context["target_source_scope_expected_pages"] = expected_scope_pages[:400]
    structure_context["target_source_scope_missing_pages"] = [
        page for page in expected_scope_pages[:400] if page not in scope_diagnostics["pages"]
    ]

    merged_rows = merge_retrieval_rows(vector_rows, keyword_rows, candidate_limit, scope_rows)
    blueprint_draft_fact_ids = lesson_author_blueprint_draft_fact_ids(request)
    blueprint_draft_supporting_fact_ids = lesson_author_blueprint_draft_supporting_fact_ids(request)
    blueprint_source_rows: list[dict[str, Any]] = []
    blueprint_source_truncated = False
    if (isinstance(request, (RagLessonAuthorBlueprintRequest, RagLessonAuthorSourceSnapshotV2Request))
            or blueprint_draft_fact_ids or blueprint_draft_supporting_fact_ids):
        blueprint_source_rows, blueprint_source_truncated = await load_lesson_author_blueprint_source_chunks(
            pool,
            request,
        )
    if (blueprint_draft_fact_ids or blueprint_draft_supporting_fact_ids) and blueprint_source_rows:
        # The persisted Blueprint allocation is authoritative. A TOC range can
        # be off by a slide (for example a continuation slide); using it here
        # changes the manifest and makes a valid Blueprint impossible to draft.
        rows = blueprint_source_rows
        structure_context["blueprint_draft_source_contract"] = True
        structure_context["blueprint_draft_source_scope_truncated"] = blueprint_source_truncated
        structure_context["target_source_scope_hard_locked"] = True
        structure_context["out_of_scope_retrieval_count"] = 0
    elif target_scopes:
        # A selected chapter is an authoritative boundary. Relevance-ranked
        # chunks outside that range are never allowed to fill the context and
        # silently contaminate the generated lesson plan.
        rows: list[dict[str, Any]] = []
        seen_scope_content: set[str] = set()
        for row in sorted(
            scope_rows,
            key=lambda item: (
                int(item.get("source_page") or 0),
                int(item.get("chunk_no") or 0),
            ),
        ):
            signature = retrieval_text_signature(str(row.get("content") or ""))
            if signature in seen_scope_content:
                continue
            seen_scope_content.add(signature)
            rows.append(row)
        structure_context["target_source_scope_hard_locked"] = True
        scope_keys = {
            (str(row.get("document_id") or ""), int(row.get("chunk_no") or 0))
            for row in scope_rows
        }
        structure_context["out_of_scope_retrieval_count"] = sum(
            1
            for row in merged_rows
            if (
                str(row.get("document_id") or ""),
                int(row.get("chunk_no") or 0),
            ) not in scope_keys
        )
    elif blueprint_source_rows:
        # Blueprint design is whole-source work. Preserve deterministic source
        # order instead of discarding all but relevance-ranked chunks.
        rows = blueprint_source_rows
        structure_context["course_blueprint_source_scope_hard_locked"] = True
        structure_context["course_blueprint_source_scope_truncated"] = blueprint_source_truncated
        structure_context["out_of_scope_retrieval_count"] = 0
    else:
        rows = apply_retrieval_quality_controls(
            merged_rows,
            limits["top_k"],
            max_chunks_per_document=limits["max_chunks_per_document"],
        )
        structure_context["target_source_scope_hard_locked"] = False
        structure_context["out_of_scope_retrieval_count"] = 0
    structure_context["retrieval_candidate_count"] = len(merged_rows)
    target_source_refs = lesson_author_target_source_refs(request)
    source_coverage_manifest = (
        build_source_coverage_manifest(
            rows,
            structure_nodes=structure_context.get("source_structure_nodes", []),
            target_source_refs=target_source_refs,
            target_scopes=target_scopes,
        )
        if request.target == "lesson_author"
        else None
    )
    if (blueprint_draft_fact_ids or blueprint_draft_supporting_fact_ids) and source_coverage_manifest is not None:
        source_coverage_manifest, missing_blueprint_fact_ids = restrict_blueprint_draft_source_manifest(
            source_coverage_manifest,
            blueprint_draft_fact_ids,
            blueprint_draft_supporting_fact_ids,
        )
        structure_context["blueprint_draft_source_fact_ids"] = sorted(blueprint_draft_fact_ids)
        structure_context["blueprint_draft_supporting_evidence_fact_ids"] = sorted(blueprint_draft_supporting_fact_ids)
        structure_context["blueprint_draft_missing_source_fact_ids"] = missing_blueprint_fact_ids
    structure_context["source_coverage_manifest"] = source_coverage_manifest
    covered_refs = {
        str((decode_json_object(row.get("metadata")) or {}).get("source_ref"))
        for row in rows
        if str((decode_json_object(row.get("metadata")) or {}).get("source_ref") or "").strip()
    }
    covered_refs.update(
        str(scope.get("source_ref"))
        for scope in target_scopes
        if str(scope.get("source_ref") or "").strip()
        and any(str(row.get("document_id") or "") == str(scope.get("document_id") or "") for row in scope_rows)
    )
    structure_context["covered_source_refs"] = covered_refs
    logger.info(
        "rag_retrieve_completed",
        extra={"event": "rag_retrieve_completed", "tenant_id": request.tenant_id,
               "kb_id": request.kb_id, "query_count": len(query_texts),
               "vector_count": len(vector_rows), "keyword_count": len(keyword_rows),
               "scope_count": len(scope_rows), "merged_count": len(merged_rows),
               "accepted_count": len(rows)},
    )
    return rows, usage, structure_context


def format_sources(
    rows: list[dict[str, Any]],
    *,
    max_context_chars: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    current_chars = 0
    context_limit = max_context_chars or settings.max_context_chars
    for index, row in enumerate(rows, start=1):
        metadata = decode_json_object(row.get("metadata")) or {}
        source_ref = str(metadata.get("source_ref") or "").strip()
        label = f"Nguồn {index}"
        if source_ref:
            label += f" [{source_ref}]"
        label += f": {row['document_name']}"
        if row["source_page"]:
            label += f", trang/slide {row['source_page']}"
        if row["source_section"]:
            label += f", mục {row['source_section']}"
        content = clean_text(row["content"])
        if current_chars + len(content) > context_limit:
            break
        context_parts.append(f"[{label}]\n{content}")
        current_chars += len(content)
        sources.append(
            {
                "document_id": row["document_id"],
                "document_name": row["document_name"],
                "source_page": row["source_page"],
                "source_section": row["source_section"],
                "score": float(row["score"] or 0),
                "vector_score": float(row.get("vector_score") or 0),
                "keyword_score": float(row.get("keyword_score") or 0),
                "method": row.get("method") or "unknown",
                "methods": row.get("methods") or [],
                "source_ref": metadata.get("source_ref"),
                "heading_path": metadata.get("heading_path"),
            }
        )
    return "\n\n".join(context_parts), sources


def build_retrieval_diagnostics(
    request: RagChatRequest,
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    structure_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    methods = sorted(
        {
            method
            for row in rows
            for method in (row.get("methods") or [row.get("method") or "unknown"])
            if method
        }
    )
    top_row = rows[0] if rows else None
    limits = retrieval_limits(request)
    target_scope_hard_locked = bool(structure_context.get("target_source_scope_hard_locked")) if structure_context else False
    target_scope_truncated = bool(structure_context.get("target_source_scope_truncated")) if structure_context else False
    target_scope_missing_pages = structure_context.get("target_source_scope_missing_pages", []) if structure_context else []
    reason: str | None = None
    if not request.kb_id:
        reason = "missing_kb_id"
    # Missing page numbers can represent blank/textless PDF pages. Only an
    # actual scope/chunk/context truncation makes the retrieval incomplete.
    elif target_scope_hard_locked and (target_scope_truncated or not rows):
        reason = "target_source_scope_incomplete"
    elif not rows:
        reason = "no_confident_matching_chunks"
    elif not sources:
        reason = "context_limit_exhausted"
    elif len(sources) < len(rows):
        reason = "context_limit_exhausted"

    known_refs = structure_context.get("known_source_refs", set()) if structure_context else set()
    covered_refs = structure_context.get("covered_source_refs", set()) if structure_context else set()
    coverage_ratio = len(covered_refs) / len(known_refs) if known_refs else None
    return {
        "kb_id": request.kb_id,
        "source_document_count": len(request.source_documents),
        "retrieved_count": len(rows),
        "returned_source_count": len(sources),
        "top_score": float(top_row["score"]) if top_row and top_row.get("score") is not None else None,
        "top_document_name": top_row.get("document_name") if top_row else None,
        "methods": methods,
        "top_k": limits["top_k"],
        "max_context_chars": limits["max_context_chars"],
        "min_score": settings.retrieval_min_score,
        "keyword_min_score": settings.retrieval_keyword_min_score,
        "max_chunks_per_document": limits["max_chunks_per_document"],
        "target_source_scope_count": len(structure_context.get("target_source_scopes", [])) if structure_context else 0,
        "target_source_scope_chunk_count": sum(
            1
            for row in rows
            if "source_scope" in (row.get("methods") or [row.get("method") or ""])
        ),
        "context_chars": sum(len(clean_text(str(row.get("content") or ""))) for row in rows[: len(sources)]),
        "retrieval_candidate_count": structure_context.get("retrieval_candidate_count", len(rows)) if structure_context else len(rows),
        "context_truncated": len(sources) < len(rows),
        "omitted_retrieved_count": max(0, len(rows) - len(sources)),
        "target_source_scope_candidate_count": structure_context.get("target_source_scope_candidate_count", 0) if structure_context else 0,
        "target_source_scope_hard_locked": target_scope_hard_locked,
        "target_source_scope_pages": structure_context.get("target_source_scope_pages", []) if structure_context else [],
        "target_source_scope_expected_pages": structure_context.get("target_source_scope_expected_pages", []) if structure_context else [],
        "target_source_scope_missing_pages": target_scope_missing_pages,
        "target_source_scope_truncated": target_scope_truncated,
        "out_of_scope_retrieval_count": structure_context.get("out_of_scope_retrieval_count", 0) if structure_context else 0,
        "structure_source": structure_context.get("structure_source") if structure_context else None,
        "structure_confidence": structure_context.get("structure_confidence") if structure_context else None,
        "structure_node_count": structure_context.get("structure_node_count", 0) if structure_context else 0,
        "source_structure_warnings": structure_context.get("source_structure_warnings", []) if structure_context else [],
        "known_source_ref_count": len(known_refs),
        "covered_source_ref_count": len(covered_refs),
        "source_coverage_ratio": round(coverage_ratio, 4) if coverage_ratio is not None else None,
        "reason": reason,
    }


def target_source_scope_is_incomplete(
    structure_context: dict[str, Any],
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    source_coverage_manifest: dict[str, Any] | None,
) -> bool:
    """Return whether a hard-locked source scope cannot safely be generated.

    PDF/PPT extractors intentionally omit blank or textless pages. A page
    number gap is therefore diagnostic evidence, not proof that source data
    was lost. The hard guard must still reject an empty scope, chunk-limit
    truncation, or context truncation so valid source material is never
    silently replaced by out-of-scope retrievals.
    """
    if not structure_context.get("target_source_scope_hard_locked"):
        return False
    return bool(
        not rows
        or structure_context.get("target_source_scope_truncated")
        or len(sources) < len(rows)
        or bool((source_coverage_manifest or {}).get("truncated"))
    )
