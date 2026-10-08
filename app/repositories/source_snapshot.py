"""SQL of the orchestration-v2 source snapshot: learned index revisions and source pages."""

from __future__ import annotations

from typing import Any

from app.repositories.executor import SqlExecutor


async def fetch_learned_indexes(
    pool: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    document_ids: Any,
    embedding_model: Any,
    embedding_dimensions: Any,
) -> list[Any]:
    """Active learned index of each selected document."""
    return await pool.fetch(
        """
        SELECT r.id::text AS index_id,r.document_id::text AS document_id,r.content_sha256,
               r.chunk_count,r.embedding_model,r.embedding_dimensions
        FROM rag_document_indexes r
        WHERE r.tenant_id=$1::uuid AND r.kb_id=$2::uuid AND r.document_id=ANY($3::uuid[])
          AND r.engine='self_built_rag' AND r.status='learned' AND r.is_active=true
          AND r.embedding_model=$4 AND r.embedding_dimensions=$5::int
        ORDER BY r.document_id
        """,
        tenant_id,
        kb_id,
        document_ids,
        embedding_model,
        embedding_dimensions,
    )


async def fetch_index_evidence_summary(pool: SqlExecutor, tenant_id: Any, kb_id: Any, index_ids: Any) -> list[Any]:
    """Per-document source structure and evidence-revision counts of the selected indexes."""
    return await pool.fetch(
        """
        SELECT c.document_id::text AS document_id,
               (jsonb_agg(c.metadata->'source_structure' ORDER BY c.chunk_no)
                 FILTER (WHERE c.metadata ? 'source_structure'))->0 AS source_structure,
               min(lower(c.metadata->>'source_evidence_revision')) FILTER (
                 WHERE coalesce(c.metadata->>'source_evidence_revision','') ~ '^[0-9a-fA-F]{64}$'
               ) AS source_evidence_revision,
               count(*)::integer AS actual_chunk_count,
               count(*) FILTER (
                 WHERE coalesce(c.metadata->>'source_evidence_revision','') ~ '^[0-9a-fA-F]{64}$'
               )::integer AS evidence_revision_chunk_count,
               count(DISTINCT lower(c.metadata->>'source_evidence_revision')) FILTER (
                 WHERE coalesce(c.metadata->>'source_evidence_revision','') ~ '^[0-9a-fA-F]{64}$'
               )::integer AS evidence_revision_distinct_count
        FROM rag_chunks c
        WHERE c.tenant_id=$1::uuid AND c.kb_id=$2::uuid AND c.index_id=ANY($3::uuid[])
        GROUP BY c.document_id
        ORDER BY c.document_id
        """,
        tenant_id,
        kb_id,
        index_ids,
    )


async def fetch_source_page(
    pool: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    index_ids: Any,
    after_document_id: Any,
    after_chunk_no: Any,
    limit: Any,
) -> list[Any]:
    """One keyset page of learned chunks after the cursor."""
    return await pool.fetch(
        """
        SELECT c.content,c.source_page,c.source_section,c.metadata,c.chunk_no,c.index_id::text AS index_id,
               c.document_id::text AS document_id,d.name AS document_name
        FROM rag_chunks c
        JOIN kb_documents d ON d.id=c.document_id AND d.tenant_id=c.tenant_id AND d.kb_id=c.kb_id
        WHERE c.tenant_id=$1::uuid AND c.kb_id=$2::uuid AND c.index_id=ANY($3::uuid[])
          AND ($4::uuid IS NULL OR c.document_id>$4::uuid
               OR (c.document_id=$4::uuid AND c.chunk_no>$5::int))
        ORDER BY c.document_id,c.chunk_no
        LIMIT $6::int
        """,
        tenant_id,
        kb_id,
        index_ids,
        after_document_id,
        after_chunk_no,
        limit,
    )
