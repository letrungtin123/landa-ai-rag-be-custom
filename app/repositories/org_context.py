"""SQL of the organization context distillation: the learned chunks of exactly the requested context documents.

Every statement filters by tenant and knowledge base. The chunk reads are keyed by the document's active learned index
and walk ``idx_rag_chunks_scope`` (tenant, kb, document, index, chunk number): sizes first, in pages, then the texts of
the chunks the distillation reads only.
"""

from __future__ import annotations

from typing import Any

from app.repositories.executor import SqlExecutor


async def fetch_context_indexes(pool: SqlExecutor, tenant_id: Any, kb_id: Any, document_ids: Any) -> list[Any]:
    """The active learned self-built index of each requested document of the tenant's knowledge base."""
    return await pool.fetch(
        """
        SELECT r.id::text AS index_id,r.document_id::text AS document_id
        FROM rag_document_indexes r
        WHERE r.tenant_id=$1::uuid AND r.kb_id=$2::uuid AND r.document_id=ANY($3::uuid[])
          AND r.engine='self_built_rag' AND r.status='learned' AND r.is_active=true
        ORDER BY r.document_id
        """,
        tenant_id,
        kb_id,
        document_ids,
    )


async def fetch_context_chunk_sizes(
    pool: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    document_id: Any,
    index_id: Any,
    after_chunk_no: Any,
    limit: Any,
) -> list[Any]:
    """One keyset page of an index's chunk numbers and text lengths (characters), in source order."""
    return await pool.fetch(
        """
        SELECT c.chunk_no,char_length(c.content)::int AS chars
        FROM rag_chunks c
        WHERE c.tenant_id=$1::uuid AND c.kb_id=$2::uuid AND c.document_id=$3::uuid AND c.index_id=$4::uuid
          AND c.chunk_no>$5::int
        ORDER BY c.chunk_no
        LIMIT $6::int
        """,
        tenant_id,
        kb_id,
        document_id,
        index_id,
        after_chunk_no,
        limit,
    )


async def fetch_context_chunk_texts(
    pool: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    document_id: Any,
    index_id: Any,
    chunk_nos: Any,
) -> list[Any]:
    """The texts of the given chunks of one index, in source order."""
    return await pool.fetch(
        """
        SELECT c.chunk_no,c.content
        FROM rag_chunks c
        WHERE c.tenant_id=$1::uuid AND c.kb_id=$2::uuid AND c.document_id=$3::uuid AND c.index_id=$4::uuid
          AND c.chunk_no=ANY($5::int[])
        ORDER BY c.chunk_no
        """,
        tenant_id,
        kb_id,
        document_id,
        index_id,
        chunk_nos,
    )
