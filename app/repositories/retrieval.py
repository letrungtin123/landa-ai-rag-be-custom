"""SQL of retrieval: learned chunks (scoped, blueprint, vector and keyword matches) and stored source structure.

Every query also matches the tenant of the joined index and document rows to the tenant of the chunk or node
(audit 2026-10-24, L6): in the join, or for the ranked vector and keyword matches outside the ranked query, whose
plan the extra join clause changed (review of wave 2, F1).
"""

from __future__ import annotations

from typing import Any, Literal

import asyncpg

from app.repositories.executor import SqlExecutor


async def fetch_scope_chunks(
    pool: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    document_id: Any,
    start_page: Any,
    end_page: Any,
    embedding_model: Any,
    embedding_dimensions: Any,
    limit: Any,
) -> list[Any]:
    """Learned chunks of one document page range, in source order (one row past the limit)."""
    return await pool.fetch(
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
        JOIN rag_document_indexes r ON r.id = c.index_id AND r.tenant_id = c.tenant_id
        JOIN kb_documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id
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
        tenant_id,
        kb_id,
        document_id,
        start_page,
        end_page,
        embedding_model,
        embedding_dimensions,
        limit,
    )


async def fetch_blueprint_source_chunks(
    pool: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    document_ids: Any,
    embedding_model: Any,
    embedding_dimensions: Any,
    limit: Any,
) -> list[Any]:
    """Learned chunks of the selected documents, in source order (one row past the limit)."""
    return await pool.fetch(
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
        JOIN rag_document_indexes r ON r.id = c.index_id AND r.tenant_id = c.tenant_id
        JOIN kb_documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id
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
        tenant_id,
        kb_id,
        document_ids,
        embedding_model,
        embedding_dimensions,
        limit,
    )


async def fetch_vector_matches(
    pool: asyncpg.Pool,
    tenant_id: Any,
    kb_id: Any,
    document_ids: Any,
    embedding: Any,
    limit: Any,
    embedding_model: Any,
    embedding_dimensions: Any,
    *,
    ef_search: int,
    iterative_scan: Literal["off", "strict_order"],
) -> list[Any]:
    """Nearest learned chunks to one query embedding, the HNSW scan tuned for this query (audit 2026-10-24, P3).

    ``ef_search`` 0 runs the query with the database's settings. Otherwise ``hnsw.ef_search`` and, when the
    installed pgvector has it (0.8+), ``hnsw.iterative_scan`` are set for this transaction only: one pooled
    connection, set_config(..., true), then the query.
    """
    args = (tenant_id, kb_id, document_ids, embedding, limit, embedding_model, embedding_dimensions)
    if ef_search <= 0:
        return await _vector_matches(pool, *args)
    async with pool.acquire() as conn, conn.transaction():
        # The vector literal loads pgvector's library while this statement is parsed, so hnsw.iterative_scan is
        # defined exactly when the installed pgvector has it; an older pgvector gets ef_search only.
        await conn.execute(
            """
            SELECT set_config('hnsw.ef_search', $1::text, true),
                   CASE WHEN $2::text <> 'off' AND current_setting('hnsw.iterative_scan', true) IS NOT NULL
                        THEN set_config('hnsw.iterative_scan', $2::text, true)
                   END
            FROM (SELECT '[1]'::vector) AS loaded
            """,
            str(ef_search),
            iterative_scan,
        )
        return await _vector_matches(conn, *args)


async def _vector_matches(
    executor: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    document_ids: Any,
    embedding: Any,
    limit: Any,
    embedding_model: Any,
    embedding_dimensions: Any,
) -> list[Any]:
    # The inner query is the nearest-neighbour query as it was before the tenant match (its columns, FROM, WHERE,
    # ORDER BY and LIMIT; ``distance`` is the ORDER BY expression, which the planner computed anyway), so the
    # planner costs and chooses its plans as before: the HNSW index scan where it did, the same rows. The tenant
    # of the joined index and document rows is checked outside, on its at most ``limit`` rows; PostgreSQL neither
    # flattens a subquery with a LIMIT nor pushes the outer filter into it. The tenant match inside the joins
    # lowered the row estimates and turned the vector query into an exact sort of the KB's chunks, returning
    # other rows than before (review of wave 2, F1).
    return await executor.fetch(
        """
        SELECT m.content,
               m.source_page,
               m.source_section,
               m.metadata,
               m.chunk_no,
               m.document_id,
               m.document_name,
               m.score,
               m.vector_score,
               m.keyword_score,
               m.method
        FROM (
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
                 'vector' AS method,
                 c.embedding <=> $4::vector AS distance,
                 r.tenant_id AS index_tenant_id,
                 d.tenant_id AS document_tenant_id
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
        ) m
        WHERE m.index_tenant_id = $1::uuid
          AND m.document_tenant_id = $1::uuid
        ORDER BY m.distance
        """,
        tenant_id,
        kb_id,
        document_ids,
        embedding,
        limit,
        embedding_model,
        embedding_dimensions,
    )


async def fetch_keyword_matches(
    pool: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    document_ids: Any,
    phrase_patterns: Any,
    term_patterns: Any,
    term_count: Any,
    query_text: Any,
    limit: Any,
    embedding_model: Any,
    embedding_dimensions: Any,
) -> list[Any]:
    """Learned chunks matching the query's phrases/terms, ranked by keyword hits and similarity.

    The ranked query is the one from before the tenant match, so its plan and its rows (ties included) stay the
    same; the tenant of the joined index and document rows is checked outside it, as in ``_vector_matches``
    (review of wave 2, F1).
    """
    return await pool.fetch(
        """
        SELECT m.content,
               m.source_page,
               m.source_section,
               m.metadata,
               m.chunk_no,
               m.document_id,
               m.document_name,
               m.score,
               m.vector_score,
               m.keyword_score,
               m.method
        FROM (
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
                 'keyword' AS method,
                 r.tenant_id AS index_tenant_id,
                 d.tenant_id AS document_tenant_id
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
        ) m
        WHERE m.index_tenant_id = $1::uuid
          AND m.document_tenant_id = $1::uuid
        ORDER BY m.keyword_score DESC, m.chunk_no ASC
        """,
        tenant_id,
        kb_id,
        document_ids,
        phrase_patterns,
        term_patterns,
        term_count,
        query_text,
        limit,
        embedding_model,
        embedding_dimensions,
    )


async def fetch_chunks_for_structure_rebuild(
    pool: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    document_ids: Any,
) -> list[Any]:
    """Bounded learned chunks of documents whose stored structure predates the current parser."""
    return await pool.fetch(
        """
        SELECT c.document_id::text AS document_id,
               d.name AS document_name,
               c.content,
               c.source_page,
               c.chunk_no
        FROM rag_chunks c
        JOIN rag_document_indexes r ON r.id = c.index_id AND r.tenant_id = c.tenant_id
        JOIN kb_documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id
        WHERE c.tenant_id = $1::uuid
          AND c.kb_id = $2::uuid
          AND r.engine = 'self_built_rag'
          AND r.status = 'learned'
          AND r.is_active = true
          AND c.document_id = ANY($3::uuid[])
        ORDER BY c.document_id, c.chunk_no ASC
        LIMIT 480
        """,
        tenant_id,
        kb_id,
        document_ids,
    )


async def fetch_structure_nodes(pool: SqlExecutor, tenant_id: Any, kb_id: Any, document_ids: Any) -> list[Any]:
    """Stored structure nodes per learned document (optional table)."""
    return await pool.fetch(
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
        JOIN rag_document_indexes r ON r.id = n.index_id AND r.tenant_id = n.tenant_id
        JOIN kb_documents d ON d.id = n.document_id AND d.tenant_id = n.tenant_id
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
        tenant_id,
        kb_id,
        document_ids,
    )


async def fetch_chunk_source_structures(pool: SqlExecutor, tenant_id: Any, kb_id: Any, document_ids: Any) -> list[Any]:
    """Source structure kept in chunk metadata (fallback when no structure nodes exist)."""
    return await pool.fetch(
        """
        SELECT DISTINCT ON (c.document_id)
               c.document_id::text AS document_id,
               d.name AS document_name,
               c.metadata->'source_structure' AS source_structure
        FROM rag_chunks c
        JOIN rag_document_indexes r ON r.id = c.index_id AND r.tenant_id = c.tenant_id
        JOIN kb_documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id
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
        tenant_id,
        kb_id,
        document_ids,
    )
