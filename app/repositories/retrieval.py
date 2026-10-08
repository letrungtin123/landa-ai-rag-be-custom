"""SQL of retrieval: learned chunks (scoped, blueprint, vector and keyword matches) and stored source structure."""

from __future__ import annotations

from typing import Any

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
        tenant_id,
        kb_id,
        document_ids,
        embedding_model,
        embedding_dimensions,
        limit,
    )


async def fetch_vector_matches(
    pool: SqlExecutor,
    tenant_id: Any,
    kb_id: Any,
    document_ids: Any,
    embedding: Any,
    limit: Any,
    embedding_model: Any,
    embedding_dimensions: Any,
) -> list[Any]:
    """Nearest learned chunks to one query embedding."""
    return await pool.fetch(
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
    """Learned chunks matching the query's phrases/terms, ranked by keyword hits and similarity."""
    return await pool.fetch(
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
        tenant_id,
        kb_id,
        document_ids,
    )
