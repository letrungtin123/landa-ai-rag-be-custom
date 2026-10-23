"""SQL of document indexing: the source document, index rows and stored structure nodes."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, cast

import asyncpg

from app.core.logging import SERVICE_LOGGER_NAME, log_event
from app.repositories.executor import SqlExecutor

logger = logging.getLogger(SERVICE_LOGGER_NAME)

# Chunk rows per INSERT statement (audit 2026-10-24, P1). asyncpg's executemany runs one statement per row, so the
# statement-level quota trigger on rag_chunks (tenant_data_quota_direct_insert, FOR EACH STATEMENT) ran once per
# chunk inside the indexing transaction; one multi-row INSERT per batch runs it once per batch. 200 rows x 12 bind
# parameters stay far below the 32,767 parameters of one statement.
CHUNK_INSERT_BATCH_ROWS = 200
# The value of each rag_chunks column, in insert order: the cast of its placeholder. The embedding is a real[] of
# the values ``pgvector.vector_values`` rounds exactly as the vector literal did.
_CHUNK_INSERT_COLUMNS = (
    ("tenant_id", "::uuid"), ("kb_id", "::uuid"), ("document_id", "::uuid"), ("index_id", "::uuid"),
    ("chunk_no", "::int"), ("content", ""), ("content_hash", ""), ("token_count", "::int"), ("source_page", "::int"),
    ("source_section", ""), ("metadata", "::jsonb"), ("embedding", "::real[]::vector"),
)


async def load_document(pool: asyncpg.Pool, tenant_id: str, kb_id: str, document_id: str) -> asyncpg.Record | None:
    """The tenant's knowledge-base document, or ``None`` (the caller answers 404)."""
    return await pool.fetchrow(
        """
        SELECT id::text, tenant_id::text, kb_id::text, type, name, status,
               source_info, file_path, content
        FROM kb_documents
        WHERE id = $1::uuid
          AND tenant_id = $2::uuid
          AND kb_id = $3::uuid
        """,
        document_id,
        tenant_id,
        kb_id,
    )


async def load_active_index(
    pool: asyncpg.Pool,
    row: asyncpg.Record,
    embedding_model: str,
    embedding_dimensions: int,
) -> asyncpg.Record | None:
    """The document's active learned index for this embedding model and size, or ``None``.

    ``index_running`` says whether another index of the document is still running (it would replace
    the active one). One statement, so the active row and that flag come from the same snapshot.
    """
    return await pool.fetchrow(
        """
        SELECT active.id::text AS index_id,
               active.content_sha256,
               active.chunk_count,
               EXISTS (
                 SELECT 1
                 FROM rag_document_indexes running
                 WHERE running.document_id = active.document_id
                   AND running.engine = 'self_built_rag'
                   AND running.status = 'running'
               ) AS index_running
        FROM rag_document_indexes active
        WHERE active.tenant_id = $1::uuid
          AND active.kb_id = $2::uuid
          AND active.document_id = $3::uuid
          AND active.engine = 'self_built_rag'
          AND active.status = 'learned'
          AND active.is_active = true
          AND active.embedding_model = $4
          AND active.embedding_dimensions = $5::int
        ORDER BY active.version DESC
        LIMIT 1
        """,
        row["tenant_id"],
        row["kb_id"],
        row["id"],
        embedding_model,
        embedding_dimensions,
    )


async def start_index_row(pool: asyncpg.Pool, row: asyncpg.Record, embedding_model: str) -> str:
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """
                UPDATE rag_document_indexes
                SET status = 'error',
                    error_reason = 'Phiên học tài liệu trước đó chưa hoàn tất và đã được thay bằng phiên mới.',
                    completed_at = now(),
                    updated_at = now()
                WHERE document_id = $1::uuid
                  AND engine = 'self_built_rag'
                  AND status = 'running'
                """,
            row["id"],
        )
        version = await conn.fetchval(
            """
                SELECT COALESCE(MAX(version), 0) + 1
                FROM rag_document_indexes
                WHERE document_id = $1::uuid
                  AND engine = 'self_built_rag'
                """,
            row["id"],
        )
        return cast(str, await conn.fetchval(
            """
                INSERT INTO rag_document_indexes (
                  tenant_id, kb_id, document_id, version, status, is_active,
                  embedding_model, embedding_dimensions, started_at
                )
                VALUES ($1::uuid, $2::uuid, $3::uuid, $4::int, 'running', false,
                        $5, 768, now())
                RETURNING id::text
                """,
            row["tenant_id"],
            row["kb_id"],
            row["id"],
            version,
            embedding_model,
        ))


async def mark_index_error(pool: asyncpg.Pool, index_id: str | None, reason: str) -> None:
    if not index_id:
        return
    await pool.execute(
        """
        UPDATE rag_document_indexes
        SET status = 'error',
            error_reason = $2,
            completed_at = now(),
            updated_at = now()
        WHERE id = $1::uuid
          AND status = 'running'
        """,
        index_id,
        reason[:1000],
    )


async def persist_structure_nodes_if_available(
    conn: asyncpg.Connection,
    row: asyncpg.Record,
    index_id: str,
    structure: dict[str, Any],
) -> bool:
    nodes = [node for node in structure.get("nodes", []) if isinstance(node, dict)]
    if not nodes:
        return False
    try:
        # The savepoint keeps the current indexing transaction usable when the
        # optional normalized table has not been deployed yet.
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO rag_document_structure_nodes (
                  tenant_id, kb_id, document_id, index_id, source_ref,
                  parent_source_ref, level, node_type, title, number_label,
                  page_start, logical_page, sort_order, confidence,
                  parser_version, metadata
                )
                VALUES ($1::uuid, $2::uuid, $3::uuid, $4::uuid, $5,
                        $6, $7::smallint, $8, $9, $10,
                        $11::int, $12::int, $13::int, $14::numeric,
                        $15, $16::jsonb)
                ON CONFLICT (index_id, source_ref) DO UPDATE
                SET parent_source_ref = EXCLUDED.parent_source_ref,
                    level = EXCLUDED.level,
                    node_type = EXCLUDED.node_type,
                    title = EXCLUDED.title,
                    number_label = EXCLUDED.number_label,
                    page_start = EXCLUDED.page_start,
                    logical_page = EXCLUDED.logical_page,
                    sort_order = EXCLUDED.sort_order,
                    confidence = EXCLUDED.confidence,
                    parser_version = EXCLUDED.parser_version,
                    metadata = EXCLUDED.metadata
                """,
                [
                    (
                        row["tenant_id"],
                        row["kb_id"],
                        row["id"],
                        index_id,
                        node.get("source_ref"),
                        node.get("parent_source_ref"),
                        node.get("level") or 1,
                        node.get("node_type") or "section",
                        node.get("title"),
                        node.get("number_label"),
                        node.get("page"),
                        node.get("logical_page"),
                        node.get("order") or 0,
                        node.get("confidence") or structure.get("confidence") or 0,
                        structure.get("parser_version") or "source-structure-v2",
                        json.dumps(
                            {
                                "structure_source": structure.get("structure_source"),
                                "parser_version": structure.get("parser_version"),
                                "warnings": structure.get("warnings", []),
                                "chapter_authority": structure.get("chapter_authority"),
                            },
                            ensure_ascii=False,
                        ),
                    )
                    for node in nodes
                ],
            )
        return True
    except asyncpg.exceptions.UndefinedTableError:
        log_event(logger, logging.INFO, "rag_structure_nodes_unavailable", fallback="chunk_metadata")
        return False


async def delete_previous_structure_nodes_if_available(
    conn: asyncpg.Connection,
    row: asyncpg.Record,
    current_index_id: str,
) -> None:
    """Remove normalized rows belonging to superseded document indexes."""
    try:
        await conn.execute(
            """
            DELETE FROM rag_document_structure_nodes
            WHERE tenant_id = $1::uuid
              AND kb_id = $2::uuid
              AND document_id = $3::uuid
              AND index_id <> $4::uuid
            """,
            row["tenant_id"],
            row["kb_id"],
            row["id"],
            current_index_id,
        )
    except asyncpg.exceptions.UndefinedTableError:
        log_event(logger, logging.INFO, "rag_structure_nodes_unavailable", fallback="skip_old_structure_cleanup")


async def lock_index_status(conn: SqlExecutor, index_id: Any, document_id: Any) -> Any:
    """Row-lock the index row inside the indexing transaction and return its status."""
    return await conn.fetchval(
        """
        SELECT status
        FROM rag_document_indexes
        WHERE id = $1::uuid
          AND document_id = $2::uuid
        FOR UPDATE
        """,
        index_id,
        document_id,
    )


@lru_cache(maxsize=8)
def chunk_insert_sql(rows: int) -> str:
    """One INSERT of ``rows`` chunk rows: ``len(_CHUNK_INSERT_COLUMNS)`` placeholders per row, rows in order."""
    width = len(_CHUNK_INSERT_COLUMNS)
    values = ", ".join(
        "(" + ", ".join(f"${row * width + column + 1}{cast_to}"
                        for column, (_name, cast_to) in enumerate(_CHUNK_INSERT_COLUMNS)) + ")"
        for row in range(rows)
    )
    columns = ", ".join(name for name, _cast in _CHUNK_INSERT_COLUMNS)
    # Built from the constant column list and numbered placeholders only; every value is a bind parameter.
    return f"INSERT INTO rag_chunks ({columns}) VALUES {values}"  # noqa: S608


async def insert_chunks(conn: SqlExecutor, rows: Sequence[Sequence[Any]]) -> None:
    """Insert every chunk row of one index, in order: one multi-row INSERT per ``CHUNK_INSERT_BATCH_ROWS`` rows."""
    for start in range(0, len(rows), CHUNK_INSERT_BATCH_ROWS):
        batch = rows[start:start + CHUNK_INSERT_BATCH_ROWS]
        await conn.execute(chunk_insert_sql(len(batch)), *[value for row in batch for value in row])


async def deactivate_other_indexes(conn: SqlExecutor, document_id: Any, index_id: Any) -> str:
    """Deactivate the document's other self-built indexes."""
    return await conn.execute(
        """
        UPDATE rag_document_indexes
        SET is_active = false
        WHERE document_id = $1::uuid
          AND engine = 'self_built_rag'
          AND id <> $2::uuid
        """,
        document_id,
        index_id,
    )


async def mark_index_learned(
    conn: SqlExecutor,
    index_id: Any,
    document_id: Any,
    content_sha256: Any,
    chunk_count: Any,
) -> str:
    """Mark the new index learned and active."""
    return await conn.execute(
        """
        UPDATE rag_document_indexes
        SET status = 'learned',
            is_active = true,
            content_sha256 = $3,
            chunk_count = $4::int,
            error_reason = NULL,
            completed_at = now(),
            updated_at = now()
        WHERE id = $1::uuid
          AND document_id = $2::uuid
        """,
        index_id,
        document_id,
        content_sha256,
        chunk_count,
    )


async def delete_inactive_indexes(conn: SqlExecutor, document_id: Any, index_id: Any) -> str:
    """Delete the document's inactive self-built indexes (chunks cascade)."""
    return await conn.execute(
        """
        DELETE FROM rag_document_indexes
        WHERE document_id = $1::uuid
          AND engine = 'self_built_rag'
          AND is_active = false
          AND id <> $2::uuid
        """,
        document_id,
        index_id,
    )
