"""SQL of document indexing: the source document, index rows and stored structure nodes."""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg
from fastapi import HTTPException

from app.core.logging import SERVICE_LOGGER_NAME

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def load_document(pool: asyncpg.Pool, tenant_id: str, kb_id: str, document_id: str) -> asyncpg.Record:
    row = await pool.fetchrow(
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
    if row is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu Knowledge Base.")
    return row


async def start_index_row(pool: asyncpg.Pool, row: asyncpg.Record, embedding_model: str) -> str:
    async with pool.acquire() as conn:
        async with conn.transaction():
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
            return await conn.fetchval(
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
            )


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
        logger.info("rag_document_structure_nodes is not deployed; using chunk metadata fallback")
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
        logger.info("rag_document_structure_nodes is not deployed; skipped old structure cleanup")
