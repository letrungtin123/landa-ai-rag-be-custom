"""Deletion of indexed documents and knowledge bases."""

from __future__ import annotations

import asyncpg

from app.schemas.kb import RagDeleteDocumentRequest, RagDeleteKbRequest


async def delete_document(request: RagDeleteDocumentRequest, pool: asyncpg.Pool) -> dict[str, bool]:
    try:
        await pool.execute(
            """
            DELETE FROM rag_document_structure_nodes
            WHERE tenant_id = $1::uuid
              AND kb_id = $2::uuid
              AND document_id = $3::uuid
            """,
            request.tenant_id,
            request.kb_id,
            request.document_id,
        )
    except asyncpg.exceptions.UndefinedTableError:
        pass
    await pool.execute(
        """
        DELETE FROM rag_document_indexes
        WHERE tenant_id = $1::uuid
          AND kb_id = $2::uuid
          AND document_id = $3::uuid
        """,
        request.tenant_id,
        request.kb_id,
        request.document_id,
    )
    return {"deleted": True}


async def delete_kb(request: RagDeleteKbRequest, pool: asyncpg.Pool) -> dict[str, bool]:
    try:
        await pool.execute(
            """
            DELETE FROM rag_document_structure_nodes
            WHERE tenant_id = $1::uuid
              AND kb_id = $2::uuid
            """,
            request.tenant_id,
            request.kb_id,
        )
    except asyncpg.exceptions.UndefinedTableError:
        pass
    await pool.execute(
        """
        DELETE FROM rag_document_indexes
        WHERE tenant_id = $1::uuid
          AND kb_id = $2::uuid
        """,
        request.tenant_id,
        request.kb_id,
    )
    return {"deleted": True}
