"""SQL of document and knowledge-base deletion (index rows and stored structure nodes)."""

from __future__ import annotations

from typing import Any

from app.repositories.executor import SqlExecutor


async def delete_document_structure_nodes(pool: SqlExecutor, tenant_id: Any, kb_id: Any, document_id: Any) -> str:
    """Delete a document's stored structure nodes."""
    return await pool.execute(
        """
        DELETE FROM rag_document_structure_nodes
        WHERE tenant_id = $1::uuid
          AND kb_id = $2::uuid
          AND document_id = $3::uuid
        """,
        tenant_id,
        kb_id,
        document_id,
    )


async def delete_document_indexes(pool: SqlExecutor, tenant_id: Any, kb_id: Any, document_id: Any) -> str:
    """Delete a document's index rows (chunks cascade)."""
    return await pool.execute(
        """
        DELETE FROM rag_document_indexes
        WHERE tenant_id = $1::uuid
          AND kb_id = $2::uuid
          AND document_id = $3::uuid
        """,
        tenant_id,
        kb_id,
        document_id,
    )


async def delete_kb_structure_nodes(pool: SqlExecutor, tenant_id: Any, kb_id: Any) -> str:
    """Delete a knowledge base's stored structure nodes."""
    return await pool.execute(
        """
        DELETE FROM rag_document_structure_nodes
        WHERE tenant_id = $1::uuid
          AND kb_id = $2::uuid
        """,
        tenant_id,
        kb_id,
    )


async def delete_kb_indexes(pool: SqlExecutor, tenant_id: Any, kb_id: Any) -> str:
    """Delete a knowledge base's index rows (chunks cascade)."""
    return await pool.execute(
        """
        DELETE FROM rag_document_indexes
        WHERE tenant_id = $1::uuid
          AND kb_id = $2::uuid
        """,
        tenant_id,
        kb_id,
    )
