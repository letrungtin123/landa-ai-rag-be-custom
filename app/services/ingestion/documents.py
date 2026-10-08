"""Deletion of indexed documents and knowledge bases."""

from __future__ import annotations

import asyncpg

from app.repositories import documents as documents_repository
from app.schemas.kb import RagDeleteDocumentRequest, RagDeleteKbRequest


async def delete_document(request: RagDeleteDocumentRequest, pool: asyncpg.Pool) -> dict[str, bool]:
    try:
        await documents_repository.delete_document_structure_nodes(
            pool,
            request.tenant_id,
            request.kb_id,
            request.document_id,
        )
    except asyncpg.exceptions.UndefinedTableError:
        pass
    await documents_repository.delete_document_indexes(pool, request.tenant_id, request.kb_id, request.document_id)
    return {"deleted": True}


async def delete_kb(request: RagDeleteKbRequest, pool: asyncpg.Pool) -> dict[str, bool]:
    try:
        await documents_repository.delete_kb_structure_nodes(pool, request.tenant_id, request.kb_id)
    except asyncpg.exceptions.UndefinedTableError:
        pass
    await documents_repository.delete_kb_indexes(pool, request.tenant_id, request.kb_id)
    return {"deleted": True}
