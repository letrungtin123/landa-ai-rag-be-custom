"""Knowledge-base routes: index a document, delete a document, delete a knowledge base."""

from __future__ import annotations

from typing import Annotated, Any

import asyncpg
from fastapi import APIRouter, Depends

from app.api.deps import get_db, require_internal_token
from app.schemas.kb import RagDeleteDocumentRequest, RagDeleteKbRequest, RagIndexRequest
from app.services.ingestion import documents as documents_service
from app.services.ingestion import index as index_service

router = APIRouter()


@router.post("/v1/kb/documents/index", dependencies=[Depends(require_internal_token)])
async def index_document(request: RagIndexRequest, pool: Annotated[asyncpg.Pool, Depends(get_db)]) -> dict[str, Any]:
    return await index_service.index_document(request, pool)


@router.post("/v1/kb/documents/delete", dependencies=[Depends(require_internal_token)])
async def delete_document(
    request: RagDeleteDocumentRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> dict[str, bool]:
    return await documents_service.delete_document(request, pool)


@router.post("/v1/kb/delete", dependencies=[Depends(require_internal_token)])
async def delete_kb(request: RagDeleteKbRequest, pool: Annotated[asyncpg.Pool, Depends(get_db)]) -> dict[str, bool]:
    return await documents_service.delete_kb(request, pool)
