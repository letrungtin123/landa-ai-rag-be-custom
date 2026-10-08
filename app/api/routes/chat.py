"""Grounded chat route."""

from __future__ import annotations

from typing import Annotated, Any

import asyncpg
from fastapi import APIRouter, Depends

from app.api.deps import get_db, require_internal_token
from app.schemas.chat import RagChatRequest
from app.services.chat import service as chat_service

router = APIRouter()


@router.post("/v1/chat", dependencies=[Depends(require_internal_token)])
async def chat(request: RagChatRequest, pool: Annotated[asyncpg.Pool, Depends(get_db)]) -> dict[str, Any]:
    return await chat_service.chat(request, pool)
