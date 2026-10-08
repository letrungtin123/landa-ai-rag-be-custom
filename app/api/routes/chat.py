"""Grounded chat route."""

from __future__ import annotations

from typing import Annotated, Any

import asyncpg
from fastapi import Depends, FastAPI

from app.api.deps import get_db, require_internal_token
from app.schemas.chat import RagChatRequest
from app.services.chat import service as chat_service


async def chat(request: RagChatRequest, pool: Annotated[asyncpg.Pool, Depends(get_db)]) -> dict[str, Any]:
    return await chat_service.chat(request, pool)


def register(app: FastAPI) -> None:
    """Declare these routes on ``app`` itself (see ``app.api.routes``)."""
    app.add_api_route("/v1/chat", chat, methods=["POST"], dependencies=[Depends(require_internal_token)])
