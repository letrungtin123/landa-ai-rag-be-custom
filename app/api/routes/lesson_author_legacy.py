"""Legacy lesson-author routes (proposal, chapter checkpoint, blueprint); live behind backend flags until PRD-3."""

from __future__ import annotations

from typing import Annotated, Any

import asyncpg
from fastapi import Depends, FastAPI

from app.api.deps import get_db, require_internal_token
from app.schemas.lesson_author import (
    RagLessonAuthorBlueprintRequest,
    RagLessonAuthorCheckpointRequest,
    RagLessonAuthorRequest,
)
from app.services.lesson_author import blueprint_workflow as blueprint_workflow_service
from app.services.lesson_author import chapter_checkpoint as chapter_checkpoint_service
from app.services.lesson_author import proposal as proposal_service


async def lesson_author_chapter_checkpoint(
    request: RagLessonAuthorCheckpointRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> dict[str, Any]:
    return await chapter_checkpoint_service.lesson_author_chapter_checkpoint(request, pool)


async def lesson_author_proposal(
    request: RagLessonAuthorRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> dict[str, Any]:
    return await proposal_service.lesson_author_proposal(request, pool)


async def lesson_author_blueprint(
    request: RagLessonAuthorBlueprintRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> dict[str, Any]:
    return await blueprint_workflow_service.lesson_author_blueprint(request, pool)


def register(app: FastAPI) -> None:
    """Declare these routes on ``app`` itself (see ``app.api.routes``)."""
    app.add_api_route(
        "/v1/lesson-author/chapter-checkpoint",
        lesson_author_chapter_checkpoint,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        "/v1/lesson-author/proposal",
        lesson_author_proposal,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        "/v1/lesson-author/blueprint",
        lesson_author_blueprint,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
