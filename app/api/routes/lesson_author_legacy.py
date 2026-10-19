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
from app.services.legacy_routes import (
    BLUEPRINT_ROUTE,
    CHAPTER_CHECKPOINT_ROUTE,
    PROPOSAL_ROUTE,
    record_legacy_route_call,
)
from app.services.lesson_author import blueprint_workflow as blueprint_workflow_service
from app.services.lesson_author import chapter_checkpoint as chapter_checkpoint_service
from app.services.lesson_author import proposal as proposal_service


async def lesson_author_chapter_checkpoint(
    request: RagLessonAuthorCheckpointRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> dict[str, Any]:
    record_legacy_route_call(CHAPTER_CHECKPOINT_ROUTE)
    return await chapter_checkpoint_service.lesson_author_chapter_checkpoint(request, pool)


async def lesson_author_proposal(
    request: RagLessonAuthorRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> dict[str, Any]:
    record_legacy_route_call(PROPOSAL_ROUTE)
    return await proposal_service.lesson_author_proposal(request, pool)


async def lesson_author_blueprint(
    request: RagLessonAuthorBlueprintRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> dict[str, Any]:
    record_legacy_route_call(BLUEPRINT_ROUTE)
    return await blueprint_workflow_service.lesson_author_blueprint(request, pool)


def register(app: FastAPI) -> None:
    """Declare these routes on ``app`` itself (see ``app.api.routes``)."""
    app.add_api_route(
        CHAPTER_CHECKPOINT_ROUTE,
        lesson_author_chapter_checkpoint,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        PROPOSAL_ROUTE,
        lesson_author_proposal,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        BLUEPRINT_ROUTE,
        lesson_author_blueprint,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
