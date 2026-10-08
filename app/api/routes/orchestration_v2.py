"""Orchestration-v2 routes: source snapshot, course skeleton, chapter shard and unit (IDM or legacy)."""

from __future__ import annotations

from typing import Annotated, Any

import asyncpg
from fastapi import Depends, FastAPI

from app.api.deps import get_db, require_internal_token
from app.schemas.orchestration_v2 import (
    RagLessonAuthorChapterShardV2Request,
    RagLessonAuthorCourseSkeletonV2Request,
    RagLessonAuthorSourceSnapshotV2Request,
    RagLessonAuthorUnitV2Request,
    SourceSnapshotV2Response,
)
from app.services.orchestration_v2 import chapter_shard as chapter_shard_service
from app.services.orchestration_v2 import course_skeleton as course_skeleton_service
from app.services.orchestration_v2 import source_snapshot as source_snapshot_service
from app.services.orchestration_v2 import unit as unit_service


async def lesson_author_orchestration_v2_source_snapshot(
    request: RagLessonAuthorSourceSnapshotV2Request,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> SourceSnapshotV2Response:
    return SourceSnapshotV2Response.model_validate(
        await source_snapshot_service.lesson_author_orchestration_v2_source_snapshot(request, pool),
    )


async def lesson_author_orchestration_v2_course_skeleton(
    request: RagLessonAuthorCourseSkeletonV2Request,
) -> dict[str, Any]:
    return await course_skeleton_service.lesson_author_orchestration_v2_course_skeleton(request)


async def lesson_author_orchestration_v2_chapter_shard(request: RagLessonAuthorChapterShardV2Request) -> dict[str, Any]:
    return await chapter_shard_service.lesson_author_orchestration_v2_chapter_shard(request)


async def lesson_author_orchestration_v2_unit(request: RagLessonAuthorUnitV2Request) -> dict[str, Any]:
    return await unit_service.lesson_author_orchestration_v2_unit(request)


def register(app: FastAPI) -> None:
    """Declare these routes on ``app`` itself (see ``app.api.routes``)."""
    app.add_api_route(
        "/v1/lesson-author/orchestration-v2/source-snapshot",
        lesson_author_orchestration_v2_source_snapshot,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        "/v1/lesson-author/orchestration-v2/course-skeleton",
        lesson_author_orchestration_v2_course_skeleton,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        "/v1/lesson-author/orchestration-v2/chapter-shard",
        lesson_author_orchestration_v2_chapter_shard,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        "/v1/lesson-author/orchestration-v2/unit",
        lesson_author_orchestration_v2_unit,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
