"""Orchestration-v2 routes: source snapshot, course skeleton, chapter shard and unit (IDM or legacy)."""

from __future__ import annotations

from typing import Annotated

import asyncpg
from fastapi import Depends, FastAPI

from app.api.deps import get_db, require_internal_token
from app.schemas.orchestration_v2 import (
    ChapterShardV2Response,
    CourseSkeletonV2Response,
    RagLessonAuthorChapterShardV2Request,
    RagLessonAuthorCourseSkeletonV2Request,
    RagLessonAuthorSourceSnapshotV2Request,
    RagLessonAuthorUnitV2Request,
    SourceSnapshotV2Response,
    UnitV2Response,
)
from app.services.legacy_routes import (
    CHAPTER_SHARD_ROUTE,
    COURSE_SKELETON_ROUTE,
    UNIT_ROUTE,
    record_legacy_route_call,
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
) -> CourseSkeletonV2Response:
    if request.idm is None:
        record_legacy_route_call(COURSE_SKELETON_ROUTE)
    return CourseSkeletonV2Response.model_validate(
        await course_skeleton_service.lesson_author_orchestration_v2_course_skeleton(request),
    )


async def lesson_author_orchestration_v2_chapter_shard(
    request: RagLessonAuthorChapterShardV2Request,
) -> ChapterShardV2Response:
    if request.idm_module_context is None:
        record_legacy_route_call(CHAPTER_SHARD_ROUTE)
    return ChapterShardV2Response.model_validate(
        await chapter_shard_service.lesson_author_orchestration_v2_chapter_shard(request),
    )


async def lesson_author_orchestration_v2_unit(request: RagLessonAuthorUnitV2Request) -> UnitV2Response:
    if request.unit_contract.idm_unit_brief is None:
        record_legacy_route_call(UNIT_ROUTE)
    return UnitV2Response.model_validate(await unit_service.lesson_author_orchestration_v2_unit(request))



def register(app: FastAPI) -> None:
    """Declare these routes on ``app`` itself (see ``app.api.routes``)."""
    app.add_api_route(
        "/v1/lesson-author/orchestration-v2/source-snapshot",
        lesson_author_orchestration_v2_source_snapshot,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        COURSE_SKELETON_ROUTE,
        lesson_author_orchestration_v2_course_skeleton,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        CHAPTER_SHARD_ROUTE,
        lesson_author_orchestration_v2_chapter_shard,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
    app.add_api_route(
        UNIT_ROUTE,
        lesson_author_orchestration_v2_unit,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
