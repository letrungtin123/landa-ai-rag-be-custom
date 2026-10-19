"""Orchestration-v2 routes: source snapshot, course skeleton, chapter shard and unit (IDM or legacy)."""

from __future__ import annotations

from typing import Annotated, Any

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
from app.services.response_contracts import observe_response_contract


async def lesson_author_orchestration_v2_source_snapshot(
    request: RagLessonAuthorSourceSnapshotV2Request,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> SourceSnapshotV2Response:
    return SourceSnapshotV2Response.model_validate(
        await source_snapshot_service.lesson_author_orchestration_v2_source_snapshot(request, pool),
    )


# The generation routes answer the service dict exactly as before the typed contracts (FastAPI's dict[str, Any]
# encoder). Their published models check the same dict for observation only: the check never raises and never
# changes the answer. It runs inline (about 0.2 ms for a 40 KB course design) rather than as a background task, which
# the disconnect-cancellation middleware would cancel once the body is sent.
async def lesson_author_orchestration_v2_course_skeleton(
    request: RagLessonAuthorCourseSkeletonV2Request,
) -> dict[str, Any]:
    if request.idm is None:
        record_legacy_route_call(COURSE_SKELETON_ROUTE)
    result = await course_skeleton_service.lesson_author_orchestration_v2_course_skeleton(request)
    observe_response_contract(COURSE_SKELETON_ROUTE, CourseSkeletonV2Response, result)
    return result


async def lesson_author_orchestration_v2_chapter_shard(
    request: RagLessonAuthorChapterShardV2Request,
) -> dict[str, Any]:
    if request.idm_module_context is None:
        record_legacy_route_call(CHAPTER_SHARD_ROUTE)
    result = await chapter_shard_service.lesson_author_orchestration_v2_chapter_shard(request)
    observe_response_contract(CHAPTER_SHARD_ROUTE, ChapterShardV2Response, result)
    return result


async def lesson_author_orchestration_v2_unit(
    request: RagLessonAuthorUnitV2Request,
) -> dict[str, Any]:
    if request.unit_contract.idm_unit_brief is None:
        record_legacy_route_call(UNIT_ROUTE)
    result = await unit_service.lesson_author_orchestration_v2_unit(request)
    observe_response_contract(UNIT_ROUTE, UnitV2Response, result)
    return result


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
