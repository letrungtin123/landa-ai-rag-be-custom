"""Orchestration-v2 course skeleton: IDM course design or the legacy skeleton."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from app.core.config import settings
from app.core.logging import SERVICE_LOGGER_NAME
from app.idm.course_design import run_idm_course_design
from app.idm.runtime import IdmError, IdmStageError
from app.lesson_author_orchestration_v2 import OrchestrationContractError
from app.lesson_author_orchestration_v2_provider import (
    COURSE_SKELETON_PROVIDER_SCHEMA_METADATA,
    CourseSkeletonDraftV2,
    CourseSkeletonProviderWireV2,
    bind_course_skeleton_v2,
    fallback_course_skeleton_draft_v2,
    skeleton_prompt_v2,
)
from app.schemas.common import AiUsage
from app.schemas.orchestration_v2 import RagLessonAuthorCourseSkeletonV2Request
from app.services.deadlines import record_fallback
from app.services.orchestration_v2.common import (
    ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES,
    _orchestration_v2_generate_content,
)
from app.services.orchestration_v2.idm import _idm_http_error, _idm_runtime
from app.services.provider import combine_usage

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def lesson_author_orchestration_v2_course_skeleton(
    request: RagLessonAuthorCourseSkeletonV2Request,
) -> dict[str, Any]:
    if request.idm is not None:
        try:
            result = await run_idm_course_design(
                request.idm, source_snapshot_hash=request.source_snapshot_hash,
                runtime=_idm_runtime(request, budget_ms=request.idm.remaining_budget_ms,
                                     allowance=request.idm.token_allowance),
                parallelism=settings.idm_w1_parallelism, max_sections=settings.idm_max_sections,
            )
        except IdmError as error:
            raise _idm_http_error(error) from error
        except ValidationError as error:
            raise _idm_http_error(IdmStageError("IDM_STAGE_OUTPUT_INVALID")) from error
    else:
        result = await _lesson_author_orchestration_v2_course_skeleton(request=request)
    record_fallback("course_skeleton", result)
    return result


async def _lesson_author_orchestration_v2_course_skeleton(
    request: RagLessonAuthorCourseSkeletonV2Request,
) -> dict[str, Any]:
    """Generate only global structure; chapter content is delegated to bounded shards."""

    prompt = skeleton_prompt_v2(request.locale, request.scope_catalog, request.source_authority)
    logger.info("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "provider_schema_projection_ready", "correlation_id": request.correlation_id,
        "generation_stage": "course_skeleton", **COURSE_SKELETON_PROVIDER_SCHEMA_METADATA,
    }, sort_keys=True))
    usage = AiUsage()
    last_code = "ARCHITECTURE_SKELETON_INVALID"
    provider_failure_code: str | None = None
    for attempt in range(1, request.max_attempts + 1):
        try:
            text, attempt_usage = await _orchestration_v2_generate_content(
                request.api_key, request.model, prompt,
                max_output_tokens=request.max_output_tokens,
                response_schema=CourseSkeletonProviderWireV2,
                correlation_id=request.correlation_id,
                generation_stage="course_skeleton",
            )
        except HTTPException as error:
            detail = error.detail if isinstance(error.detail, dict) else {}
            code = detail.get("code")
            if code not in ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES:
                raise
            provider_failure_code = str(code)
            last_code = provider_failure_code
            break
        except Exception as error:
            # The Node worker persists its dispatch fence before this HTTP
            # request. Unknown SDK/transport failures must therefore settle
            # conservatively instead of turning a valid source fallback into a
            # blank 5xx response.
            provider_failure_code = "AI_PROVIDER_UNAVAILABLE"
            last_code = provider_failure_code
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_provider_fallback",
                "correlation_id": request.correlation_id,
                "failure_code": provider_failure_code,
                "exception_type": type(error).__name__,
            }, sort_keys=True))
            break
        usage = combine_usage(usage, attempt_usage)
        try:
            draft = CourseSkeletonDraftV2.model_validate_json(text)
            skeleton = bind_course_skeleton_v2(
                draft,
                source_snapshot_hash=request.source_snapshot_hash,
                locale=request.locale,
                scope_catalog=request.scope_catalog,
                source_authority=request.source_authority,
            )
            logger.info("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_ready", "correlation_id": request.correlation_id,
                "source_snapshot_hash": request.source_snapshot_hash,
                "scope_count": len(request.scope_catalog), "chapter_count": len(skeleton.chapters),
                "provider_attempt": attempt,
            }, sort_keys=True))
            return {"contract_version": 2, "skeleton": skeleton.model_dump(mode="json"),
                    "usage": usage.model_dump(), "usage_complete": True,
                    "usage_source": "provider", "content_origin": "provider_validated",
                    "quality_state": "validated"}
        except OrchestrationContractError as error:
            last_code = error.code
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_validation_failed",
                "correlation_id": request.correlation_id,
                "provider_attempt": attempt,
                "failure_code": last_code,
            }, sort_keys=True))
        except ValidationError as error:
            last_code = "ARCHITECTURE_SKELETON_INVALID"
            safe_errors = [{"type": item.get("type"), "loc": list(item.get("loc") or ())}
                           for item in error.errors(include_url=False, include_input=False)[:10]]
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_validation_failed",
                "correlation_id": request.correlation_id,
                "provider_attempt": attempt,
                "failure_code": last_code,
                "validation_errors": safe_errors,
            }, sort_keys=True))
    fallback = fallback_course_skeleton_draft_v2(
        request.locale, request.scope_catalog, request.source_authority,
    )
    skeleton = bind_course_skeleton_v2(
        fallback,
        source_snapshot_hash=request.source_snapshot_hash,
        locale=request.locale,
        scope_catalog=request.scope_catalog,
        source_authority=request.source_authority,
    )
    logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "course_skeleton_deterministic_fallback_ready",
        "correlation_id": request.correlation_id,
        "source_snapshot_hash": request.source_snapshot_hash,
        "scope_count": len(request.scope_catalog),
        "chapter_count": len(skeleton.chapters),
        "provider_attempts": request.max_attempts,
        "last_failure_code": last_code,
    }, sort_keys=True))
    return {"contract_version": 2, "skeleton": skeleton.model_dump(mode="json"),
            "usage": usage.model_dump(), "usage_complete": provider_failure_code is None,
            "usage_source": "reserved_upper_bound" if provider_failure_code else "provider",
            "content_origin": "structured_fallback", "quality_state": "review_required"}
