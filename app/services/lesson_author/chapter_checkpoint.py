"""Legacy chapter checkpoint: a proposal run mapped to checkpoint failures."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from time import perf_counter
from typing import Any

import asyncpg
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import SERVICE_LOGGER_NAME
from app.schemas.lesson_author import RagLessonAuthorCheckpointRequest
from app.services.deadlines import run_with_deadline
from app.services.lesson_author import proposal as proposal_service
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.workflows.contracts import WorkflowFailure

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def lesson_author_chapter_checkpoint(
    request: RagLessonAuthorCheckpointRequest,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    return await run_with_deadline(
        "/v1/lesson-author/chapter-checkpoint",
        settings.lesson_author_deadline_ms,
        _lesson_author_chapter_checkpoint(request, pool),
    )


async def _lesson_author_chapter_checkpoint(
    request: RagLessonAuthorCheckpointRequest,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        # Covers retrieval and final validation too. Cancellation of a to_thread
        # await is NOT proof that the synchronous Google HTTP request was stopped.
        return await asyncio.wait_for(proposal_service.lesson_author_proposal(request, pool), request.remaining_workflow_budget_ms / 1000)
    except asyncio.TimeoutError as error:
        failure = WorkflowFailure("PROVIDER_ERROR", "The chapter invocation deadline expired.",
                                  internal_code="AI_STAGED_LESSON_WORKFLOW_TIMEOUT", failure_stage="chapter_checkpoint_deadline")
        cause: Exception = error
    except WorkflowFailure as error:
        failure, cause = error, error
    except HTTPException as error:
        detail = error.detail if isinstance(error.detail, dict) else {}
        code = str(detail.get("code") or "SOURCE_SCOPE_INCOMPLETE")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", code):
            code = "SOURCE_SCOPE_INCOMPLETE"
        provider_error = code.startswith("AI_PROVIDER_") or code == "AI_STAGED_LESSON_WORKFLOW_TIMEOUT"
        failure = WorkflowFailure("PROVIDER_ERROR" if provider_error else "SOURCE_SCOPE_INCOMPLETE",
                                  "The chapter checkpoint request could not complete.", internal_code=code,
                                  failure_stage="chapter_checkpoint_provider" if provider_error else "chapter_checkpoint_source_validation")
        cause = error
    except (ValueError, LessonAuthorProposalValidationError) as error:
        failure = WorkflowFailure("LESSON_VALIDATION_FAILED", "The chapter checkpoint contract is invalid.",
                                  internal_code="CHAPTER_CHECKPOINT_CONTRACT_INVALID", failure_stage="chapter_checkpoint_contract")
        cause = error
    except Exception as error:
        failure = WorkflowFailure("LESSON_VALIDATION_FAILED", "The chapter checkpoint operation failed.",
                                  internal_code="CHAPTER_CHECKPOINT_INTERNAL_ERROR", failure_stage="chapter_checkpoint_internal")
        cause = error
    diagnostics = {"workflow": "lesson_generation", "event": "final_failure",
                   "correlation_id": request.correlation_id, "conversation_id": request.conversation_id,
                   "checkpoint_action": request.checkpoint_action, "unit_index": request.checkpoint_unit_index,
                   "failure_stage": failure.failure_stage, "internal_failure_code": failure.internal_code,
                   "external_failure_code": failure.code, "duration_ms": round((perf_counter() - started) * 1000)}
    if failure.internal_code in {"CHAPTER_UNIT_CONTRACT_REJECTED", "CHAPTER_COMPONENT_REPAIR_EXHAUSTED"}:
        diagnostics.update(failure.diagnostics)
    if failure.code == "PROVIDER_ERROR":
        provider_keys = {"provider_http_status", "provider_error_type", "provider_status", "provider_error_category",
                         "provider_message_class", "provider_error_detail_count",
                         "provider_schema_constraint", "provider_error_markers", "usage_source", "provider_input_tokens", "provider_output_tokens",
                         "provider_total_tokens", "provider_finish_reason", "provider_schema_fingerprint"}
        diagnostics.update({key: value for key, value in failure.diagnostics.items() if key in provider_keys})
    logger.info("lesson_author_checkpoint_diagnostic %s", json.dumps(diagnostics, sort_keys=True))
    raise HTTPException(status_code=502 if failure.code == "PROVIDER_ERROR" else 422, detail={
        "code": failure.code, "message": "The chapter could not complete its validated checkpoint operation.",
        "internal_failure_code": failure.internal_code, "failure_stage": failure.failure_stage,
        "correlation_id": request.correlation_id,
    }) from cause
