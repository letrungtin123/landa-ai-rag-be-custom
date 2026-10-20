"""The error codes the service answers with outside a stage payload, and the HTTP status of each.

Node reads ``detail.code`` of a failed request (``{"detail": {"code", "message"}}``) to decide whether a
run retries, stops or falls back, so a code and its status are a wire contract: they change together
with the backend, never in one place only. ``AppError`` and ``ProviderError`` (``app.core.errors``)
are raised with these codes; ``tests/contract/test_error_codes.py`` checks that every literal ``AppError`` code
of the service and every provider code is registered here with the status it is raised with.

Stage-level codes inside a successful payload (``IDM_*``, ``ORCHESTRATION_V2_*`` findings) are not
listed here: they belong to their stage contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ErrorCode:
    code: str
    http_status: int
    # A later attempt of the same request can succeed (timeouts, an unavailable or busy service).
    retryable: bool


def _registry(*codes: ErrorCode) -> dict[str, ErrorCode]:
    return {item.code: item for item in codes}


# Provider failures (app.infra.gemini). Messages are the service's own, never the provider's.
AI_PROVIDER_TIMEOUT: Final = ErrorCode("AI_PROVIDER_TIMEOUT", 504, retryable=True)
AI_PROVIDER_UNAVAILABLE: Final = ErrorCode("AI_PROVIDER_UNAVAILABLE", 503, retryable=True)
# A per-minute limit that outlived the bounded wait (app.infra.provider_limits).
AI_PROVIDER_RATE_LIMITED: Final = ErrorCode("AI_PROVIDER_RATE_LIMITED", 503, retryable=True)
# The key cannot pay any more: no retry inside the run can succeed.
AI_PROVIDER_QUOTA_EXHAUSTED: Final = ErrorCode("AI_PROVIDER_QUOTA_EXHAUSTED", 503, retryable=False)
# A definitive provider refusal (HTTP 400 / 401 / 403) of a structured generation request.
AI_PROVIDER_REQUEST_REJECTED: Final = ErrorCode("AI_PROVIDER_REQUEST_REJECTED", 502, retryable=False)
AI_PROVIDER_AUTH_REJECTED: Final = ErrorCode("AI_PROVIDER_AUTH_REJECTED", 502, retryable=False)
# The staged lesson workflow had no time left for its next provider call.
AI_STAGED_LESSON_WORKFLOW_TIMEOUT: Final = ErrorCode("AI_STAGED_LESSON_WORKFLOW_TIMEOUT", 504, retryable=True)

PROVIDER_ERROR_CODES: Final = _registry(
    AI_PROVIDER_TIMEOUT,
    AI_PROVIDER_UNAVAILABLE,
    AI_PROVIDER_RATE_LIMITED,
    AI_PROVIDER_QUOTA_EXHAUSTED,
    AI_PROVIDER_REQUEST_REJECTED,
    AI_PROVIDER_AUTH_REJECTED,
    AI_STAGED_LESSON_WORKFLOW_TIMEOUT,
)

# AppError codes (rendered with the request's correlation id by app.core.errors.app_error_handler).
APP_ERROR_CODES: Final = _registry(
    ErrorCode("UNAUTHORIZED", 401, retryable=False),
    ErrorCode("SERVICE_BUSY", 503, retryable=True),
    ErrorCode("DATABASE_NOT_READY", 503, retryable=True),
    ErrorCode("REQUEST_DEADLINE_EXCEEDED", 504, retryable=True),
    ErrorCode("DOCUMENT_LIMIT_EXCEEDED", 422, retryable=False),
    ErrorCode("DOCUMENT_TYPE_UNSUPPORTED", 422, retryable=False),
    ErrorCode("SOURCE_DOWNLOAD_URL_REQUIRED", 422, retryable=False),
    ErrorCode("STORAGE_URL_REJECTED", 422, retryable=False),
    ErrorCode("STORAGE_REDIRECT_REJECTED", 422, retryable=False),
    ErrorCode("STORAGE_OBJECT_UNAVAILABLE", 422, retryable=False),
    ErrorCode("STORAGE_DOWNLOAD_FAILED", 502, retryable=True),
    # Organization context distillation (``POST /v1/lesson-author/org-context/distill``): none of the requested
    # documents has learned text yet, or the provider answered no valid profile after the repair (a new attempt may).
    ErrorCode("ORG_CONTEXT_NO_LEARNED_SOURCE", 409, retryable=False),
    ErrorCode("ORG_CONTEXT_DISTILL_OUTPUT_INVALID", 502, retryable=True),
)

# Request envelope codes of the framework handlers in app.core.errors.
ENVELOPE_ERROR_CODES: Final = _registry(
    ErrorCode("REQUEST_VALIDATION_FAILED", 422, retryable=False),
    ErrorCode("CHAPTER_CHECKPOINT_CONTRACT_INVALID", 422, retryable=False),
    ErrorCode("ORCHESTRATION_V2_CONTRACT_INVALID", 422, retryable=False),
    ErrorCode("INTERNAL_ERROR", 500, retryable=False),
)

ERROR_CODES: Final = {**PROVIDER_ERROR_CODES, **APP_ERROR_CODES, **ENVELOPE_ERROR_CODES}

__all__ = [
    "AI_PROVIDER_AUTH_REJECTED",
    "AI_PROVIDER_QUOTA_EXHAUSTED",
    "AI_PROVIDER_RATE_LIMITED",
    "AI_PROVIDER_REQUEST_REJECTED",
    "AI_PROVIDER_TIMEOUT",
    "AI_PROVIDER_UNAVAILABLE",
    "AI_STAGED_LESSON_WORKFLOW_TIMEOUT",
    "APP_ERROR_CODES",
    "ENVELOPE_ERROR_CODES",
    "ERROR_CODES",
    "PROVIDER_ERROR_CODES",
    "ErrorCode",
]
