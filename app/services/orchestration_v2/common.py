"""Shared orchestration-v2 helpers: HTTP error shape and provider transport."""

from __future__ import annotations

import json
import logging

from fastapi import HTTPException
from pydantic import BaseModel, SecretStr

from app.core.logging import SERVICE_LOGGER_NAME
from app.schemas.common import AiUsage
from app.services import provider
from app.services.provider import provider_http_error_status, safe_provider_error_diagnostics

logger = logging.getLogger(SERVICE_LOGGER_NAME)


# Legacy V2 skeleton/shard providers fall back on these codes; AI_PROVIDER_RATE_LIMITED was
# reported as AI_PROVIDER_QUOTA_EXHAUSTED before the 429 classification and keeps that branch.
ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES = frozenset({
    "AI_PROVIDER_TIMEOUT", "AI_PROVIDER_UNAVAILABLE", "AI_PROVIDER_QUOTA_EXHAUSTED", "AI_PROVIDER_RATE_LIMITED",
})


def _orchestration_v2_http_error(code: str, message: str, *, status_code: int = 422) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


async def _orchestration_v2_generate_content(
    api_key: str | SecretStr,
    model: str,
    prompt: str,
    *,
    max_output_tokens: int,
    response_schema: type[BaseModel],
    correlation_id: str | None,
    generation_stage: str,
) -> tuple[str, AiUsage]:
    """Call Gemini with a safe, terminal classification for definitive refusals.

    A provider HTTP 400/401/403 proves that this request was rejected rather
    than accepted for generation. Expose only a stable internal code so the
    durable worker can release its reservation and fail immediately. Network,
    timeout, and 5xx ambiguity retain the existing outcome-unknown path.
    """

    try:
        return await provider.generate_content(
            api_key,
            model,
            prompt,
            max_output_tokens=max_output_tokens,
            json_mode=True,
            response_schema=response_schema,
        )
    except HTTPException:
        raise
    except Exception as error:
        status = provider_http_error_status(error)
        diagnostics = safe_provider_error_diagnostics(error)
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "provider_request_rejected" if status in {400, 401, 403} else "provider_request_failed",
            "correlation_id": correlation_id,
            "generation_stage": generation_stage,
            "model": model,
            **diagnostics,
        }, sort_keys=True))
        if status == 400:
            raise _orchestration_v2_http_error(
                "AI_PROVIDER_REQUEST_REJECTED",
                "AI provider rejected the structured generation request.",
                status_code=502,
            ) from error
        if status in {401, 403}:
            raise _orchestration_v2_http_error(
                "AI_PROVIDER_AUTH_REJECTED",
                "AI provider rejected the configured credentials or access policy.",
                status_code=502,
            ) from error
        raise
