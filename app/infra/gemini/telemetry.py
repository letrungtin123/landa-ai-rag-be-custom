"""Safe, response-derived provider telemetry: sizes, counters and statuses, never prompts or output."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.core.logging import SERVICE_LOGGER_NAME, log_event
from app.infra.gemini.usage import estimate_tokens

logger = logging.getLogger(SERVICE_LOGGER_NAME)


def normalize_provider_finish_reason(response: Any) -> str | None:
    """Read the provider's completion status without interpreting its content."""

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return None
    value = str(finish_reason).strip()
    return value[:96] if value else None


def provider_response_telemetry(
    response: Any,
    *,
    model: str,
    max_output_tokens: int,
    prompt: str,
    response_text: str,
    duration_ms: int,
) -> dict[str, Any]:
    """Return safe, response-derived telemetry without logging model output."""

    meta = getattr(response, "usage_metadata", None)
    provider_input = getattr(meta, "prompt_token_count", None)
    provider_output = getattr(meta, "candidates_token_count", None)
    provider_total = getattr(meta, "total_token_count", None)
    has_provider_usage = any(value is not None for value in (provider_input, provider_output, provider_total))
    return {
        "provider": "google_ai_studio",
        "model": model,
        "configured_max_output_tokens": max_output_tokens,
        "provider_http_status": 200,
        "provider_finish_reason": normalize_provider_finish_reason(response),
        "provider_finish_reason_available": normalize_provider_finish_reason(response) is not None,
        "usage_source": "provider" if has_provider_usage else "local_estimate",
        # Never label estimates as Gemini/provider usage.
        "provider_input_tokens": int(provider_input) if provider_input is not None else None,
        "provider_output_tokens": int(provider_output) if provider_output is not None else None,
        "provider_total_tokens": int(provider_total) if provider_total is not None else None,
        "local_estimated_input_tokens": None if has_provider_usage else estimate_tokens(prompt),
        "local_estimated_output_tokens": None if has_provider_usage else estimate_tokens(response_text),
        "response_chars": len(response_text),
        "response_bytes": len(response_text.encode("utf-8")),
        "duration_ms": duration_ms,
    }


def emit_safe_provider_telemetry(
    callback: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    """Diagnostics are best effort and must never change generation behavior."""

    if callback is None:
        return
    try:
        callback(payload)
    except Exception as error:  # pragma: no cover - defensive logging boundary
        log_event(logger, logging.WARNING, "lesson_author_provider_telemetry_emit_failed",
                  error_type=type(error).__name__)
