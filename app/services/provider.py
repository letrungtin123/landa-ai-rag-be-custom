"""Gemini provider gateway: generation, embeddings, bounded retries, usage and safe telemetry.

Callers use ``provider.generate_content`` / ``provider.embed_texts`` through the module so a
single patch target (``app.services.provider.generate_content``) fakes the provider everywhere.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from time import perf_counter
from typing import Any, Callable, Literal

from fastapi import HTTPException
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, SecretStr

from app.core import metrics
from app.core.config import settings
from app.core.errors import AppError
from app.core.logging import SERVICE_LOGGER_NAME
from app.infra import gemini as gemini_infra
from app.infra.provider_limits import RATE_LIMITED_CODE, classify_provider_limit
from app.lesson_author_blueprint import describe_lesson_author_blueprint_response
from app.schemas.common import AiUsage
from app.services import runtime as service_runtime

logger = logging.getLogger(SERVICE_LOGGER_NAME)


DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
PROVIDER_TRANSIENT_MAX_ATTEMPTS = settings.provider_max_attempts
PROVIDER_TRANSIENT_RETRY_DELAY_SECONDS = settings.provider_retry_base_ms / 1000
PROVIDER_RATE_LIMIT_MIN_WAIT_SECONDS = 1.0
PROVIDER_RETRY_HINT_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$")


LEGACY_EMBEDDING_MODEL_ALIASES = {
    "text-embedding-004": DEFAULT_EMBEDDING_MODEL,
}


def require_provider_api_key(api_key: str | SecretStr) -> str:
    value = api_key.get_secret_value().strip() if isinstance(api_key, SecretStr) else api_key.strip()
    if not value:
        raise ValueError("Google AI Studio API key chưa được cấu hình.")
    return value


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def normalize_usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
    total_tokens: int | None = None,
) -> AiUsage:
    total = total_tokens if total_tokens is not None else input_tokens + output_tokens + embedding_tokens
    return AiUsage(
        inputTokens=max(0, int(input_tokens)),
        outputTokens=max(0, int(output_tokens)),
        embeddingTokens=max(0, int(embedding_tokens)),
        totalTokens=max(0, int(total)),
    )


def combine_usage(*items: AiUsage) -> AiUsage:
    return normalize_usage(
        input_tokens=sum(item.inputTokens for item in items),
        output_tokens=sum(item.outputTokens for item in items),
        embedding_tokens=sum(item.embeddingTokens for item in items),
        total_tokens=sum(item.totalTokens for item in items),
    )


def usage_from_google_response(response: Any, fallback_prompt: str, fallback_output: str) -> AiUsage:
    meta = getattr(response, "usage_metadata", None)
    input_tokens = getattr(meta, "prompt_token_count", None)
    output_tokens = getattr(meta, "candidates_token_count", None)
    total_tokens = getattr(meta, "total_token_count", None)
    return normalize_usage(
        input_tokens=input_tokens if input_tokens is not None else estimate_tokens(fallback_prompt),
        output_tokens=output_tokens if output_tokens is not None else estimate_tokens(fallback_output),
        total_tokens=total_tokens,
    )


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
        logger.warning("lesson_author_provider_telemetry_emit_failed error_type=%s", type(error).__name__)


def get_embedding_values(item: Any) -> list[float]:
    values = getattr(item, "values", None)
    if values is None and isinstance(item, dict):
        values = item.get("values")
    if values is None:
        raise ValueError("Google AI Studio không trả embedding hợp lệ.")
    return [float(value) for value in values]


def normalize_embedding_model(model: str) -> str:
    return LEGACY_EMBEDDING_MODEL_ALIASES.get(model.strip(), model.strip())


def embedding_batch_size(model: str) -> int:
    if model == "gemini-embedding-2":
        return 1
    return max(1, min(settings.embedding_batch_size, 100))


def provider_http_error_status(error: Exception) -> int | None:
    """SDK 1.0 APIError uses .code, not the HTTP wrapper's .status_code."""
    status = error.code if isinstance(error, genai_errors.APIError) else getattr(error, "status_code", None)
    return status if type(status) is int and 400 <= status <= 599 else None


def safe_provider_error_diagnostics(error: Exception) -> dict[str, Any]:
    """Classify locally; never emit exception messages, bodies, URLs or headers."""
    status = None if isinstance(error, HTTPException) else provider_http_error_status(error)
    details = getattr(error, "details", None)
    envelope = details if isinstance(details, dict) else {}
    error_body = envelope.get("error", envelope)
    error_body = error_body if isinstance(error_body, dict) else {}
    message_value = error_body.get("message")
    structured_details = error_body.get("details", [])
    generic_message = isinstance(message_value, str) and message_value.strip().casefold() in {
        "request contains an invalid argument.", "request contains an invalid argument", "invalid argument."
    }
    message = str(error).casefold()
    # Gemini may say "JSON schema"/"controlled generation" without naming
    # response_schema. Match diagnostic categories, never log the message.
    schema_error = status == 400 and any(key in message for key in ("schema", "controlled generation", "constrained decoding"))
    compact_message = re.sub(r"[\s_]", "", message)
    markers = [code for code, needles in (
        ("MAX_ITEMS", ("maxitems", "maximumitems")), ("MIN_ITEMS", ("minitems", "minimumitems")),
        ("COMPLEXITY", ("toocomplex", "toomanystates", "nesting", "complexity")),
        ("UNSUPPORTED", ("unsupported", "notsupported", "unknownname")),
        ("POSITIVE_BOUND", ("greaterthan0", "greaterthanzero", "positiveinteger", "mustbepositive")),
        ("NULLABLE", ("nullable",)), ("ANY_OF", ("anyof",)), ("ONE_OF", ("oneof",)),
        ("MIN_LENGTH", ("minlength",)), ("MAX_LENGTH", ("maxlength",)),
        ("TOKEN_LIMIT", ("tokenlimit", "maxtokens", "maxoutputtokens")),
        ("THINKING_CONFIG", ("thinkingconfig", "includethoughts", "thinkingbudget")),
        ("ENUM", ("enum",)), ("PROPERTY_ORDERING", ("propertyordering",)),
        ("MODEL_UNSUPPORTED", ("modeldoesnotsupport", "modelisnotsupported")),
    ) if any(needle in compact_message for needle in needles)]
    constraint = "UNAVAILABLE"
    if schema_error:
        for code, constraint_markers in (
            ("MAX_ITEMS", ("max_items", "maxitems", "max items")),
            ("MIN_ITEMS", ("min_items", "minitems", "min items")),
            ("ARRAY_ITEMS", ("items",)),
            ("SCHEMA_COMPLEXITY", ("too complex", "too many states", "nesting")),
            ("UNSUPPORTED_FIELD", ("unknown name", "unsupported", "not supported")),
        ):
            if any(marker in message for marker in constraint_markers):
                constraint = code
                break
    provider_status = getattr(error, "status", None)
    known_statuses = {"INVALID_ARGUMENT", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED",
                      "PERMISSION_DENIED", "UNAUTHENTICATED", "NOT_FOUND", "FAILED_PRECONDITION"}
    # Type names are operational metadata, but never trust a dynamically created
    # exception type to be free of customer-controlled content.
    error_type = type(error).__name__
    known_types = {"ClientError", "ServerError", "APIError", "TimeoutError", "Timeout", "ReadTimeout", "ConnectTimeout",
                   "ConnectionError", "ConnectError", "RemoteProtocolError", "SSLError", "TypeError", "ValueError",
                   "ValidationError", "RuntimeError", "HTTPException", "AttributeError", "KeyError"}
    return {
        "provider_http_status": status,
        "provider_error_type": error_type if error_type in known_types else "OtherException",
        "provider_status": provider_status if isinstance(provider_status, str) and provider_status in known_statuses else "unavailable",
        "provider_error_category": "RESPONSE_SCHEMA_INVALID" if schema_error else "HTTP_ERROR" if status else "SDK_OR_TRANSPORT_ERROR",
        "provider_schema_constraint": constraint,
        "provider_error_markers": markers,
        "provider_message_class": "GENERIC_INVALID_ARGUMENT" if generic_message else "REDACTED_OTHER",
        "provider_error_detail_count": len(structured_details) if isinstance(structured_details, list) else 0,
        "usage_source": "unavailable",
    }


def provider_retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff with +/-20% jitter, capped by the configured maximum."""
    base = PROVIDER_TRANSIENT_RETRY_DELAY_SECONDS * (2 ** max(0, attempt))
    jittered = base * random.uniform(0.8, 1.2)  # noqa: S311 - jitter, not security
    delay: float = max(0.0, min(jittered, settings.provider_retry_max_ms / 1000))
    return delay


def provider_retry_hint_seconds(error: Exception) -> float | None:
    """Read a server retry hint (google.rpc.RetryInfo or Retry-After) without logging content."""
    details = getattr(error, "details", None)
    envelope = details if isinstance(details, dict) else {}
    body = envelope.get("error", envelope)
    entries = body.get("details", []) if isinstance(body, dict) else []
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and str(entry.get("@type", "")).endswith("google.rpc.RetryInfo"):
            match = PROVIDER_RETRY_HINT_PATTERN.match(str(entry.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except Exception:
            value = None
        if isinstance(value, str):
            match = PROVIDER_RETRY_HINT_PATTERN.match(value)
            if match:
                return float(match.group(1))
    return None


async def call_provider_with_timeout(
    run: Any,
    model: str,
    *,
    request_timeout_ms: int | None = None,
    on_provider_diagnostic: Callable[[dict[str, Any]], None] | None = None,
    operation: Literal["generate", "embed"] = "generate",
    rate_limit_max_wait_ms: int | None = None,
) -> Any:
    """Bound provider calls and retry only transient 5xx responses once.

    ``asyncio.wait_for(asyncio.to_thread(...))`` bounds this coroutine, but a
    timeout cannot prove that a synchronous Google client request already
    running in the worker thread was cancelled at HTTP level.  The provider
    client's ``HttpOptions(timeout=...)`` remains the request-level bound; do
    not treat cancellation of this await as hard provider cancellation.

    A 429 is classified by ``classify_provider_limit``: an exhausted key raises
    ``AI_PROVIDER_QUOTA_EXHAUSTED`` at once, a rate limit raises
    ``AI_PROVIDER_RATE_LIMITED`` (both 503). By default a rate limit is retried
    once when the server hint is short (legacy behaviour). A caller that passes
    ``rate_limit_max_wait_ms`` (IDM) waits the server hint (or a backoff) as
    often as the cumulative wait stays within that bound; those waits do not use
    up the transient-retry attempts.
    """
    timeout_ms = max(1, request_timeout_ms or settings.provider_request_timeout_ms)
    max_attempts = max(1, PROVIDER_TRANSIENT_MAX_ATTEMPTS)
    rate_limit_waited_seconds = 0.0
    rate_limit_retries = 0
    attempt = -1
    while attempt + 1 < max_attempts:
        attempt += 1
        emit_safe_provider_telemetry(on_provider_diagnostic, {
            "event": "provider_http_attempt_started",
            "provider_attempt": attempt + 1,
            "model": model,
            "usage_source": "unavailable",
        })
        attempt_started = perf_counter()
        outcome = "error"
        try:
            async with service_runtime.concurrency.provider.slot(max_wait_seconds=timeout_ms / 1000):
                response = await asyncio.wait_for(
                    asyncio.to_thread(run),
                    timeout=timeout_ms / 1000,
                )
            outcome = "success"
            emit_safe_provider_telemetry(on_provider_diagnostic, {
                "event": "provider_http_attempt_succeeded",
                "provider_attempt": attempt + 1,
                "model": model,
                "usage_source": "unavailable",
            })
            return response
        except AppError:
            outcome = "busy"
            raise
        except asyncio.TimeoutError as error:
            outcome = "timeout"
            if on_provider_diagnostic is not None:
                emit_safe_provider_telemetry(on_provider_diagnostic, {
                    "event": "provider_timeout",
                    "internal_failure_code": "AI_PROVIDER_TIMEOUT",
                    "model": model,
                    "provider_http_status": None,
                    "provider_attempt": attempt + 1,
                    "timeout_ms": timeout_ms,
                    "usage_source": "unavailable",
                })
            else:
                logger.error(
                    "ai_provider_timeout model=%s timeout_ms=%s",
                    model,
                    timeout_ms,
                )
            raise HTTPException(
                status_code=504,
                detail={
                    "code": "AI_PROVIDER_TIMEOUT",
                    "message": "AI provider phản hồi quá lâu. Vui lòng thử lại sau.",
                },
            ) from error
        except Exception as error:
            status_code = provider_http_error_status(error)
            provider_error = str(error)
            if status_code == 429 or "RESOURCE_EXHAUSTED" in provider_error:
                retry_hint = provider_retry_hint_seconds(error)
                # The provider message is classified in memory and never logged; only the
                # category and the hint seconds leave this function.
                limit = classify_provider_limit(error, retry_hint)
                retry_after = limit.retry_after_seconds
                wait_seconds: float | None = None
                if limit.kind == "rate_limited":
                    if rate_limit_max_wait_ms is not None:
                        # The floor keeps a "retry in 0s" hint from looping without using the bound.
                        candidate = max(PROVIDER_RATE_LIMIT_MIN_WAIT_SECONDS,
                                        retry_after if retry_after is not None
                                        else provider_retry_delay_seconds(rate_limit_retries))
                        if rate_limit_waited_seconds + candidate <= rate_limit_max_wait_ms / 1000:
                            wait_seconds = candidate
                    elif (retry_after is not None and retry_after <= settings.provider_retry_max_ms / 1000
                          and attempt + 1 < max_attempts):
                        # Legacy: one retry when the per-minute limit clears quickly.
                        wait_seconds = retry_after
                if wait_seconds is not None:
                    outcome = "rate_limited_retry"
                    emit_safe_provider_telemetry(on_provider_diagnostic, {
                        "event": "provider_rate_limited_retry",
                        "model": model,
                        "provider_http_status": status_code,
                        "provider_attempt": attempt + 1,
                        "provider_limit_category": limit.kind,
                        "retry_after_ms": round(wait_seconds * 1000),
                        "usage_source": "unavailable",
                    })
                    logger.warning(
                        "ai_provider_rate_limited category=%s retry_after_s=%.1f waited_s=%.1f model=%s",
                        limit.kind,
                        wait_seconds,
                        rate_limit_waited_seconds,
                        model,
                    )
                    await asyncio.sleep(wait_seconds)
                    if rate_limit_max_wait_ms is not None:
                        # Bounded by the cumulative wait, not by the transient attempts.
                        rate_limit_waited_seconds += wait_seconds
                        rate_limit_retries += 1
                        attempt -= 1
                    continue
                outcome = limit.kind
                if on_provider_diagnostic is not None:
                    emit_safe_provider_telemetry(on_provider_diagnostic, {
                        "event": "provider_quota_exhausted" if limit.kind == "quota_exhausted"
                        else "provider_rate_limited",
                        "model": model,
                        "provider_http_status": status_code,
                        "provider_attempt": attempt + 1,
                        "provider_error_type": type(error).__name__,
                        "provider_limit_category": limit.kind,
                        "retry_after_ms": round(retry_after * 1000) if retry_after is not None else None,
                        "usage_source": "unavailable",
                    })
                else:
                    logger.error(
                        "ai_provider_%s model=%s error_type=%s retry_after_s=%s waited_s=%.1f",
                        limit.kind,
                        model,
                        type(error).__name__,
                        f"{retry_after:.1f}" if retry_after is not None else "none",
                        rate_limit_waited_seconds,
                    )
                if limit.kind == "rate_limited":
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "code": RATE_LIMITED_CODE,
                            "message": "AI provider đang giới hạn tần suất gọi. Vui lòng thử lại sau ít phút.",
                        },
                    ) from error
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "AI_PROVIDER_QUOTA_EXHAUSTED",
                        "message": "AI provider đã hết hạn mức. Vui lòng nạp thêm hạn mức hoặc đổi API key trước khi thử lại.",
                    },
                ) from error
            is_transient = (isinstance(status_code, int) and status_code >= 500) or "UNAVAILABLE" in provider_error
            if is_transient and attempt + 1 < max_attempts:
                outcome = "transient_retry"
                if on_provider_diagnostic is not None:
                    emit_safe_provider_telemetry(on_provider_diagnostic, {
                        "event": "provider_transient_retry",
                        "model": model,
                        "provider_http_status": status_code,
                        "provider_attempt": attempt + 1,
                        "provider_error_type": type(error).__name__,
                        "usage_source": "unavailable",
                    })
                else:
                    logger.warning(
                        "ai_provider_transient_retry model=%s attempt=%s status_code=%s",
                        model,
                        attempt + 1,
                        status_code,
                    )
                await asyncio.sleep(provider_retry_delay_seconds(attempt))
                continue
            if is_transient:
                outcome = "unavailable"
                if on_provider_diagnostic is not None:
                    emit_safe_provider_telemetry(on_provider_diagnostic, {
                        "event": "provider_unavailable",
                        "model": model,
                        "provider_http_status": status_code,
                        "provider_attempt": attempt + 1,
                        "provider_error_type": type(error).__name__,
                        "usage_source": "unavailable",
                    })
                else:
                    logger.error(
                        "ai_provider_unavailable model=%s status_code=%s error_type=%s",
                        model,
                        status_code,
                        type(error).__name__,
                    )
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "AI_PROVIDER_UNAVAILABLE",
                        "message": "AI provider hiện không khả dụng. Vui lòng thử lại sau.",
                    },
                ) from error
            emit_safe_provider_telemetry(on_provider_diagnostic, {
                "event": "provider_request_failed",
                "model": model,
                "provider_attempt": attempt + 1,
                **safe_provider_error_diagnostics(error),
            })
            raise
        finally:
            metrics.PROVIDER_CALLS.labels(operation=operation, outcome=outcome).inc()
            metrics.PROVIDER_CALL_DURATION.labels(operation=operation).observe(perf_counter() - attempt_started)
    raise RuntimeError("Provider retry loop exhausted without a result.")


async def embed_text_batch(
    api_key: str | SecretStr,
    model: str,
    contents: list[str],
    *,
    task_type: str | None = None,
    output_dimensionality: int = 768,
) -> tuple[list[list[float]], AiUsage]:
    effective_model = normalize_embedding_model(model)
    safe_api_key = require_provider_api_key(api_key)

    def run() -> Any:
        client = gemini_infra.gemini_client(safe_api_key, settings.provider_request_timeout_ms)
        config_args: dict[str, Any] = {"outputDimensionality": output_dimensionality}
        if effective_model == "gemini-embedding-001" and task_type:
            config_args["taskType"] = task_type
        config = types.EmbedContentConfig(**config_args)
        return client.models.embed_content(model=effective_model, contents=contents, config=config)

    response = await call_provider_with_timeout(run, effective_model, operation="embed")
    raw_embeddings = getattr(response, "embeddings", None)
    if raw_embeddings is None and isinstance(response, dict):
        raw_embeddings = response.get("embeddings")
    if raw_embeddings is None:
        raw_embeddings = [response]
    embeddings = [get_embedding_values(item) for item in raw_embeddings]
    usage = normalize_usage(embedding_tokens=sum(estimate_tokens(content) for content in contents))
    metrics.PROVIDER_TOKENS.labels(operation="embed", kind="embedding").inc(usage.embeddingTokens)
    return embeddings, usage


async def embed_texts(
    api_key: str | SecretStr,
    model: str,
    contents: list[str],
    *,
    task_type: str | None = None,
    output_dimensionality: int = 768,
) -> tuple[list[list[float]], AiUsage]:
    if not contents:
        return [], AiUsage()

    effective_model = normalize_embedding_model(model)
    batch_size = embedding_batch_size(effective_model)
    embeddings: list[list[float]] = []
    usage = AiUsage()
    for start in range(0, len(contents), batch_size):
        batch_embeddings, batch_usage = await embed_text_batch(
            api_key,
            effective_model,
            contents[start : start + batch_size],
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )
        embeddings.extend(batch_embeddings)
        usage = combine_usage(usage, batch_usage)
    return embeddings, usage


async def generate_content(
    api_key: str | SecretStr,
    model: str,
    prompt: str,
    *,
    max_output_tokens: int,
    json_mode: bool = False,
    response_schema: types.Schema | type[BaseModel] | None = None,
    thinking_config: types.ThinkingConfig | dict[str, Any] | None = None,
    thinking_level: Literal["low", "medium", "high"] | None = None,
    request_timeout_ms: int | None = None,
    on_provider_telemetry: Callable[[dict[str, Any]], None] | None = None,
    rate_limit_max_wait_ms: int | None = None,
) -> tuple[str, AiUsage]:
    safe_api_key = require_provider_api_key(api_key)
    if response_schema is not None and not json_mode:
        raise ValueError("response_schema requires JSON mode.")
    provider_timeout_ms = max(1, request_timeout_ms or settings.provider_request_timeout_ms)

    model_id = model.strip().lower().split("/")[-1]
    gemini_38 = model_id == "gemini-3.8-flash"

    def run() -> Any:
        client = gemini_infra.gemini_client(safe_api_key, provider_timeout_ms)
        config: dict[str, Any] = {"max_output_tokens": max_output_tokens}
        if gemini_38:
            # Gemini 3.8 rejects legacy sampling parameters. Use its supported
            # reasoning-level contract and deliberately ignore old
            # include-thoughts-only configs supplied by legacy call sites.
            config["thinking_config"] = {
                "thinking_level": thinking_level or settings.gemini_38_thinking_level,
            }
        else:
            config["temperature"] = settings.generation_temperature
        if json_mode:
            config["response_mime_type"] = "application/json"
        if response_schema is not None:
            config["response_schema"] = response_schema
        if thinking_config is not None and not gemini_38:
            config["thinking_config"] = thinking_config
        return client.models.generate_content(model=model, contents=prompt, config=config)

    provider_started = perf_counter()
    last_provider_attempt = 0
    if on_provider_telemetry is not None:
        # Size measurements only. Never emit schema text, prompts or local token
        # estimates under provider usage. Failure to measure must not affect AI.
        try:
            schema_object = (
                response_schema.model_json_schema()
                if isinstance(response_schema, type) and issubclass(response_schema, BaseModel)
                else response_schema.model_dump(mode="json", exclude_none=True)
                if isinstance(response_schema, BaseModel) else None
            )
            schema_wire = json.dumps(schema_object, ensure_ascii=False, separators=(",", ":")) if schema_object is not None else ""
            emit_safe_provider_telemetry(on_provider_telemetry, {
                "stage": "provider_transport", "event": "provider_request_started",
                "model": model, "prompt_chars": len(prompt),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "response_schema_chars": len(schema_wire),
                "schema_measurement_source": "local_declared_schema_not_sdk_wire",
                "response_schema_sha256": hashlib.sha256(schema_wire.encode("utf-8")).hexdigest(),
                "configured_max_output_tokens": max_output_tokens,
                "provider_timeout_ms": provider_timeout_ms,
                "usage_source": "unavailable",
            })
        except Exception:
            emit_safe_provider_telemetry(on_provider_telemetry, {
                "stage": "provider_transport", "event": "request_size_unavailable",
                "usage_source": "unavailable",
            })
    def on_provider_diagnostic(metadata: dict[str, Any]) -> None:
        nonlocal last_provider_attempt
        provider_attempt = metadata.get("provider_attempt")
        if type(provider_attempt) is int and provider_attempt > 0:
            last_provider_attempt = provider_attempt
        emit_safe_provider_telemetry(
            on_provider_telemetry,
            {
                "stage": "provider_transport",
                "configured_max_output_tokens": max_output_tokens,
                "duration_ms": int((perf_counter() - provider_started) * 1000),
                **metadata,
            },
        )

    rate_limit_options: dict[str, Any] = (
        {"rate_limit_max_wait_ms": rate_limit_max_wait_ms} if rate_limit_max_wait_ms is not None else {}
    )
    response = await call_provider_with_timeout(
        run,
        model,
        request_timeout_ms=provider_timeout_ms,
        on_provider_diagnostic=on_provider_diagnostic if on_provider_telemetry is not None else None,
        **rate_limit_options,
    )
    # Capture real provider metadata before SDK response access/parsing can fail.
    # An HTTP success is not a validated lesson, nor permission to persist it.
    received = provider_response_telemetry(response, model=model, max_output_tokens=max_output_tokens,
                                          prompt=prompt, response_text="", duration_ms=round((perf_counter() - provider_started) * 1000))
    emit_safe_provider_telemetry(on_provider_telemetry, {
        **{key: value for key, value in received.items() if key.startswith("provider_") or key in {"model", "duration_ms", "configured_max_output_tokens"}},
        "usage_source": "provider" if received["usage_source"] == "provider" else "unavailable",
        "event": "provider_response_received",
        "provider_attempt": last_provider_attempt or 1,
    })
    text = getattr(response, "text", "") or ""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, BaseModel) and not getattr(response_schema, "retain_raw_provider_text", False):
            # Do not synthesize optional response fields from Pydantic defaults.
            # In particular, v4 Course Architect output must stay semantic-only
            # until the deterministic server allocator injects canonical facts.
            text = json.dumps(parsed.model_dump(mode="json", exclude_unset=True), ensure_ascii=False, separators=(",", ":"))
        elif isinstance(parsed, (dict, list)):
            text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    elif response_schema is not None:
        diagnostics = describe_lesson_author_blueprint_response(text)
        candidates = getattr(response, "candidates", None) or []
        if on_provider_telemetry is not None:
            emit_safe_provider_telemetry(on_provider_telemetry, {
                "stage": "provider_response",
                "event": "structured_response_unparsed",
                "schema": getattr(response_schema, "__name__", type(response_schema).__name__),
                "model": model,
                "candidate_count": len(candidates),
                "finish_reasons": [str(getattr(candidate, "finish_reason", None))[:80] for candidate in candidates[:3]],
                "has_prompt_feedback": bool(getattr(response, "prompt_feedback", None)),
                # This helper already reports only structural counters/types.
                "response_diagnostics": diagnostics,
            })
        else:
            logger.warning(
                "structured_response_unparsed schema=%s model=%s candidates=%s finish_reasons=%s prompt_feedback=%s diagnostics=%s",
                getattr(response_schema, "__name__", type(response_schema).__name__),
                model,
                len(candidates),
                [str(getattr(candidate, "finish_reason", None))[:80] for candidate in candidates[:3]],
                bool(getattr(response, "prompt_feedback", None)),
                diagnostics,
            )
    emit_safe_provider_telemetry(
        on_provider_telemetry,
        provider_response_telemetry(
            response,
            model=model,
            max_output_tokens=max_output_tokens,
            prompt=prompt,
            response_text=text,
            duration_ms=max(0, round((perf_counter() - provider_started) * 1000)),
        ),
    )
    usage = usage_from_google_response(response, prompt, text)
    metrics.PROVIDER_TOKENS.labels(operation="generate", kind="input").inc(usage.inputTokens)
    metrics.PROVIDER_TOKENS.labels(operation="generate", kind="output").inc(usage.outputTokens)
    return text, usage


NON_RETRYABLE_PROVIDER_ERROR_CODES = frozenset({
    "AI_PROVIDER_QUOTA_EXHAUSTED",
    # Classified out of AI_PROVIDER_QUOTA_EXHAUSTED; legacy flows keep treating both alike.
    "AI_PROVIDER_RATE_LIMITED",
    "AI_PROVIDER_UNAVAILABLE",
    "AI_PROVIDER_TIMEOUT",
    "AI_STAGED_LESSON_WORKFLOW_TIMEOUT",
})


def http_error_detail(error: HTTPException) -> dict[str, Any]:
    """The structured ``detail`` of a service HTTPException (``{"code", "message"}``), else ``{}``.

    Starlette types ``detail`` as ``str``; the service raises dict details, so it is read untyped.
    """
    detail: Any = error.detail
    return detail if isinstance(detail, dict) else {}


def is_non_retryable_provider_error(error: HTTPException) -> bool:
    detail = http_error_detail(error)
    return detail.get("code") in NON_RETRYABLE_PROVIDER_ERROR_CODES
