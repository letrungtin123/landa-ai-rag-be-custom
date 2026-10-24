"""One bounded provider call: deadline, concurrency slot, transient retries, 429 waits and call metrics.

The SDK is synchronous: each attempt runs on the process runtime's provider threads (two per provider slot,
not the default executor) under the request's own deadline (``pool.request_http_options``) and a provider slot
of its ``WorkloadLimiter``; ``app.services.runtime`` installs both with ``use_provider_slots`` (infra never
imports services).
Failures leave as ``ProviderError`` with a code of ``app.core.error_codes``; other SDK errors are
re-raised unchanged after a content-free diagnostic.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from app.core import metrics
from app.core.concurrency import WorkloadLimiter
from app.core.config import settings
from app.core.errors import AppError, ProviderError
from app.core.handled_exceptions import report_handled_exception
from app.core.logging import SERVICE_LOGGER_NAME, log_event
from app.infra.gemini import retry
from app.infra.gemini.errors import provider_http_error_status, safe_provider_error_diagnostics
from app.infra.gemini.telemetry import emit_safe_provider_telemetry
from app.infra.provider_limits import RATE_LIMITED_CODE, ProviderLimit, classify_provider_limit

logger = logging.getLogger(SERVICE_LOGGER_NAME)

HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_MIN = 500


ProviderCallRunner = Callable[[Callable[[], Any]], Awaitable[Any]]


@dataclass(slots=True)
class _ProviderSlots:
    limiter: Callable[[], WorkloadLimiter] | None = None
    runner: Callable[[], ProviderCallRunner] | None = None


_provider_slots = _ProviderSlots()


def use_provider_slots(limiter: Callable[[], WorkloadLimiter], runner: Callable[[], ProviderCallRunner]) -> None:
    """Install the provider limiter and the provider-thread runner of the process runtime, read at every attempt."""

    _provider_slots.limiter = limiter
    _provider_slots.runner = runner


def provider_limiter() -> WorkloadLimiter:
    if _provider_slots.limiter is None:
        raise RuntimeError("The provider concurrency limiter is not installed (app.services.runtime).")
    return _provider_slots.limiter()


async def run_in_provider_thread(run: Callable[[], Any]) -> Any:
    """One synchronous SDK call on the provider threads of the process runtime."""

    if _provider_slots.runner is None:
        raise RuntimeError("The provider threads are not installed (app.services.runtime).")
    return await _provider_slots.runner()(run)


def _timeout_error(
    model: str, timeout_ms: int, attempt: int, on_provider_diagnostic: Callable[[dict[str, Any]], None] | None,
) -> ProviderError:
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
        log_event(logger, logging.ERROR, "ai_provider_timeout", model=model, timeout_ms=timeout_ms)
    return ProviderError("AI_PROVIDER_TIMEOUT", "AI provider phản hồi quá lâu. Vui lòng thử lại sau.")


def _rate_limit_wait(
    limit: ProviderLimit,
    *,
    attempt: int,
    max_attempts: int,
    rate_limit_retries: int,
    rate_limit_waited_seconds: float,
    rate_limit_max_wait_ms: int | None,
) -> float | None:
    """Seconds to wait before retrying a rate-limited attempt, or None when the 429 is final."""

    retry_after = limit.retry_after_seconds
    if limit.kind != "rate_limited":
        return None
    if rate_limit_max_wait_ms is not None:
        # The floor keeps a "retry in 0s" hint from looping without using the bound.
        candidate = max(retry.PROVIDER_RATE_LIMIT_MIN_WAIT_SECONDS,
                        retry_after if retry_after is not None
                        else retry.provider_retry_delay_seconds(rate_limit_retries))
        if rate_limit_waited_seconds + candidate <= rate_limit_max_wait_ms / 1000:
            return candidate
        return None
    if (retry_after is not None and retry_after <= settings.provider_retry_max_ms / 1000
            and attempt + 1 < max_attempts):
        # Legacy: one retry when the per-minute limit clears quickly.
        return retry_after
    return None


def _note_rate_limited_retry(
    on_provider_diagnostic: Callable[[dict[str, Any]], None] | None,
    *,
    model: str,
    status_code: int | None,
    attempt: int,
    limit: ProviderLimit,
    wait_seconds: float,
    rate_limit_waited_seconds: float,
) -> None:
    emit_safe_provider_telemetry(on_provider_diagnostic, {
        "event": "provider_rate_limited_retry",
        "model": model,
        "provider_http_status": status_code,
        "provider_attempt": attempt + 1,
        "provider_limit_category": limit.kind,
        "retry_after_ms": round(wait_seconds * 1000),
        "usage_source": "unavailable",
    })
    log_event(logger, logging.WARNING, "ai_provider_rate_limited", category=limit.kind,
              retry_after_s=round(wait_seconds, 1), waited_s=round(rate_limit_waited_seconds, 1),
              model=model)


def _limit_error(
    limit: ProviderLimit,
    error: Exception,
    *,
    model: str,
    status_code: int | None,
    attempt: int,
    rate_limit_waited_seconds: float,
    on_provider_diagnostic: Callable[[dict[str, Any]], None] | None,
) -> ProviderError:
    retry_after = limit.retry_after_seconds
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
        log_event(logger, logging.ERROR, f"ai_provider_{limit.kind}", model=model,
                  error_type=type(error).__name__,
                  retry_after_s=round(retry_after, 1) if retry_after is not None else None,
                  waited_s=round(rate_limit_waited_seconds, 1))
    if limit.kind == "rate_limited":
        return ProviderError(RATE_LIMITED_CODE,
                             "AI provider đang giới hạn tần suất gọi. Vui lòng thử lại sau ít phút.")
    return ProviderError(
        "AI_PROVIDER_QUOTA_EXHAUSTED",
        "AI provider đã hết hạn mức. Vui lòng nạp thêm hạn mức hoặc đổi API key trước khi thử lại.",
    )


def _unavailable_error(
    error: Exception,
    *,
    model: str,
    status_code: int | None,
    attempt: int,
    on_provider_diagnostic: Callable[[dict[str, Any]], None] | None,
) -> ProviderError:
    report_handled_exception(logger, error, stage=metrics.current_provider_stage(),
                             outcome="AI_PROVIDER_UNAVAILABLE", model=model, provider_attempt=attempt + 1)
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
        log_event(logger, logging.ERROR, "ai_provider_unavailable", model=model,
                  status_code=status_code, error_type=type(error).__name__)
    return ProviderError("AI_PROVIDER_UNAVAILABLE", "AI provider hiện không khả dụng. Vui lòng thử lại sau.")


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

    ``asyncio.wait_for(run_in_provider_thread(...))`` bounds this coroutine, but a
    timeout cannot prove that a synchronous Google client request already
    running in the worker thread was cancelled at HTTP level.  The request's
    own ``http_options`` timeout (``pool.request_http_options``) remains
    the request-level bound; do not treat cancellation of this await as hard
    provider cancellation.

    A 429 is classified by ``classify_provider_limit``: an exhausted key raises
    ``AI_PROVIDER_QUOTA_EXHAUSTED`` at once, a rate limit raises
    ``AI_PROVIDER_RATE_LIMITED`` (both 503). By default a rate limit is retried
    once when the server hint is short (legacy behaviour). A caller that passes
    ``rate_limit_max_wait_ms`` (IDM) waits the server hint (or a backoff) as
    often as the cumulative wait stays within that bound; those waits do not use
    up the transient-retry attempts.
    """
    timeout_ms = max(1, request_timeout_ms or settings.provider_request_timeout_ms)
    max_attempts = max(1, retry.PROVIDER_TRANSIENT_MAX_ATTEMPTS)
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
            async with provider_limiter().slot(max_wait_seconds=timeout_ms / 1000):
                response = await asyncio.wait_for(run_in_provider_thread(run), timeout=timeout_ms / 1000)
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
        except TimeoutError as error:
            outcome = "timeout"
            raise _timeout_error(model, timeout_ms, attempt, on_provider_diagnostic) from error
        except Exception as error:
            status_code = provider_http_error_status(error)
            provider_error = str(error)
            if status_code == HTTP_TOO_MANY_REQUESTS or "RESOURCE_EXHAUSTED" in provider_error:
                # The provider message is classified in memory and never logged; only the
                # category and the hint seconds leave this function.
                limit = classify_provider_limit(error, retry.provider_retry_hint_seconds(error))
                wait_seconds = _rate_limit_wait(
                    limit, attempt=attempt, max_attempts=max_attempts, rate_limit_retries=rate_limit_retries,
                    rate_limit_waited_seconds=rate_limit_waited_seconds, rate_limit_max_wait_ms=rate_limit_max_wait_ms)
                if wait_seconds is not None:
                    outcome = "rate_limited_retry"
                    _note_rate_limited_retry(on_provider_diagnostic, model=model, status_code=status_code,
                                             attempt=attempt, limit=limit, wait_seconds=wait_seconds,
                                             rate_limit_waited_seconds=rate_limit_waited_seconds)
                    await asyncio.sleep(wait_seconds)
                    if rate_limit_max_wait_ms is not None:
                        # Bounded by the cumulative wait, not by the transient attempts.
                        rate_limit_waited_seconds += wait_seconds
                        rate_limit_retries += 1
                        attempt -= 1
                    continue
                outcome = limit.kind
                raise _limit_error(limit, error, model=model, status_code=status_code, attempt=attempt,
                                   rate_limit_waited_seconds=rate_limit_waited_seconds,
                                   on_provider_diagnostic=on_provider_diagnostic) from error
            is_transient = ((isinstance(status_code, int) and status_code >= HTTP_SERVER_ERROR_MIN)
                            or "UNAVAILABLE" in provider_error)
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
                    log_event(logger, logging.WARNING, "ai_provider_transient_retry", model=model,
                              attempt=attempt + 1, status_code=status_code)
                await asyncio.sleep(retry.provider_retry_delay_seconds(attempt))
                continue
            if is_transient:
                outcome = "unavailable"
                raise _unavailable_error(error, model=model, status_code=status_code, attempt=attempt,
                                         on_provider_diagnostic=on_provider_diagnostic) from error
            emit_safe_provider_telemetry(on_provider_diagnostic, {
                "event": "provider_request_failed",
                "model": model,
                "provider_attempt": attempt + 1,
                **safe_provider_error_diagnostics(error),
            })
            raise
        finally:
            stage = metrics.current_provider_stage()
            metrics.PROVIDER_CALLS.labels(operation=operation, outcome=outcome, stage=stage).inc()
            metrics.PROVIDER_CALL_DURATION.labels(operation=operation, stage=stage).observe(
                perf_counter() - attempt_started)
    raise RuntimeError("Provider retry loop exhausted without a result.")
