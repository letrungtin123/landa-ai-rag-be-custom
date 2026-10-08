"""Whole-request deadlines and fallback accounting shared by the routes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, TypeVar

from app.core import metrics
from app.core.concurrency import deadline_seconds
from app.core.errors import AppError
from app.core.logging import SERVICE_LOGGER_NAME

logger = logging.getLogger(SERVICE_LOGGER_NAME)


T = TypeVar("T")


def record_fallback(stage: str, result: Any) -> None:
    if isinstance(result, dict) and result.get("content_origin") == "structured_fallback":
        metrics.FALLBACKS.labels(stage=stage).inc()


async def run_with_deadline(route: str, deadline_ms: int, work: Awaitable[T]) -> T:
    """Bound a whole route. Only the deadline's own expiry maps to 504."""
    timeout = asyncio.timeout(deadline_seconds(deadline_ms))
    try:
        async with timeout:
            return await work
    except TimeoutError:
        if not timeout.expired():
            raise
        metrics.DEADLINE_EXCEEDED.labels(route=route).inc()
        logger.warning("request_deadline_exceeded", extra={"event": "request_deadline_exceeded", "route": route})
        raise AppError("REQUEST_DEADLINE_EXCEEDED", 504, "The request exceeded its processing deadline.") from None
