"""Application lifespan: startup, readiness state and graceful drain."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Route classes of the in-flight counters reported by /v1/meta (app.core.request_context.in_flight_route_class).
IN_FLIGHT_ROUTE_CLASSES = ("ai_id", "chat", "knowledge_base", "lesson_author_legacy", "other")
OTHER_ROUTE_CLASS = "other"


class RuntimeState:
    """Tracks in-flight requests and whether the process is draining.

    uvicorn stops accepting connections on SIGTERM and waits for open requests
    (``timeout_graceful_shutdown``). This state lets ``/readyz`` report
    "not ready" during that window and lets shutdown wait for in-flight work
    before closing the database pool underneath it.
    """

    def __init__(self) -> None:
        self.draining = False
        self.started = False
        self.in_flight = 0
        self._in_flight_by_class = dict.fromkeys(IN_FLIGHT_ROUTE_CLASSES, 0)
        self._idle: asyncio.Event | None = None

    def reset(self) -> None:
        self.draining = False
        self.started = False
        self.in_flight = 0
        self._in_flight_by_class = dict.fromkeys(IN_FLIGHT_ROUTE_CLASSES, 0)
        self._idle = None

    def begin_request(self, route_class: str = OTHER_ROUTE_CLASS) -> None:
        self.in_flight += 1
        key = route_class if route_class in self._in_flight_by_class else OTHER_ROUTE_CLASS
        self._in_flight_by_class[key] += 1
        if self._idle is not None:
            self._idle.clear()

    def end_request(self, route_class: str = OTHER_ROUTE_CLASS) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        key = route_class if route_class in self._in_flight_by_class else OTHER_ROUTE_CLASS
        self._in_flight_by_class[key] = max(0, self._in_flight_by_class[key] - 1)
        if self.in_flight == 0 and self._idle is not None:
            self._idle.set()

    def in_flight_summary(self) -> dict[str, Any]:
        """Requests being served now, in all and per route class: a deploy waits for ``total`` 0 (audit O1)."""
        return {"total": self.in_flight, "by_route_class": dict(self._in_flight_by_class)}

    async def drain(self, grace_seconds: float) -> bool:
        """Mark draining and wait for in-flight requests; return True when idle."""
        self.draining = True
        if self.in_flight == 0:
            return True
        self._idle = asyncio.Event()
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=grace_seconds)
            return True
        except TimeoutError:
            logger.warning(
                "shutdown_grace_expired",
                extra={"event": "shutdown_grace_expired", "in_flight": self.in_flight},
            )
            return False


def build_lifespan(
    *,
    state: RuntimeState,
    startup: Callable[[], Awaitable[None]],
    shutdown: Callable[[], Awaitable[None]],
    grace_seconds: float,
) -> Callable[[Any], Any]:
    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        state.reset()
        await startup()
        state.started = True
        logger.info("service_started", extra={"event": "service_started"})
        try:
            yield
        finally:
            idle = await state.drain(grace_seconds)
            await shutdown()
            logger.info("service_stopped", extra={"event": "service_stopped", "drained": idle})

    return lifespan
