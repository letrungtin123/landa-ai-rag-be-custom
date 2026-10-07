"""Request context, access logging, metrics and disconnect cancellation (ASGI)."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from contextlib import suppress
from time import perf_counter
from typing import Any

from app.core import metrics
from app.core.lifespan import RuntimeState
from app.core.logging import correlation_id_var, request_id_var

logger = logging.getLogger(__name__)

AsgiMessage = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[AsgiMessage]]
Send = Callable[[AsgiMessage], Awaitable[None]]

REQUEST_ID_HEADER = b"x-request-id"
CORRELATION_ID_HEADER = b"x-correlation-id"
LANDA_REQUEST_ID_HEADER = b"x-landa-request-id"
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
OPERATIONAL_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})
HTTP_STATUS_SERVER_ERROR = 500


def _header(headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
    for key, value in headers:
        if bytes(key).lower() == name:
            try:
                text = bytes(value).decode("latin-1").strip()
            except UnicodeDecodeError:
                return None
            return text if SAFE_ID_PATTERN.fullmatch(text) else None
    return None


def route_template(scope: MutableMapping[str, Any]) -> str:
    """Return the matched route template, never a raw path (bounded label cardinality)."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    raw = str(scope.get("path") or "")
    return raw if raw in OPERATIONAL_PATHS else "unmatched"


class RequestContextMiddleware:
    """Bind request/correlation IDs, count in-flight work, log and measure requests."""

    def __init__(self, app: Any, *, state: RuntimeState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: MutableMapping[str, Any], receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers") or [])
        request_id = _header(headers, REQUEST_ID_HEADER) or _header(headers, LANDA_REQUEST_ID_HEADER)
        request_id = request_id or str(uuid.uuid4())
        correlation_id = _header(headers, CORRELATION_ID_HEADER)
        request_token = request_id_var.set(request_id)
        correlation_token = correlation_id_var.set(correlation_id)
        path = str(scope.get("path") or "")
        method = str(scope.get("method") or "GET")
        operational = path in OPERATIONAL_PATHS
        status_code = HTTP_STATUS_SERVER_ERROR
        started = perf_counter()

        async def send_with_context(message: AsgiMessage) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", HTTP_STATUS_SERVER_ERROR))
                response_headers = list(message.get("headers") or [])
                response_headers.append((REQUEST_ID_HEADER, request_id.encode("latin-1")))
                if correlation_id:
                    response_headers.append((CORRELATION_ID_HEADER, correlation_id.encode("latin-1")))
                message["headers"] = response_headers
            await send(message)

        if not operational:
            self.state.begin_request()
        try:
            await self.app(scope, receive, send_with_context)
        finally:
            if not operational:
                self.state.end_request()
            duration = perf_counter() - started
            route = route_template(scope)
            metrics.HTTP_REQUESTS.labels(route=route, method=method, status=str(status_code)).inc()
            metrics.HTTP_REQUEST_DURATION.labels(route=route, method=method).observe(duration)
            if not operational:
                logger.info(
                    "http_request_completed",
                    extra={
                        "event": "http_request_completed",
                        "route": route,
                        "method": method,
                        "status": status_code,
                        "duration_ms": round(duration * 1000),
                    },
                )
            request_id_var.reset(request_token)
            correlation_id_var.reset(correlation_token)


class DisconnectCancellationMiddleware:
    """Cancel route work when the caller disconnects before a response is sent.

    Without this, a long lesson-author request keeps calling the provider (and
    spending tenant tokens) after the backend has already given up. The request
    body is buffered first, so place this middleware *inside* the body-size
    limit. Work already running in a worker thread cannot be interrupted, but
    no new awaits (provider calls, DB writes) are scheduled after cancellation;
    open database transactions roll back.
    """

    def __init__(self, app: Any, *, path_prefix: str = "/v1/") -> None:
        self.app = app
        self.path_prefix = path_prefix

    async def __call__(self, scope: MutableMapping[str, Any], receive: Receive, send: Send) -> None:
        if (
            scope.get("type") != "http"
            or str(scope.get("method") or "") != "POST"
            or not str(scope.get("path") or "").startswith(self.path_prefix)
        ):
            await self.app(scope, receive, send)
            return

        buffered: list[AsgiMessage] = []
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            buffered.append(message)
            if not message.get("more_body", False):
                break

        disconnected = asyncio.Event()
        replay = iter(buffered)

        async def replay_receive() -> AsgiMessage:
            with suppress(StopIteration):
                return next(replay)
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def watch_disconnect() -> None:
            while True:
                message = await receive()
                if message.get("type") == "http.disconnect":
                    disconnected.set()
                    return

        app_task = asyncio.ensure_future(self.app(scope, replay_receive, send))
        watch_task = asyncio.ensure_future(watch_disconnect())
        try:
            await asyncio.wait({app_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            app_task.cancel()
            watch_task.cancel()
            raise
        if app_task.done():
            watch_task.cancel()
            with suppress(asyncio.CancelledError):
                await watch_task
            app_task.result()
            return

        app_task.cancel()
        with suppress(asyncio.CancelledError):
            await app_task
        route = route_template(scope)
        metrics.HTTP_CLIENT_DISCONNECTS.labels(route=route).inc()
        logger.warning(
            "http_client_disconnected",
            extra={"event": "http_client_disconnected", "route": route},
        )
