"""Request context, access logging, metrics and disconnect cancellation (ASGI)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from contextlib import suppress
from time import perf_counter
from typing import Any

from app.core import metrics
from app.core.lifespan import OTHER_ROUTE_CLASS, RuntimeState
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
# Not counted in flight either: /v1/meta reports the count (a deploy polls it until no request is in flight).
UNCOUNTED_PATHS = OPERATIONAL_PATHS | {"/v1/meta"}
# Path prefix -> route class of the in-flight counters (app.core.lifespan.IN_FLIGHT_ROUTE_CLASSES); first match wins.
IN_FLIGHT_ROUTE_PREFIXES = (
    ("/v1/lesson-author/orchestration-v2/", "ai_id"),
    ("/v1/lesson-author/org-context/", "ai_id"),
    ("/v1/lesson-author/", "lesson_author_legacy"),
    ("/v1/chat", "chat"),
    ("/v1/kb/", "knowledge_base"),
)
# The method label of the HTTP metrics: any client can send any method token, so anything else counts as OTHER
# and the label set stays bounded (security audit 2026-10-24, L3).
METRIC_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
OTHER_METHOD = "OTHER"
HTTP_STATUS_SERVER_ERROR = 500
SUCCESS_STATUSES = range(200, 300)
# A non-2xx answer: the access log names the error code of its body ({"detail": {"code": ...}}), read from at most
# this many bytes, and only when it is a plain code; no other body text is logged (audit 2026-10-24, O5).
ERROR_BODY_CAPTURE_BYTES = 4_096
ERROR_CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{2,95}")
# What Starlette's ServerErrorMiddleware (outside this middleware) answers an escaped exception with: the body of
# app.core.errors.unhandled_error_handler.
UNHANDLED_ERROR_CODE = "INTERNAL_ERROR"


def _header(headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
    for key, value in headers:
        if bytes(key).lower() == name:
            try:
                text = bytes(value).decode("latin-1").strip()
            except UnicodeDecodeError:
                return None
            return text if SAFE_ID_PATTERN.fullmatch(text) else None
    return None


def response_error_code(body: bytes) -> str | None:
    """``detail.code`` of an error answer when it is a plain error code (UNAUTHORIZED, SERVICE_BUSY, ...)."""

    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):  # not JSON, cut at ERROR_BODY_CAPTURE_BYTES, or not UTF-8
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    return code if isinstance(code, str) and ERROR_CODE_PATTERN.fullmatch(code) else None


def method_label(method: str) -> str:
    return method if method in METRIC_METHODS else OTHER_METHOD


def in_flight_route_class(path: str) -> str:
    """The route class a request to ``path`` counts under while it is in flight (bounded: "other" else)."""
    return next((route_class for prefix, route_class in IN_FLIGHT_ROUTE_PREFIXES if path.startswith(prefix)),
                OTHER_ROUTE_CLASS)


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
        counted = path not in UNCOUNTED_PATHS
        route_class = in_flight_route_class(path)
        status_code = HTTP_STATUS_SERVER_ERROR
        error_body = bytearray()
        escaped_error = False
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
            elif message.get("type") == "http.response.body" and status_code not in SUCCESS_STATUSES:
                error_body.extend(bytes(message.get("body", b""))[:ERROR_BODY_CAPTURE_BYTES - len(error_body)])
            await send(message)

        if counted:
            self.state.begin_request(route_class)
        try:
            await self.app(scope, receive, send_with_context)
        except Exception:
            escaped_error = True
            raise
        finally:
            if counted:
                self.state.end_request(route_class)
            duration = perf_counter() - started
            route = route_template(scope)
            label = method_label(method)
            metrics.HTTP_REQUESTS.labels(route=route, method=label, status=str(status_code)).inc()
            metrics.HTTP_REQUEST_DURATION.labels(route=route, method=label).observe(duration)
            if not operational:
                fields: dict[str, Any] = {
                    "event": "http_request_completed",
                    "route": route,
                    "method": method,
                    "status": status_code,
                    "duration_ms": round(duration * 1000),
                }
                if status_code not in SUCCESS_STATUSES:
                    fields["error_code"] = (UNHANDLED_ERROR_CODE if escaped_error and not error_body
                                            else response_error_code(bytes(error_body)))
                logger.info("http_request_completed", extra=fields)
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
