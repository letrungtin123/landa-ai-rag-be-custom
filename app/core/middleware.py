from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from app.core.errors import error_payload

AsgiMessage = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[AsgiMessage]]
Send = Callable[[AsgiMessage], Awaitable[None]]


class _RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Reject oversized bodies while ASGI is still streaming them."""

    def __init__(
        self,
        app: Any,
        *,
        max_request_bytes: int,
        idm_max_request_bytes: int,
    ) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes
        self.idm_max_request_bytes = idm_max_request_bytes

    def _limit_for_path(self, path: str) -> int:
        if path == "/v1/lesson-author/orchestration-v2/course-skeleton":
            return self.idm_max_request_bytes
        return self.max_request_bytes

    async def __call__(self, scope: MutableMapping[str, Any], receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        limit = self._limit_for_path(str(scope.get("path") or ""))
        headers = {bytes(key).lower(): bytes(value) for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > limit:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return

        received = 0

        async def limited_receive() -> AsgiMessage:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = json.dumps(
            error_payload("REQUEST_TOO_LARGE", "Request body exceeds the configured limit."),
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
