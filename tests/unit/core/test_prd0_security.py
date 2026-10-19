from __future__ import annotations

import json
import logging
import os
import unittest
from collections.abc import Iterable
from unittest.mock import patch

from fastapi import Request

from app.core.config import Settings, validate_startup_settings
from app.core.errors import AppError, unhandled_error_handler
from app.core.logging import correlation_id_var
from app.core.middleware import RequestBodyLimitMiddleware
from app.core.security import (
    HMAC_KEY_ID_HEADER,
    HMAC_REQUEST_ID_HEADER,
    HMAC_SIGNATURE_HEADER,
    HMAC_TIMESTAMP_HEADER,
    ReplayCache,
    require_configured_service_auth,
    require_internal_auth,
    sign_hmac_request,
)

TEST_SECRET = "test-secret-0123456789"
TEST_PATH = "/v1/lesson-author/orchestration-v2/unit"
TEST_BODY = '{"hello":"xin chào","count":2}'.encode()
TEST_TIMESTAMP = "1791417600"
TEST_REQUEST_ID = "cb7785da-c8b4-4d5b-a02f-30b17e074ccc"
TEST_SIGNATURE = "6b211dacc5906ae64263200f3777a8f22c0839486b50f2c6297a73845fff8e1d"


def build_request(
    body: bytes,
    *,
    headers: dict[str, str],
    path: str = TEST_PATH,
) -> Request:
    encoded_headers = [(key.lower().encode(), value.encode()) for key, value in headers.items()]
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": encoded_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("ai-rag.internal", 443),
    }
    return Request(scope, receive)


def signed_headers(
    *,
    key_id: str = "kid1",
    secret: str = TEST_SECRET,
    timestamp: str = TEST_TIMESTAMP,
    request_id: str = TEST_REQUEST_ID,
    signed_body: bytes = TEST_BODY,
) -> dict[str, str]:
    return {
        HMAC_KEY_ID_HEADER: key_id,
        HMAC_TIMESTAMP_HEADER: timestamp,
        HMAC_REQUEST_ID_HEADER: request_id,
        HMAC_SIGNATURE_HEADER: sign_hmac_request(secret, timestamp, "POST", TEST_PATH, signed_body),
    }


async def authenticate(
    request: Request,
    *,
    secrets: str = f"kid1:{TEST_SECRET}",
    cache: ReplayCache | None = None,
) -> None:
    await require_internal_auth(
        request,
        auth_mode="hmac",
        service_token="",
        hmac_secrets=secrets,
        clock_skew_seconds=300,
        replay_ttl_seconds=600,
        replay_cache=cache or ReplayCache(),
    )


class HmacSecurityTests(unittest.IsolatedAsyncioTestCase):
    def test_python_signature_matches_cross_language_vector(self) -> None:
        self.assertEqual(
            sign_hmac_request(TEST_SECRET, TEST_TIMESTAMP, "POST", TEST_PATH, TEST_BODY),
            TEST_SIGNATURE,
        )

    async def test_valid_signature_is_accepted(self) -> None:
        with patch("app.core.security.time.time", return_value=float(TEST_TIMESTAMP)):
            await authenticate(build_request(TEST_BODY, headers=signed_headers()))

    async def test_wrong_signature_body_key_and_expired_timestamp_are_rejected(self) -> None:
        cases = [
            build_request(TEST_BODY, headers={**signed_headers(), HMAC_SIGNATURE_HEADER: "0" * 64}),
            build_request(b'{"hello":"altered"}', headers=signed_headers()),
            build_request(TEST_BODY, headers=signed_headers(key_id="missing")),
        ]
        for request in cases:
            with (
                self.subTest(headers=dict(request.headers)),
                patch("app.core.security.time.time", return_value=float(TEST_TIMESTAMP)),
                self.assertRaises(AppError),
            ):
                await authenticate(request)

        expired = signed_headers(timestamp=str(int(TEST_TIMESTAMP) - 301))
        with (
            patch("app.core.security.time.time", return_value=float(TEST_TIMESTAMP)),
            self.assertRaises(AppError),
        ):
            await authenticate(build_request(TEST_BODY, headers=expired))

    async def test_replay_is_rejected(self) -> None:
        cache = ReplayCache()
        with patch("app.core.security.time.time", return_value=float(TEST_TIMESTAMP)):
            await authenticate(build_request(TEST_BODY, headers=signed_headers()), cache=cache)
            with self.assertRaises(AppError):
                await authenticate(build_request(TEST_BODY, headers=signed_headers()), cache=cache)

    async def test_rotated_key_is_accepted(self) -> None:
        second_secret = "rotated-secret-0123456789"
        headers = signed_headers(key_id="kid2", secret=second_secret)
        with patch("app.core.security.time.time", return_value=float(TEST_TIMESTAMP)):
            await authenticate(
                build_request(TEST_BODY, headers=headers),
                secrets=f"kid1:{TEST_SECRET},kid2:{second_secret}",
            )

    def test_auth_configuration_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "AI_RAG_SERVICE_TOKEN"):
            require_configured_service_auth(auth_mode="token", service_token="", hmac_secrets="")
        with self.assertRaisesRegex(RuntimeError, "AI_RAG_SERVICE_HMAC_SECRETS"):
            require_configured_service_auth(auth_mode="hmac", service_token="", hmac_secrets="")


class ConfigSecurityTests(unittest.TestCase):
    def test_production_settings_hide_docs_and_default_to_loopback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configured = Settings(environment="production")
        self.assertFalse(configured.is_development)
        self.assertEqual(configured.host, "127.0.0.1")
        self.assertEqual(configured.port, 8010)
        self.assertEqual(configured.workers, 1)

    def test_missing_settings_error_lists_names_without_secret_values(self) -> None:
        configured = Settings(
            environment="production",
            database_url="",
            supabase_url="",
            supabase_service_key="top-secret-value",
            auth_mode="token",
            service_token="",
        )
        with self.assertRaises(RuntimeError) as context:
            validate_startup_settings(configured)
        message = str(context.exception)
        self.assertIn("DATABASE_URL", message)
        self.assertIn("SUPABASE_URL", message)
        self.assertIn("AI_RAG_SERVICE_TOKEN", message)
        self.assertNotIn("top-secret-value", message)

    def test_public_token_only_bind_emits_warning(self) -> None:
        configured = Settings(
            environment="production",
            host="0.0.0.0",  # noqa: S104 - validates the explicit public-bind warning.
            database_url="postgresql://configured",
            supabase_url="https://supabase.invalid",
            supabase_service_key="service-secret",
            auth_mode="token",
            service_token="token-secret",
        )
        with self.assertLogs("app.core.config", logging.WARNING) as captured:
            validate_startup_settings(configured)
        self.assertIn("public_bind_without_hmac", "\n".join(captured.output))


class RequestBodyLimitTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        chunks: Iterable[bytes],
        *,
        content_length: int | None = None,
        path: str = "/v1/chat",
    ) -> tuple[list[dict[str, object]], bool]:
        queue = list(chunks)
        sent: list[dict[str, object]] = []
        reached_app = False

        async def receive() -> dict[str, object]:
            body = queue.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(queue)}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def app(_scope: dict[str, object], wrapped_receive: object, wrapped_send: object) -> None:
            nonlocal reached_app
            reached_app = True
            receiver = wrapped_receive
            sender = wrapped_send
            while True:
                message = await receiver()  # type: ignore[operator]
                if not message.get("more_body"):
                    break
            await sender({"type": "http.response.start", "status": 204, "headers": []})  # type: ignore[operator]
            await sender({"type": "http.response.body", "body": b""})  # type: ignore[operator]

        headers = [] if content_length is None else [(b"content-length", str(content_length).encode())]
        middleware = RequestBodyLimitMiddleware(app, max_request_bytes=5, idm_max_request_bytes=8)
        await middleware({"type": "http", "path": path, "headers": headers}, receive, send)
        return sent, reached_app

    async def test_content_length_is_rejected_before_app(self) -> None:
        sent, reached_app = await self._run([b"ignored"], content_length=6)
        self.assertFalse(reached_app)
        self.assertEqual(sent[0]["status"], 413)
        self.assertIn(b"REQUEST_TOO_LARGE", sent[1]["body"])

    async def test_chunked_body_is_rejected_while_streaming(self) -> None:
        sent, reached_app = await self._run([b"123", b"456"])
        self.assertTrue(reached_app)
        self.assertEqual(sent[0]["status"], 413)

    async def test_idm_route_uses_separate_limit(self) -> None:
        sent, _ = await self._run(
            [b"12345678"],
            path="/v1/lesson-author/orchestration-v2/course-skeleton",
        )
        self.assertEqual(sent[0]["status"], 204)


class SafeErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_unhandled_error_does_not_echo_exception(self) -> None:
        token = correlation_id_var.set("corr-safe")
        try:
            response = await unhandled_error_handler(
                build_request(b"", headers={}),
                RuntimeError("password=leaked C:\\private\\database.sql"),
            )
        finally:
            correlation_id_var.reset(token)
        payload = json.loads(response.body)
        self.assertEqual(payload["detail"]["code"], "INTERNAL_ERROR")
        self.assertEqual(payload["detail"]["correlation_id"], "corr-safe")
        self.assertNotIn("leaked", response.body.decode())
        self.assertNotIn("database.sql", response.body.decode())


if __name__ == "__main__":
    unittest.main()
