from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import unittest
from collections.abc import Iterable
from unittest.mock import patch

import httpx
from fastapi import Request
from pydantic import SecretStr

from app.core import metrics
from app.core.config import AuthMode, Settings, settings, validate_startup_settings
from app.core.errors import AppError, unhandled_error_handler
from app.core.logging import correlation_id_var
from app.core.middleware import RequestBodyLimitMiddleware
from app.core.security import (
    HMAC_KEY_ID_HEADER,
    HMAC_REQUEST_ID_HEADER,
    HMAC_SIGNATURE_HEADER,
    HMAC_SIGNATURE_VERSION_HEADER,
    HMAC_TIMESTAMP_HEADER,
    ReplayCache,
    hmac_v2_canonical_message,
    require_configured_service_auth,
    require_internal_auth,
    sign_hmac_request,
    sign_hmac_v2_request,
)
from app.main import app
from app.services import runtime as runtime_service

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
    accept_v1_signature: bool = True,
) -> None:
    await require_internal_auth(
        request,
        auth_mode="hmac",
        service_token="",
        hmac_secrets=secrets,
        clock_skew_seconds=300,
        replay_ttl_seconds=600,
        accept_v1_signature=accept_v1_signature,
        replay_cache=cache or ReplayCache(),
    )


def metric_value(name: str, labels: dict[str, str]) -> float:
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


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

    async def test_a_timestamp_of_other_digits_is_a_401_not_a_500(self) -> None:
        # Starlette reads header bytes as latin-1: b"\xb2" is SUPERSCRIPT TWO, which str.isdigit() accepted and
        # int() rejected (ValueError, an unhandled 500). Thirteen ASCII digits are no Unix time in seconds either.
        for raw in (b"\xb2", b"1791417600\xb2", "\N{ARABIC-INDIC DIGIT ONE}".encode(), b"1" * 13, b"", b"-1"):
            headers = [(name.lower().encode(), value.encode()) for name, value in signed_headers().items()
                       if name != HMAC_TIMESTAMP_HEADER] + [(HMAC_TIMESTAMP_HEADER.lower().encode(), raw)]
            request = build_request(TEST_BODY, headers={})
            request.scope["headers"] = headers
            with (
                self.subTest(timestamp=raw),
                patch("app.core.security.time.time", return_value=float(TEST_TIMESTAMP)),
                self.assertRaises(AppError) as caught,
            ):
                await authenticate(request)
            self.assertEqual((caught.exception.code, caught.exception.http_status), ("UNAUTHORIZED", 401))

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


class AuthRejectionReasonTests(unittest.IsolatedAsyncioTestCase):
    """A refused request logs one short reason and counts it (audit 2026-10-24, O5); never a header value."""

    async def reason_of(self, request: Request, *, auth_mode: AuthMode = "hmac", token: str = "",
                        secrets: str = f"kid1:{TEST_SECRET}", cache: ReplayCache | None = None) -> str:
        with (
            patch("app.core.security.time.time", return_value=float(TEST_TIMESTAMP)),
            self.assertLogs("app.core.security", level="WARNING") as logs,
            self.assertRaises(AppError) as caught,
        ):
            await require_internal_auth(request, auth_mode=auth_mode, service_token=token,
                                        hmac_secrets=secrets, clock_skew_seconds=300, replay_ttl_seconds=600,
                                        accept_v1_signature=True, replay_cache=cache or ReplayCache())
        self.assertEqual(caught.exception.http_status, 401)
        [record] = logs.records
        self.assertEqual(record.getMessage(), "auth_rejected")
        for secret in (TEST_SECRET, TEST_SIGNATURE, "kid1", "kid-unknown", "wrong-token-value"):
            self.assertNotIn(secret, json.dumps(record.__dict__, default=str))
        return str(record.reason)

    async def test_each_refusal_names_its_reason_and_counts_it(self) -> None:
        unsigned = {key: value for key, value in signed_headers().items() if key != HMAC_SIGNATURE_HEADER}
        cases = {
            "missing_header": build_request(TEST_BODY, headers=unsigned),
            "unknown_key": build_request(TEST_BODY, headers=signed_headers(key_id="kid-unknown")),
            "bad_timestamp": build_request(TEST_BODY, headers=signed_headers(timestamp="17914176001234")),
            "bad_signature": build_request(TEST_BODY, headers={**signed_headers(), HMAC_SIGNATURE_HEADER: "zz"}),
            "bad_request_id": build_request(TEST_BODY, headers=signed_headers(request_id="not-a-uuid")),
            "clock_skew": build_request(TEST_BODY, headers=signed_headers(timestamp=str(int(TEST_TIMESTAMP) - 301))),
        }
        cases["bad_signature "] = build_request(b'{"hello":"altered"}', headers=signed_headers())
        for expected, request in cases.items():
            with self.subTest(expected):
                before = metric_value("ai_rag_auth_rejections_total", {"reason": expected.strip()})
                self.assertEqual(await self.reason_of(request), expected.strip())
                self.assertEqual(metric_value("ai_rag_auth_rejections_total", {"reason": expected.strip()}),
                                 before + 1)

    async def test_replay_token_and_configuration_reasons(self) -> None:
        cache = ReplayCache()
        with patch("app.core.security.time.time", return_value=float(TEST_TIMESTAMP)):
            await authenticate(build_request(TEST_BODY, headers=signed_headers()), cache=cache)
        self.assertEqual(await self.reason_of(build_request(TEST_BODY, headers=signed_headers()), cache=cache),
                         "replay")
        token_header = {"X-Landa-AI-Service-Token": "wrong-token-value"}
        self.assertEqual(await self.reason_of(build_request(TEST_BODY, headers={}), auth_mode="token",
                                              token="right-token-value"), "missing_header")
        self.assertEqual(await self.reason_of(build_request(TEST_BODY, headers=token_header), auth_mode="token",
                                              token="right-token-value"), "bad_token")
        self.assertEqual(await self.reason_of(build_request(TEST_BODY, headers=token_header),
                                              auth_mode="token_or_hmac", token="right-token-value"), "bad_token")
        self.assertEqual(await self.reason_of(build_request(TEST_BODY, headers=signed_headers(key_id="kid-unknown")),
                                              auth_mode="token_or_hmac"), "unknown_key")
        self.assertEqual(await self.reason_of(build_request(TEST_BODY, headers=signed_headers()), secrets=""),
                         "not_configured")
        self.assertEqual(await self.reason_of(build_request(TEST_BODY, headers={}), auth_mode="token_or_hmac",
                                              secrets=""), "missing_header")


# The shared test vector of the HMAC v2 spec (security audit 2026-10-24, M1): backend and AI service both
# reproduce it exactly.
V2_SECRET = "test-secret-0123456789abcdef0123456789abcdef"
V2_KEY_ID = "k1"
V2_REQUEST_ID = "2b7e1516-28ae-4d2a-8b0f-6c1e9f1a2b3c"
V2_TIMESTAMP = "1792900000"
V2_PATH = "/v1/chat"
V2_BODY = '{"tenant_id":"t-1","question":"Xin chào"}'.encode()
V2_BODY_SHA256 = "49afe22aa1e95b79611ccaabc43cd20dbac65675b714b31269d111a496f4b9a3"
V2_SIGNATURE = "7c7514c97b51daf6fe0991ba901a5289aaf0ade8a36f1063d9cd48f4a79c891d"
V1_SIGNATURE = "fad90ac1c9b5c73e6c4cc02e07df41d1341b21f69e75a2a96e71838aa7b26822"
UNAUTHORIZED_BODY = b'{"detail":{"code":"UNAUTHORIZED","message":"Unauthorized AI RAG service request."}}'


def v2_headers(*, request_id: str = V2_REQUEST_ID, signature: str = V2_SIGNATURE,
               version: str | None = "2") -> dict[str, str]:
    headers = {HMAC_KEY_ID_HEADER: V2_KEY_ID, HMAC_TIMESTAMP_HEADER: V2_TIMESTAMP,
               HMAC_REQUEST_ID_HEADER: request_id, HMAC_SIGNATURE_HEADER: signature}
    if version is not None:
        headers[HMAC_SIGNATURE_VERSION_HEADER] = version
    return headers


class CapturedLines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class HmacV2Tests(unittest.IsolatedAsyncioTestCase):
    """Signature v2 (security audit 2026-10-24, M1): the key id and the request id are signed as well."""

    async def outcome(self, headers: dict[str, str], *, accept_v1_signature: bool = True,
                      cache: ReplayCache | None = None, secrets: str = f"{V2_KEY_ID}:{V2_SECRET}") -> str:
        """"accepted", "accepted legacy_signature" (logged and counted) or "refused <reason>"."""

        security_logger = logging.getLogger("app.core.security")
        captured = CapturedLines()
        level, propagate = security_logger.level, security_logger.propagate
        security_logger.addHandler(captured)
        security_logger.setLevel(logging.INFO)
        security_logger.propagate = False
        refused: AppError | None = None
        try:
            with patch("app.core.security.time.time", return_value=float(V2_TIMESTAMP)):
                await require_internal_auth(build_request(V2_BODY, headers=headers, path=V2_PATH),
                                            auth_mode="hmac", service_token="", hmac_secrets=secrets,
                                            clock_skew_seconds=300, replay_ttl_seconds=600,
                                            accept_v1_signature=accept_v1_signature,
                                            replay_cache=cache or ReplayCache())
        except AppError as error:
            refused = error
        finally:
            security_logger.removeHandler(captured)
            security_logger.setLevel(level)
            security_logger.propagate = propagate
        if refused is not None:
            self.assertEqual((refused.code, refused.http_status, refused.safe_message),
                             ("UNAUTHORIZED", 401, "Unauthorized AI RAG service request."))
            [record] = captured.records
            self.assertEqual(record.getMessage(), "auth_rejected")
            return f"refused {record.reason}"
        if not captured.records:
            return "accepted"
        [record] = captured.records
        self.assertEqual(record.getMessage(), "auth_accepted")
        return f"accepted {record.reason}"

    def test_the_shared_test_vector(self) -> None:
        self.assertEqual(hashlib.sha256(V2_BODY).hexdigest(), V2_BODY_SHA256)
        self.assertEqual(
            hmac_v2_canonical_message(V2_KEY_ID, V2_REQUEST_ID, V2_TIMESTAMP, "post", V2_PATH, V2_BODY),
            f"v2\nk1\n{V2_REQUEST_ID}\n{V2_TIMESTAMP}\nPOST\n/v1/chat\n{V2_BODY_SHA256}".encode(),
        )
        self.assertEqual(sign_hmac_v2_request(V2_SECRET, V2_KEY_ID, V2_REQUEST_ID, V2_TIMESTAMP, "POST", V2_PATH,
                                              V2_BODY), V2_SIGNATURE)
        self.assertEqual(sign_hmac_request(V2_SECRET, V2_TIMESTAMP, "POST", V2_PATH, V2_BODY), V1_SIGNATURE)

    async def test_the_v2_vector_is_accepted_and_not_counted_as_legacy(self) -> None:
        before = metric_value("ai_rag_auth_legacy_accepted_total", {"reason": "legacy_signature"})
        self.assertEqual(await self.outcome(v2_headers()), "accepted")
        self.assertEqual(metric_value("ai_rag_auth_legacy_accepted_total", {"reason": "legacy_signature"}), before)

    async def test_the_v1_vector_is_accepted_logged_and_counted_while_allowed(self) -> None:
        before = metric_value("ai_rag_auth_legacy_accepted_total", {"reason": "legacy_signature"})
        self.assertEqual(await self.outcome(v2_headers(signature=V1_SIGNATURE, version=None)),
                         "accepted legacy_signature")
        self.assertEqual(metric_value("ai_rag_auth_legacy_accepted_total", {"reason": "legacy_signature"}),
                         before + 1)

    async def test_v1_is_refused_once_the_transition_is_over(self) -> None:
        before = metric_value("ai_rag_auth_rejections_total", {"reason": "legacy_signature"})
        self.assertEqual(await self.outcome(v2_headers(signature=V1_SIGNATURE, version=None),
                                            accept_v1_signature=False), "refused legacy_signature")
        self.assertEqual(metric_value("ai_rag_auth_rejections_total", {"reason": "legacy_signature"}), before + 1)
        self.assertEqual(await self.outcome(v2_headers(), accept_v1_signature=False), "accepted")

    async def test_a_captured_request_sent_again_with_a_new_request_id_is_refused_under_v2(self) -> None:
        cache = ReplayCache()
        fresh_id = "6f1c2d3e-4b5a-4c6d-8e7f-0a1b2c3d4e5f"
        self.assertEqual(await self.outcome(v2_headers(), cache=cache), "accepted")
        self.assertEqual(await self.outcome(v2_headers(), cache=cache), "refused replay")
        self.assertEqual(await self.outcome(v2_headers(request_id=fresh_id), cache=cache), "refused bad_signature")
        # The hole v2 closes: v1 does not sign the request id, so a fresh id lets the same signature in again.
        v1_cache = ReplayCache()
        self.assertEqual(await self.outcome(v2_headers(signature=V1_SIGNATURE, version=None), cache=v1_cache),
                         "accepted legacy_signature")
        self.assertEqual(await self.outcome(v2_headers(request_id=fresh_id, signature=V1_SIGNATURE, version=None),
                                            cache=v1_cache), "accepted legacy_signature")

    async def test_the_v2_signature_covers_the_key_id(self) -> None:
        # Two key ids with the same secret: the k1 signature names k1, so it does not pass as k2.
        secrets = f"k1:{V2_SECRET},k2:{V2_SECRET}"
        self.assertEqual(await self.outcome(v2_headers(), secrets=secrets), "accepted")
        self.assertEqual(await self.outcome({**v2_headers(), HMAC_KEY_ID_HEADER: "k2"}, secrets=secrets),
                         "refused bad_signature")

    async def test_an_unknown_signature_version_is_refused(self) -> None:
        for version in ("1", "3", "", "v2", "02", "2,2"):
            with self.subTest(version=version):
                self.assertEqual(await self.outcome(v2_headers(version=version)), "refused bad_version")


class AuthResponseTests(unittest.TestCase):
    """Every refusal answers the same 401 body: the reason is logged and counted, never answered."""

    def response(self, headers: dict[str, str], *, accept_v1_signature: bool = True) -> httpx.Response:
        async def scenario() -> httpx.Response:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                return await client.post(V2_PATH, content=V2_BODY,
                                         headers={**headers, "Content-Type": "application/json"})

        with (
            patch.object(settings, "auth_mode", "hmac"),
            patch.object(settings, "service_hmac_secrets", SecretStr(f"{V2_KEY_ID}:{V2_SECRET}")),
            patch.object(settings, "hmac_accept_v1_signature", accept_v1_signature),
            patch("app.core.security.time.time", return_value=float(V2_TIMESTAMP)),
            self.assertLogs("app.core.security", level="WARNING"),
        ):
            runtime_service.runtime_state.reset()
            return asyncio.run(scenario())

    def test_every_reason_answers_the_same_401_body(self) -> None:
        cases = {
            "missing_header": {key: value for key, value in v2_headers().items() if key != HMAC_SIGNATURE_HEADER},
            "unknown_key": {**v2_headers(), HMAC_KEY_ID_HEADER: "k-unknown"},
            "bad_timestamp": {**v2_headers(), HMAC_TIMESTAMP_HEADER: "17929000001234"},
            "bad_signature (shape)": {**v2_headers(), HMAC_SIGNATURE_HEADER: "zz"},
            "bad_request_id": {**v2_headers(), HMAC_REQUEST_ID_HEADER: "not-a-uuid"},
            "bad_version": v2_headers(version="3"),
            "clock_skew": {**v2_headers(), HMAC_TIMESTAMP_HEADER: str(int(V2_TIMESTAMP) - 301)},
            "bad_signature (body)": v2_headers(signature="0" * 64),
        }
        answers = {reason: self.response(headers) for reason, headers in cases.items()}
        answers["legacy_signature"] = self.response(v2_headers(signature=V1_SIGNATURE, version=None),
                                                    accept_v1_signature=False)
        self.assertEqual({reason: (answer.status_code, answer.content) for reason, answer in answers.items()},
                         dict.fromkeys(answers, (401, UNAUTHORIZED_BODY)))


class ConfigSecurityTests(unittest.TestCase):
    def test_replay_cache_must_outlive_twice_the_clock_skew(self) -> None:
        configured = Settings(environment="production", database_url="postgresql://configured",
                              supabase_url="https://supabase.invalid", auth_mode="hmac",
                              service_hmac_secrets=f"{V2_KEY_ID}:{V2_SECRET}",
                              auth_clock_skew_seconds=301, auth_replay_ttl_seconds=600)
        with self.assertRaisesRegex(RuntimeError, "AI_RAG_AUTH_REPLAY_TTL_SECONDS must be at least twice"):
            validate_startup_settings(configured)
        validate_startup_settings(configured.model_copy(update={"auth_clock_skew_seconds": 300}))

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
