"""AuthHeaderPrecheckMiddleware (security audit 2026-10-24, L4): a request whose headers already fail is refused
before its body is read, with exactly the answer, log line and counters the route's auth dependency gives."""

from __future__ import annotations

import asyncio
import logging
import unittest
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

from fastapi import Request
from pydantic import SecretStr

from app.api import deps as api_deps
from app.core import metrics
from app.core.config import AuthMode, settings
from app.core.errors import AppError
from app.core.security import (
    HMAC_KEY_ID_HEADER,
    HMAC_REQUEST_ID_HEADER,
    HMAC_SIGNATURE_HEADER,
    HMAC_TIMESTAMP_HEADER,
    TOKEN_HEADER,
    ReplayCache,
    auth_header_rejection,
    require_internal_auth,
    sign_hmac_v2_request,
)
from app.main import app
from app.services import runtime as runtime_service
from tests.unit.core.test_prd0_security import (
    UNAUTHORIZED_BODY,
    V1_SIGNATURE,
    V2_BODY,
    V2_KEY_ID,
    V2_PATH,
    V2_SECRET,
    V2_TIMESTAMP,
    build_request,
    v2_headers,
)

TOKEN = "precheck-test-token-0123456789"


def metric_value(name: str, labels: dict[str, str]) -> float:
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


@contextmanager
def auth_settings(auth_mode: AuthMode = "hmac", *, accept_v1_signature: bool = True) -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(patch.object(settings, "auth_mode", auth_mode))
        stack.enter_context(patch.object(settings, "service_token", SecretStr(TOKEN)))
        stack.enter_context(patch.object(settings, "service_hmac_secrets", SecretStr(f"{V2_KEY_ID}:{V2_SECRET}")))
        stack.enter_context(patch.object(settings, "hmac_accept_v1_signature", accept_v1_signature))
        stack.enter_context(patch("app.core.security.time.time", return_value=float(V2_TIMESTAMP)))
        runtime_service.runtime_state.reset()
        yield
        runtime_service.runtime_state.reset()


class AsgiReply:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.body_reads = 0

    @property
    def status(self) -> int:
        return int(self.messages[0]["status"])

    @property
    def headers(self) -> list[tuple[bytes, bytes]]:
        return [(bytes(key), bytes(value)) for key, value in self.messages[0]["headers"]]

    @property
    def body(self) -> bytes:
        return b"".join(bytes(message.get("body", b"")) for message in self.messages[1:])


def call(method: str, path: str, headers: dict[str, str], body: bytes = V2_BODY) -> AsgiReply:
    """One request through the whole ASGI stack; counts how often the body was asked for."""

    reply = AsgiReply()

    async def scenario() -> None:
        answered = asyncio.Event()

        async def receive() -> dict[str, Any]:
            reply.body_reads += 1
            if reply.body_reads == 1:
                return {"type": "http.request", "body": body, "more_body": False}
            await answered.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            reply.messages.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body"):
                answered.set()

        encoded = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in headers.items()]
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
                 "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
                 "headers": [(b"content-type", b"application/json"), *encoded],
                 "client": ("127.0.0.1", 50000), "server": ("test", 80)}
        await app(scope, receive, send)

    asyncio.run(scenario())
    return reply


REQUEST_IDS = {"X-Request-Id": "req-precheck-1", "X-Correlation-Id": "corr-precheck-1"}
LARGE_BODY = b"x" * (1024 * 1024)


class AuthHeaderPrecheckTests(unittest.TestCase):
    def test_a_request_whose_headers_fail_is_refused_before_its_body_is_read(self) -> None:
        unsigned = {key: value for key, value in v2_headers().items() if key != HMAC_SIGNATURE_HEADER}
        cases = {
            "missing_header": unsigned,
            "unknown_key": {**v2_headers(), HMAC_KEY_ID_HEADER: "k-unknown"},
            "bad_timestamp": {**v2_headers(), HMAC_TIMESTAMP_HEADER: "1792900000.5"},
            "bad_signature": {**v2_headers(), HMAC_SIGNATURE_HEADER: "not-hex"},
            "bad_request_id": {**v2_headers(), HMAC_REQUEST_ID_HEADER: "42"},
            "bad_version": v2_headers(version="9"),
        }
        for reason, headers in cases.items():
            with self.subTest(reason), auth_settings(), self.assertLogs("app.core.security", "WARNING") as logs:
                reply = call("POST", V2_PATH, headers)
            self.assertEqual((reply.status, reply.body, reply.body_reads), (401, UNAUTHORIZED_BODY, 0))
            [record] = logs.records
            self.assertEqual((record.reason, record.route), (reason, V2_PATH))

    def test_the_early_401_is_byte_identical_to_the_one_the_route_dependency_answers(self) -> None:
        with auth_settings(), self.assertLogs("app.core.security", "WARNING"):
            early = call("POST", V2_PATH, {HMAC_KEY_ID_HEADER: V2_KEY_ID, **REQUEST_IDS})
            # Well-formed headers: the precheck lets it through and the route dependency refuses the signature.
            late = call("POST", V2_PATH, {**v2_headers(signature="0" * 64), **REQUEST_IDS})
            v1_refused = call("POST", V2_PATH, {**v2_headers(signature=V1_SIGNATURE, version=None), **REQUEST_IDS})
        self.assertEqual((early.body_reads, late.body_reads), (0, 2))
        self.assertEqual((early.status, early.headers, early.body), (late.status, late.headers, late.body))
        self.assertEqual(early.body, b'{"detail":{"code":"UNAUTHORIZED","message":"Unauthorized AI RAG service '
                                     b'request.","correlation_id":"corr-precheck-1"}}')
        self.assertIn((b"x-request-id", b"req-precheck-1"), early.headers)
        self.assertEqual(v1_refused.body_reads, 2)  # v1 still allowed: the dependency checks it

    def test_the_access_log_and_the_counters_name_the_route_as_before(self) -> None:
        labels = {"route": V2_PATH, "method": "POST", "status": "401"}
        before = (metric_value("ai_rag_http_requests_total", labels),
                  metric_value("ai_rag_auth_rejections_total", {"reason": "missing_header"}))
        with auth_settings(), self.assertLogs("app", "INFO") as logs:
            call("POST", V2_PATH, {})
        self.assertEqual((metric_value("ai_rag_http_requests_total", labels),
                          metric_value("ai_rag_auth_rejections_total", {"reason": "missing_header"})),
                         (before[0] + 1, before[1] + 1))
        [completed] = [record for record in logs.records if record.getMessage() == "http_request_completed"]
        self.assertEqual((completed.route, completed.status, completed.error_code), (V2_PATH, 401, "UNAUTHORIZED"))

    def test_a_signed_request_still_reaches_the_route(self) -> None:
        request_id = "c0ffee00-0000-4000-8000-0000000000aa"
        signature = sign_hmac_v2_request(V2_SECRET, V2_KEY_ID, request_id, V2_TIMESTAMP, "GET", "/v1/meta", b"")
        with auth_settings(accept_v1_signature=False):
            reply = call("GET", "/v1/meta", v2_headers(request_id=request_id, signature=signature), body=b"")
        self.assertEqual(reply.status, 200)
        with auth_settings("token"):
            self.assertEqual(call("GET", "/v1/meta", {TOKEN_HEADER: TOKEN}, body=b"").status, 200)
            with self.assertLogs("app.core.security", "WARNING") as logs:
                refused = call("POST", V2_PATH, {TOKEN_HEADER: "wrong-token"})
        self.assertEqual((refused.status, refused.body_reads, logs.records[0].reason), (401, 0, "bad_token"))

    def test_public_requests_and_paths_outside_the_api_pass_through(self) -> None:
        with auth_settings():
            self.assertEqual(call("GET", "/healthz", {}, body=b"").status, 200)
            unknown = call("POST", "/unknown", {}, body=LARGE_BODY)
            self.assertEqual((unknown.status, unknown.body_reads), (404, 0))

    def test_an_unsigned_request_no_api_route_serves_is_refused_before_its_body_is_read(self) -> None:
        # Review of wave 2, F3: an unknown /v1 path (404), a trailing slash (307) or another method (405) reached the
        # router only after DisconnectCancellationMiddleware had buffered the whole body (up to 16 MiB).
        with auth_settings(), self.assertLogs("app.core.security", "WARNING") as logs:
            refused = {path_and_method: call(path_and_method[0], path_and_method[1], dict(REQUEST_IDS),
                                             body=LARGE_BODY)
                       for path_and_method in (("POST", "/v1/unknown"), ("POST", V2_PATH + "/"), ("POST", "/v1/meta"),
                                               ("GET", V2_PATH), ("POST", "/v1/kb/documents/index/x"))}
            routed = call("POST", V2_PATH, dict(REQUEST_IDS), body=LARGE_BODY)
        for (method, path), reply in refused.items():
            with self.subTest(method=method, path=path):
                self.assertEqual((reply.status, reply.body_reads), (401, 0))
                # Byte for byte the refusal of a routed request: status, headers and body.
                self.assertEqual((reply.status, reply.headers, reply.body),
                                 (routed.status, routed.headers, routed.body))
        self.assertEqual([(record.reason, record.route) for record in logs.records],
                         [("missing_header", "unmatched")] * len(refused) + [("missing_header", V2_PATH)])

    def test_a_request_with_valid_headers_that_no_route_serves_is_answered_unbuffered(self) -> None:
        # Headers the precheck lets through (the dependency would verify the signature): the router answers without
        # the body, and nothing buffers it before.
        with auth_settings():
            cases = {("POST", "/v1/unknown"): 404, ("POST", V2_PATH + "/"): 307, ("POST", "/v1/meta"): 405}
            for (method, path), status in cases.items():
                with self.subTest(method=method, path=path):
                    reply = call(method, path, v2_headers(), body=LARGE_BODY)
                    self.assertEqual((reply.status, reply.body_reads), (status, 0))

    def test_an_overridden_auth_dependency_is_not_checked_early(self) -> None:
        app.dependency_overrides[api_deps.require_internal_token] = lambda: None
        try:
            with auth_settings():
                reply = call("GET", "/v1/meta", {}, body=b"")
                unknown = call("POST", "/v1/unknown", {}, body=LARGE_BODY)
        finally:
            app.dependency_overrides.pop(api_deps.require_internal_token, None)
        self.assertEqual(reply.status, 200)
        self.assertEqual((unknown.status, unknown.body_reads), (404, 0))


class PrecheckParityTests(unittest.IsolatedAsyncioTestCase):
    """Whenever the headers alone settle it, the precheck reaches the dependency's own reason."""

    async def test_every_header_only_refusal_matches_the_dependency(self) -> None:
        base = v2_headers()
        variants: list[dict[str, str]] = [{}, {TOKEN_HEADER: TOKEN}, {TOKEN_HEADER: "wrong"}, base,
                                          {**base, TOKEN_HEADER: "wrong"}, {**base, TOKEN_HEADER: TOKEN}]
        for name in (HMAC_KEY_ID_HEADER, HMAC_TIMESTAMP_HEADER, HMAC_SIGNATURE_HEADER, HMAC_REQUEST_ID_HEADER):
            variants.append({key: value for key, value in base.items() if key != name})
        variants += [{**base, HMAC_KEY_ID_HEADER: "nope"}, {**base, HMAC_TIMESTAMP_HEADER: "x"},
                     {**base, HMAC_SIGNATURE_HEADER: "ab"}, {**base, HMAC_REQUEST_ID_HEADER: "x"},
                     v2_headers(version="7"), v2_headers(signature=V1_SIGNATURE, version=None),
                     {**base, HMAC_KEY_ID_HEADER: "nope", HMAC_TIMESTAMP_HEADER: "x", HMAC_REQUEST_ID_HEADER: "x"}]
        compared = 0
        for mode in ("hmac", "token", "token_or_hmac"):
            for secrets in (f"{V2_KEY_ID}:{V2_SECRET}", ""):
                for accept_v1 in (True, False):
                    for headers in variants:
                        early = auth_header_rejection(build_request(V2_BODY, headers=headers, path=V2_PATH),
                                                      auth_mode=mode, service_token=TOKEN, hmac_secrets=secrets,
                                                      accept_v1_signature=accept_v1)
                        if early is None:
                            continue
                        with self.subTest(mode=mode, secrets=bool(secrets), accept_v1=accept_v1, headers=headers):
                            self.assertEqual(await self.dependency_reason(headers, mode, secrets, accept_v1), early)
                        compared += 1
        self.assertGreater(compared, 60)

    async def dependency_reason(self, headers: dict[str, str], mode: AuthMode, secrets: str, accept_v1: bool) -> str:
        with (
            patch("app.core.security.time.time", return_value=float(V2_TIMESTAMP)),
            self.assertLogs("app.core.security", logging.WARNING) as logs,
            self.assertRaises(AppError),
        ):
            await require_internal_auth(build_request(V2_BODY, headers=headers, path=V2_PATH), auth_mode=mode,
                                        service_token=TOKEN, hmac_secrets=secrets, clock_skew_seconds=300,
                                        replay_ttl_seconds=600, accept_v1_signature=accept_v1,
                                        replay_cache=ReplayCache())
        return str(logs.records[-1].reason)

    def test_the_precheck_needs_no_request_body(self) -> None:
        request = Request({"type": "http", "method": "POST", "path": V2_PATH, "headers": [], "query_string": b""})
        self.assertEqual(auth_header_rejection(request, auth_mode="hmac", service_token="",
                                               hmac_secrets=f"{V2_KEY_ID}:{V2_SECRET}", accept_v1_signature=True),
                         "missing_header")


if __name__ == "__main__":
    unittest.main()
