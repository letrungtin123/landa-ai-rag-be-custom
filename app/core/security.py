from __future__ import annotations

import hashlib
import hmac
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from fastapi import Request
from fastapi.routing import APIRoute
from starlette.datastructures import Headers
from starlette.routing import BaseRoute, Match

from app.core import metrics
from app.core.config import AuthMode
from app.core.errors import AppError, app_error_handler
from app.core.logging import SECRET_PATTERNS, redact_secret_like_values
from app.core.request_context import API_PATH_PREFIX, UNROUTED_SCOPE_KEY, AsgiMessage, Receive, Send, route_template

logger = logging.getLogger(__name__)

HMAC_KEY_ID_HEADER = "X-Landa-Key-Id"
HMAC_TIMESTAMP_HEADER = "X-Landa-Timestamp"
HMAC_SIGNATURE_HEADER = "X-Landa-Signature"
HMAC_REQUEST_ID_HEADER = "X-Landa-Request-Id"
# "2": the signature covers the key id and the request id as well (security audit 2026-10-24, M1). Absent: v1.
HMAC_SIGNATURE_VERSION_HEADER = "X-Landa-Signature-Version"
HMAC_SIGNATURE_V2 = "2"
TOKEN_HEADER = "X-Landa-AI-Service-Token"  # noqa: S105 - HTTP header name, not a token.
HEX_SIGNATURE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
# ASCII digits only: str.isdigit() also accepts other scripts' digits and SUPERSCRIPT TWO, which int() rejects (an
# unsigned request then ended in an unhandled 500 instead of 401). Twelve digits hold any Unix time in seconds.
HMAC_TIMESTAMP_PATTERN = re.compile(r"[0-9]{1,12}")
# 32 characters at least (security audit 2026-10-24, L2): `openssl rand -hex 32` gives 64.
MINIMUM_HMAC_SECRET_LENGTH = 32
# Why a request was refused: the bounded reason of the auth_rejected log line and of ai_rag_auth_rejections_total
# (audit 2026-10-24, O5). Never a header value, a key id or a secret. "bad_version": an unknown signature version;
# "legacy_signature": a v1 signature while AI_RAG_HMAC_ACCEPT_V1_SIGNATURE is false.
AuthRejection = Literal["missing_header", "unknown_key", "bad_timestamp", "bad_signature", "bad_request_id",
                        "bad_version", "legacy_signature", "clock_skew", "replay", "bad_token", "not_configured"]
SignatureVersion = Literal["1", "2"]
# The reason of the auth_accepted log line and of ai_rag_auth_legacy_accepted_total: a request let in by a path
# that is being retired, so operators can see its traffic reach 0 before turning it off.
LEGACY_SIGNATURE = "legacy_signature"


@dataclass(frozen=True, slots=True)
class HmacCredential:
    key_id: str
    secret: str


@dataclass(frozen=True, slots=True)
class _SignedRequest:
    """The HMAC headers of a request that passed every check that needs no clock and no body."""

    credential: HmacCredential
    timestamp: str
    signature: str
    request_id: str
    version: SignatureVersion


class ReplayCache:
    def __init__(self, *, ttl_seconds: int = 600, max_entries: int = 100_000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def consume(self, request_id: str, *, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        cutoff = current - self.ttl_seconds
        with self._lock:
            while self._entries:
                _, created = next(iter(self._entries.items()))
                if created >= cutoff:
                    break
                self._entries.popitem(last=False)
            if request_id in self._entries:
                return False
            self._entries[request_id] = current
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return True


_replay_cache = ReplayCache()


def parse_hmac_secrets(value: str) -> dict[str, HmacCredential]:
    credentials: dict[str, HmacCredential] = {}
    if not value.strip():
        return credentials
    for entry in value.split(","):
        key_id, separator, secret = entry.strip().partition(":")
        if (
            not separator
            or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", key_id)
            or len(secret) < MINIMUM_HMAC_SECRET_LENGTH
            or key_id in credentials
        ):
            raise RuntimeError("AI_RAG_SERVICE_HMAC_SECRETS is invalid.")
        credentials[key_id] = HmacCredential(key_id=key_id, secret=secret)
    return credentials


def hmac_canonical_message(timestamp: str, method: str, path: str, body: bytes) -> bytes:
    body_digest = hashlib.sha256(body).hexdigest()
    return f"{timestamp}\n{method.upper()}\n{path}\n{body_digest}".encode()


def sign_hmac_request(secret: str, timestamp: str, method: str, path: str, body: bytes) -> str:
    return hmac.new(
        secret.encode(),
        hmac_canonical_message(timestamp, method, path, body),
        hashlib.sha256,
    ).hexdigest()


def hmac_v2_canonical_message(  # noqa: PLR0917 - the lines of the canonical message, in their order
    key_id: str, request_id: str, timestamp: str, method: str, path: str, body: bytes,
) -> bytes:
    """``v2``, the key id, the request id and the timestamp exactly as sent, METHOD, path, sha256 of the body."""

    body_digest = hashlib.sha256(body).hexdigest()
    return f"v2\n{key_id}\n{request_id}\n{timestamp}\n{method.upper()}\n{path}\n{body_digest}".encode()


def sign_hmac_v2_request(  # noqa: PLR0917 - the secret, then the lines of the canonical message in their order
    secret: str, key_id: str, request_id: str, timestamp: str, method: str, path: str, body: bytes,
) -> str:
    return hmac.new(
        secret.encode(),
        hmac_v2_canonical_message(key_id, request_id, timestamp, method, path, body),
        hashlib.sha256,
    ).hexdigest()


def require_configured_service_auth(
    *,
    auth_mode: AuthMode,
    service_token: str,
    hmac_secrets: str,
) -> None:
    parsed = parse_hmac_secrets(hmac_secrets)
    if auth_mode == "token" and not service_token:
        raise RuntimeError("Missing required environment variables: AI_RAG_SERVICE_TOKEN")
    if auth_mode == "hmac" and not parsed:
        raise RuntimeError("Missing required environment variables: AI_RAG_SERVICE_HMAC_SECRETS")
    if auth_mode == "token_or_hmac" and not service_token and not parsed:
        raise RuntimeError(
            "Missing required environment variables: "
            "AI_RAG_SERVICE_HMAC_SECRETS, AI_RAG_SERVICE_TOKEN"
        )


def _token_is_valid(received: str, expected: str) -> bool:
    if not received or not expected:
        return False
    return hmac.compare_digest(received.encode(), expected.encode())


def _signed_request(
    headers: Headers,
    credentials: dict[str, HmacCredential],
    *,
    accept_v1_signature: bool,
) -> _SignedRequest | AuthRejection:
    """The checks that need neither the clock nor the body, in their order: the first failing one, or the headers."""

    key_id = headers.get(HMAC_KEY_ID_HEADER, "")
    timestamp = headers.get(HMAC_TIMESTAMP_HEADER, "")
    signature = headers.get(HMAC_SIGNATURE_HEADER, "").lower()
    request_id = headers.get(HMAC_REQUEST_ID_HEADER, "")
    if not (key_id and timestamp and signature and request_id):
        return "missing_header"
    credential = credentials.get(key_id)
    if credential is None:
        return "unknown_key"
    if not HMAC_TIMESTAMP_PATTERN.fullmatch(timestamp):
        return "bad_timestamp"
    if not HEX_SIGNATURE_PATTERN.fullmatch(signature):
        return "bad_signature"
    try:
        UUID(request_id)
    except (ValueError, AttributeError):
        return "bad_request_id"
    version: SignatureVersion
    sent_version = headers.get(HMAC_SIGNATURE_VERSION_HEADER)
    if sent_version is None:
        if not accept_v1_signature:
            return "legacy_signature"
        version = "1"
    elif sent_version == HMAC_SIGNATURE_V2:
        version = "2"
    else:
        return "bad_version"
    return _SignedRequest(credential=credential, timestamp=timestamp, signature=signature, request_id=request_id,
                          version=version)


async def _hmac_rejection(
    request: Request,
    credentials: dict[str, HmacCredential],
    *,
    clock_skew_seconds: int,
    replay_cache: ReplayCache,
    accept_v1_signature: bool,
) -> tuple[AuthRejection | None, SignatureVersion | None]:
    """``None`` when a configured key signed the request (its request id is then spent), else why not; and the
    signature version once the headers name one."""

    signed = _signed_request(request.headers, credentials, accept_v1_signature=accept_v1_signature)
    if isinstance(signed, str):
        return signed, None
    current = int(time.time())
    if abs(current - int(signed.timestamp)) > clock_skew_seconds:
        return "clock_skew", signed.version
    body = await request.body()
    secret = signed.credential.secret
    if signed.version == HMAC_SIGNATURE_V2:
        expected = sign_hmac_v2_request(secret, signed.credential.key_id, signed.request_id, signed.timestamp,
                                        request.method, request.url.path, body)
    else:
        expected = sign_hmac_request(secret, signed.timestamp, request.method, request.url.path, body)
    if not hmac.compare_digest(signed.signature.encode(), expected.encode()):
        return "bad_signature", signed.version
    # The request id is spent only once the signature is valid, so a forged request cannot burn a real one.
    return (None if replay_cache.consume(signed.request_id, now=float(current)) else "replay"), signed.version


def _rejection_reason(auth_mode: AuthMode, token_sent: bool, hmac_rejection: AuthRejection) -> AuthRejection:
    """The reason to report: the token's when the token is the only way in or no HMAC headers came."""

    if auth_mode == "token" or (auth_mode == "token_or_hmac" and hmac_rejection in {"missing_header",
                                                                                    "not_configured"}):
        return "bad_token" if token_sent else "missing_header"
    return hmac_rejection


def auth_rejected(request: Request, reason: AuthRejection) -> AppError:
    """Count and log a refusal; the error every refusal answers with (one 401 body for every reason)."""

    metrics.AUTH_REJECTIONS.labels(reason=reason).inc()
    logger.warning("auth_rejected", extra={"event": "auth_rejected", "reason": reason,
                                           "route": route_template(request.scope)})
    return AppError("UNAUTHORIZED", 401, "Unauthorized AI RAG service request.")


async def require_internal_auth(
    request: Request,
    *,
    auth_mode: AuthMode,
    service_token: str,
    hmac_secrets: str,
    clock_skew_seconds: int,
    replay_ttl_seconds: int,
    accept_v1_signature: bool,
    replay_cache: ReplayCache | None = None,
) -> None:
    token = request.headers.get(TOKEN_HEADER, "")
    token_valid = _token_is_valid(token, service_token)
    if auth_mode == "token" and token_valid:
        return

    credentials = parse_hmac_secrets(hmac_secrets)
    active_cache = replay_cache or _replay_cache
    active_cache.ttl_seconds = replay_ttl_seconds
    hmac_rejection: AuthRejection | None = "not_configured"
    version: SignatureVersion | None = None
    if auth_mode in {"hmac", "token_or_hmac"} and credentials:
        hmac_rejection, version = await _hmac_rejection(
            request,
            credentials,
            clock_skew_seconds=clock_skew_seconds,
            replay_cache=active_cache,
            accept_v1_signature=accept_v1_signature,
        )
    if hmac_rejection is None and version == "1":
        metrics.AUTH_LEGACY_ACCEPTED.labels(reason=LEGACY_SIGNATURE).inc()
        logger.info("auth_accepted", extra={"event": "auth_accepted", "reason": LEGACY_SIGNATURE,
                                            "route": route_template(request.scope)})
    if hmac_rejection is None or (auth_mode == "token_or_hmac" and token_valid):
        return
    raise auth_rejected(request, _rejection_reason(auth_mode, bool(token), hmac_rejection))


def auth_header_rejection(
    request: Request,
    *,
    auth_mode: AuthMode,
    service_token: str,
    hmac_secrets: str,
    accept_v1_signature: bool,
) -> AuthRejection | None:
    """The refusal ``require_internal_auth`` reaches from the headers alone, or ``None`` when it needs more.

    The same decision on the checks it runs before the clock skew, the body and the replay cache (in their order,
    so the reason is the one it would log); a request that passes them is left to ``require_internal_auth``.
    """

    token = request.headers.get(TOKEN_HEADER, "")
    if auth_mode in {"token", "token_or_hmac"} and _token_is_valid(token, service_token):
        return None
    credentials = parse_hmac_secrets(hmac_secrets)
    hmac_rejection: AuthRejection = "not_configured"
    if auth_mode in {"hmac", "token_or_hmac"} and credentials:
        signed = _signed_request(request.headers, credentials, accept_v1_signature=accept_v1_signature)
        if not isinstance(signed, str):
            return None
        hmac_rejection = signed
    return _rejection_reason(auth_mode, bool(token), hmac_rejection)


class AuthHeaderPrecheckMiddleware:
    """Refuse a protected request whose headers alone fail authentication before anything reads its body.

    The route's auth dependency runs after DisconnectCancellationMiddleware has buffered the whole body (up to
    24 MiB), so any caller could make the service read that much without a valid key (security audit 2026-10-24,
    L4). Placed inside RequestContextMiddleware (request id, access log, metrics) and outside the body buffering,
    this answers the dependency's own refusal (same reason, log line, counter and 401 bytes) for a request to a
    route that depends on ``dependency``, and for any request under ``protected_prefixes`` that no route matches
    fully (an unknown path, a trailing slash, another method): the router would answer those with 404, 307 or 405,
    but the body was buffered first (review of wave 2, F3). Such a request that passes is marked
    ``UNROUTED_SCOPE_KEY`` so that nothing buffers its body. Everything else passes through; the dependency still
    authenticates what passes here (clock skew, signature, replay).
    """

    def __init__(
        self,
        app: Any,
        *,
        routes: Sequence[BaseRoute],
        dependency: Callable[..., Any],
        precheck: Callable[[Request], AuthRejection | None],
        protected_prefixes: tuple[str, ...] = (API_PATH_PREFIX,),
    ) -> None:
        self.app = app
        # The application's own route list (filled as routes register), searched like the router does.
        self.routes = routes
        self.dependency = dependency
        self.precheck = precheck
        self.protected_prefixes = protected_prefixes

    def _overridden(self, route: APIRoute) -> bool:
        # An override of the dependency (app.dependency_overrides) replaces its check, as it does in FastAPI.
        return self.dependency in getattr(route.dependency_overrides_provider, "dependency_overrides", {})

    def _match(self, scope: AsgiMessage) -> tuple[BaseRoute | None, bool]:
        """The route that serves the request (``None``: no route matches fully) and whether to check its headers."""
        for route in self.routes:
            match, _child_scope = route.matches(scope)
            if match is not Match.FULL:
                continue
            protected = (isinstance(route, APIRoute) and not self._overridden(route) and any(
                dependant.call is self.dependency for dependant in route.dependant.dependencies))
            return route, protected
        path = str(scope.get("path") or "")
        overridden = any(self._overridden(route) for route in self.routes if isinstance(route, APIRoute))
        return None, path.startswith(self.protected_prefixes) and not overridden

    async def __call__(self, scope: AsgiMessage, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http":
            route, protected = self._match(scope)
            if protected:
                request = Request(scope)
                reason = self.precheck(request)
                if reason is not None:
                    if route is not None:
                        # As if the router had matched it: the access log and the metrics name the route template.
                        scope["route"] = route
                    response = await app_error_handler(request, auth_rejected(request, reason))
                    await response(scope, receive, send)
                    return
            if route is None:
                scope[UNROUTED_SCOPE_KEY] = True
        await self.app(scope, receive, send)


__all__ = [
    "HMAC_KEY_ID_HEADER",
    "HMAC_REQUEST_ID_HEADER",
    "HMAC_SIGNATURE_HEADER",
    "HMAC_SIGNATURE_V2",
    "HMAC_SIGNATURE_VERSION_HEADER",
    "HMAC_TIMESTAMP_HEADER",
    "SECRET_PATTERNS",
    "AuthHeaderPrecheckMiddleware",
    "ReplayCache",
    "auth_header_rejection",
    "parse_hmac_secrets",
    "redact_secret_like_values",
    "require_configured_service_auth",
    "require_internal_auth",
    "sign_hmac_request",
    "sign_hmac_v2_request",
]
