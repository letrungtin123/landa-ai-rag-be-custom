from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from uuid import UUID

from fastapi import Request

from app.core.config import AuthMode
from app.core.errors import AppError
from app.core.logging import SECRET_PATTERNS, redact_secret_like_values

HMAC_KEY_ID_HEADER = "X-Landa-Key-Id"
HMAC_TIMESTAMP_HEADER = "X-Landa-Timestamp"
HMAC_SIGNATURE_HEADER = "X-Landa-Signature"
HMAC_REQUEST_ID_HEADER = "X-Landa-Request-Id"
TOKEN_HEADER = "X-Landa-AI-Service-Token"  # noqa: S105 - HTTP header name, not a token.
HEX_SIGNATURE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MINIMUM_HMAC_SECRET_LENGTH = 16


@dataclass(frozen=True, slots=True)
class HmacCredential:
    key_id: str
    secret: str


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


async def _hmac_is_valid(
    request: Request,
    credentials: dict[str, HmacCredential],
    *,
    clock_skew_seconds: int,
    replay_cache: ReplayCache,
) -> bool:
    key_id = request.headers.get(HMAC_KEY_ID_HEADER, "")
    timestamp = request.headers.get(HMAC_TIMESTAMP_HEADER, "")
    signature = request.headers.get(HMAC_SIGNATURE_HEADER, "").lower()
    request_id = request.headers.get(HMAC_REQUEST_ID_HEADER, "")
    credential = credentials.get(key_id)
    if credential is None or not timestamp.isdigit() or not HEX_SIGNATURE_PATTERN.fullmatch(signature):
        return False
    try:
        UUID(request_id)
    except (ValueError, AttributeError):
        return False
    current = int(time.time())
    if abs(current - int(timestamp)) > clock_skew_seconds:
        return False
    body = await request.body()
    expected = sign_hmac_request(
        credential.secret,
        timestamp,
        request.method,
        request.url.path,
        body,
    )
    if not hmac.compare_digest(signature.encode(), expected.encode()):
        return False
    return replay_cache.consume(request_id, now=float(current))


async def require_internal_auth(
    request: Request,
    *,
    auth_mode: AuthMode,
    service_token: str,
    hmac_secrets: str,
    clock_skew_seconds: int,
    replay_ttl_seconds: int,
    replay_cache: ReplayCache | None = None,
) -> None:
    token_valid = _token_is_valid(request.headers.get(TOKEN_HEADER, ""), service_token)
    if auth_mode == "token" and token_valid:
        return

    credentials = parse_hmac_secrets(hmac_secrets)
    active_cache = replay_cache or _replay_cache
    active_cache.ttl_seconds = replay_ttl_seconds
    hmac_valid = False
    if auth_mode in {"hmac", "token_or_hmac"} and credentials:
        hmac_valid = await _hmac_is_valid(
            request,
            credentials,
            clock_skew_seconds=clock_skew_seconds,
            replay_cache=active_cache,
        )
    if hmac_valid or (auth_mode == "token_or_hmac" and token_valid):
        return
    raise AppError("UNAUTHORIZED", 401, "Unauthorized AI RAG service request.")


__all__ = [
    "HMAC_KEY_ID_HEADER",
    "HMAC_REQUEST_ID_HEADER",
    "HMAC_SIGNATURE_HEADER",
    "HMAC_TIMESTAMP_HEADER",
    "SECRET_PATTERNS",
    "ReplayCache",
    "parse_hmac_secrets",
    "redact_secret_like_values",
    "require_configured_service_auth",
    "require_internal_auth",
    "sign_hmac_request",
]
