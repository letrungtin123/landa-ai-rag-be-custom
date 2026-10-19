"""Retry budget of one provider call: bounded attempts, jittered backoff and server retry hints."""

from __future__ import annotations

import random
import re

from app.core.config import settings

PROVIDER_TRANSIENT_MAX_ATTEMPTS = settings.provider_max_attempts
PROVIDER_TRANSIENT_RETRY_DELAY_SECONDS = settings.provider_retry_base_ms / 1000
PROVIDER_RATE_LIMIT_MIN_WAIT_SECONDS = 1.0
PROVIDER_RETRY_HINT_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$")


def provider_retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff with +/-20% jitter, capped by the configured maximum."""
    base = PROVIDER_TRANSIENT_RETRY_DELAY_SECONDS * (2 ** max(0, attempt))
    jittered = base * random.uniform(0.8, 1.2)  # noqa: S311 - jitter, not security
    delay: float = max(0.0, min(jittered, settings.provider_retry_max_ms / 1000))
    return delay


def provider_retry_hint_seconds(error: Exception) -> float | None:
    """Read a server retry hint (google.rpc.RetryInfo or Retry-After) without logging content."""
    details = getattr(error, "details", None)
    envelope = details if isinstance(details, dict) else {}
    body = envelope.get("error", envelope)
    entries = body.get("details", []) if isinstance(body, dict) else []
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and str(entry.get("@type", "")).endswith("google.rpc.RetryInfo"):
            match = PROVIDER_RETRY_HINT_PATTERN.match(str(entry.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except Exception:
            value = None
        if isinstance(value, str):
            match = PROVIDER_RETRY_HINT_PATTERN.match(value)
            if match:
                return float(match.group(1))
    return None
