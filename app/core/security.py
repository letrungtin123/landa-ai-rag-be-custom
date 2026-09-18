from __future__ import annotations

import hmac
import re

from fastapi import HTTPException, Request


SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"AQ\.[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(api[_-]?key|token|secret|password)\s*[:=]\s*([^\s,;]+)", re.IGNORECASE),
]


def redact_secret_like_values(value: object) -> str:
    text = str(value)
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda match: f"{match.group(1)}=<redacted>", text)
        else:
            text = pattern.sub("<redacted>", text)
    return text


def require_configured_service_token(service_token: str, *, is_production: bool) -> None:
    if is_production and not service_token:
        raise RuntimeError("AI_RAG_SERVICE_TOKEN is required when NODE_ENV=production.")


async def require_internal_token(request: Request, service_token: str) -> None:
    if not service_token:
        return
    received = request.headers.get("X-Landa-AI-Service-Token", "")
    if not hmac.compare_digest(received.encode("utf-8"), service_token.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Unauthorized AI RAG service request.")
