"""Route dependencies: internal authentication and the database pool."""

from __future__ import annotations

import asyncpg
from fastapi import Request

from app.core.config import settings
from app.core.errors import AppError
from app.core.security import AuthRejection, auth_header_rejection
from app.core.security import require_internal_auth as verify_internal_auth
from app.services import runtime as service_runtime


async def require_internal_token(request: Request) -> None:
    await verify_internal_auth(
        request,
        auth_mode=settings.auth_mode,
        service_token=settings.service_token.get_secret_value(),
        hmac_secrets=settings.service_hmac_secrets.get_secret_value(),
        clock_skew_seconds=settings.auth_clock_skew_seconds,
        replay_ttl_seconds=settings.auth_replay_ttl_seconds,
        accept_v1_signature=settings.hmac_accept_v1_signature,
    )


def internal_auth_header_rejection(request: Request) -> AuthRejection | None:
    """What ``require_internal_token`` refuses ``request`` for from its headers alone (AuthHeaderPrecheckMiddleware)."""
    return auth_header_rejection(
        request,
        auth_mode=settings.auth_mode,
        service_token=settings.service_token.get_secret_value(),
        hmac_secrets=settings.service_hmac_secrets.get_secret_value(),
        accept_v1_signature=settings.hmac_accept_v1_signature,
    )


async def get_db() -> asyncpg.Pool:
    if service_runtime.db_pool is None:
        # The pool is (re)connecting in the background; the backend's durable workers retry 503.
        raise AppError("DATABASE_NOT_READY", 503, "The database connection is not ready.")
    return service_runtime.db_pool
