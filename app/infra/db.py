"""Postgres pool configuration and a connect loop that never crash-loops the process (SEP-1 #1).

The AI service may run on another host than Postgres. Everything that shapes the connection
(pool size, TLS, statement cache for poolers, timeouts, server-side TCP keepalives) is a setting,
and an unreachable database at startup leaves the process serving (``/readyz`` = 503) while a
background task reconnects with capped exponential backoff.

Logs carry only event names, attempt counters, error *types* and SQLSTATEs: never the DSN, the
host, the user or a driver message (they can echo credentials or topology).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import ssl
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

import asyncpg  # type: ignore[import-untyped]

from app.infra.net import is_loopback_host

logger = logging.getLogger(__name__)

SslMode = Literal["disable", "prefer", "require", "verify-ca", "verify-full"]
SSL_MODES: tuple[SslMode, ...] = ("disable", "prefer", "require", "verify-ca", "verify-full")
APPLICATION_NAME = "landa-ai-rag"
# Probe interval/count after the idle time: a dead peer is detected within idle + 6 x 10 s.
TCP_KEEPALIVES_INTERVAL_SECONDS = 10
TCP_KEEPALIVES_COUNT = 6
# Pool connections idle longer than this are closed (asyncpg default 300 s).
MAX_INACTIVE_CONNECTION_LIFETIME_SECONDS = 300.0
CONNECT_RETRY_INITIAL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    dsn: str
    production: bool
    pool_min: int
    pool_max: int
    ssl_mode: SslMode | None
    ssl_root_cert: str
    statement_cache_size: int
    connect_timeout_seconds: float
    command_timeout_seconds: float
    tcp_keepalives_idle_seconds: int


def _dsn_query(dsn: str) -> dict[str, str]:
    try:
        query = urlsplit(dsn).query
    except ValueError:
        return {}
    return {key: values[-1] for key, values in parse_qs(query).items() if values}


def _dsn_host(dsn: str) -> str:
    try:
        return (urlsplit(dsn).hostname or "").strip("[]").lower()
    except ValueError:
        return ""


def effective_ssl_mode(config: DatabaseConfig) -> SslMode:
    """Explicit setting > DSN ``sslmode`` > environment default."""
    if config.ssl_mode is not None:
        return config.ssl_mode
    from_dsn = _dsn_query(config.dsn).get("sslmode", "").strip().lower()
    if from_dsn in SSL_MODES:
        return from_dsn
    if from_dsn == "allow":  # libpq-only mode; closest asyncpg equivalent
        return "prefer"
    host = _dsn_host(config.dsn)
    if config.production and host and not is_loopback_host(host):
        return "require"
    # A loopback (or unset, i.e. local socket) host gains nothing from TLS and the co-located
    # Supabase Postgres may have it disabled.
    return "prefer"


def effective_ssl_root_cert(config: DatabaseConfig) -> str:
    return config.ssl_root_cert.strip() or _dsn_query(config.dsn).get("sslrootcert", "").strip()


def build_ssl_argument(mode: SslMode, root_cert: str) -> ssl.SSLContext | bool | str:
    """Translate a libpq sslmode into asyncpg's ``ssl`` argument.

    ``require`` encrypts without verifying, unless a root certificate is given; then, like
    libpq, the server certificate chain is verified (but not the host name).
    """
    if mode == "disable":
        return False
    if mode == "prefer":
        return "prefer"
    context = ssl.create_default_context(cafile=root_cert or None)
    if mode == "verify-full":
        return context
    context.check_hostname = False
    if mode == "require" and not root_cert:
        context.verify_mode = ssl.CERT_NONE
    return context


def server_settings(config: DatabaseConfig) -> dict[str, str]:
    values = {"application_name": APPLICATION_NAME}
    if config.tcp_keepalives_idle_seconds > 0:
        values.update({
            "tcp_keepalives_idle": str(config.tcp_keepalives_idle_seconds),
            "tcp_keepalives_interval": str(TCP_KEEPALIVES_INTERVAL_SECONDS),
            "tcp_keepalives_count": str(TCP_KEEPALIVES_COUNT),
        })
    return values


def pool_arguments(config: DatabaseConfig) -> dict[str, Any]:
    """Keyword arguments for ``asyncpg.create_pool`` (the explicit ``ssl`` overrides DSN sslmode)."""
    mode = effective_ssl_mode(config)
    return {
        "dsn": config.dsn,
        "min_size": config.pool_min,
        "max_size": config.pool_max,
        "command_timeout": config.command_timeout_seconds,
        "statement_cache_size": config.statement_cache_size,
        "timeout": config.connect_timeout_seconds,
        "ssl": build_ssl_argument(mode, effective_ssl_root_cert(config)),
        "server_settings": server_settings(config),
        "max_inactive_connection_lifetime": MAX_INACTIVE_CONNECTION_LIFETIME_SECONDS,
    }


def safe_error_fields(error: BaseException) -> dict[str, str]:
    """Error type and SQLSTATE only; the driver message may contain the user or host."""
    fields = {"error_type": type(error).__name__}
    sqlstate = getattr(error, "sqlstate", None)
    if isinstance(sqlstate, str) and sqlstate:
        fields["sqlstate"] = sqlstate[:5]
    return fields


def retry_delay_seconds(attempt: int, *, maximum: float, jitter: Callable[[float, float], float]) -> float:
    """Capped exponential backoff (1, 2, 4 ... maximum) with +-20 % jitter, never below 0.5 s."""
    exponent = max(0, min(attempt - 1, 16))
    base = min(maximum, CONNECT_RETRY_INITIAL_SECONDS * float(1 << exponent))
    return max(0.5, base * float(jitter(0.8, 1.2)))


ConnectState = Literal["idle", "connecting", "connected", "retrying", "closed"]


class DatabaseRuntime:
    """Owns the pool. ``start`` tries once inline and otherwise keeps retrying in the background."""

    def __init__(
        self,
        *,
        create_pool: Callable[[], Awaitable[Any]],
        on_connected: Callable[[Any], Awaitable[None]],
        retry_max_seconds: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._create_pool = create_pool
        self._on_connected = on_connected
        self._retry_max_seconds = retry_max_seconds
        self._sleep = sleep
        self._jitter = jitter
        self._task: asyncio.Task[None] | None = None
        self.pool: Any | None = None
        self.state: ConnectState = "idle"
        self.attempts = 0
        self._closed = False

    async def _attempt(self) -> bool:
        self.attempts += 1
        self.state = "connecting"
        try:
            pool = await self._create_pool()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.state = "retrying"
            logger.warning(
                "database_connect_failed",
                extra={"event": "database_connect_failed", "attempt": self.attempts, **safe_error_fields(error)},
            )
            return False
        self.pool = pool
        self.state = "connected"
        logger.info("database_connected", extra={"event": "database_connected", "attempt": self.attempts})
        try:
            await self._on_connected(pool)
        except Exception as error:  # the pool stays usable; readiness reports the callback's own state
            logger.warning(
                "database_post_connect_failed",
                extra={"event": "database_post_connect_failed", **safe_error_fields(error)},
            )
        return True

    async def _retry_loop(self) -> None:
        while not self._closed:
            delay = retry_delay_seconds(self.attempts, maximum=self._retry_max_seconds, jitter=self._jitter)
            logger.info(
                "database_connect_retry_scheduled",
                extra={"event": "database_connect_retry_scheduled", "attempt": self.attempts + 1,
                       "retry_in_s": round(delay, 2)},
            )
            await self._sleep(delay)
            if self._closed or await self._attempt():
                return

    async def start(self) -> None:
        self.state = "idle"
        self.attempts = 0
        self._closed = False
        if not await self._attempt():
            self._task = asyncio.create_task(self._retry_loop(), name="database-connect-retry")

    async def wait_retry(self) -> None:
        """Test/diagnostic helper: wait for a running background connect loop."""
        if self._task is not None:
            await self._task

    async def close(self) -> None:
        self._closed = True
        self.state = "closed"
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        pool, self.pool = self.pool, None
        if pool is not None:
            await pool.close()


def create_pool_factory(config: DatabaseConfig) -> Callable[[], Awaitable[Any]]:
    async def create() -> Any:
        return await asyncpg.create_pool(**pool_arguments(config))

    return create
