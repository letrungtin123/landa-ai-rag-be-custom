"""Process runtime: readiness state, concurrency limits, database pool and storage client lifecycle.

Other modules read the mutable resources through this module (``service_runtime.db_pool``) so startup,
shutdown and tests replace them in one place.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg
from supabase import create_client

from app.core.concurrency import ConcurrencyRuntime
from app.core.config import settings, validate_startup_settings
from app.core.lifespan import RuntimeState
from app.core.logging import SERVICE_LOGGER_NAME, log_event
from app.core.security import require_configured_service_auth
from app.idm.policy import IDM_PROMPT_POLICY_VERSION
from app.idm.prompt_manifest import prompt_templates_hash
from app.infra import db as db_infra
from app.infra import schema_check as schema_infra
from app.infra import storage as storage_infra
from app.infra.gemini import pool as gemini_pool
from app.infra.gemini import transport as gemini_transport

logger = logging.getLogger(SERVICE_LOGGER_NAME)


runtime_state = RuntimeState()


concurrency = ConcurrencyRuntime(
    provider_limit=settings.max_concurrent_provider_calls,
    index_limit=settings.max_concurrent_index_jobs,
    cpu_workers=settings.cpu_workers,
    acquire_timeout_seconds=settings.limiter_acquire_timeout_ms / 1000,
    extraction_executor=settings.extraction_executor,
)
# Provider calls take their slots from this runtime's limiter and run on its provider threads, both looked up at
# every attempt so a replaced ``concurrency`` (tests) is the one in use.
gemini_transport.use_provider_slots(lambda: concurrency.provider, lambda: concurrency.run_provider_call)


db_pool: asyncpg.Pool | None = None
supabase_client: Any | None = None


def require_settings() -> None:
    validate_startup_settings(settings)
    require_configured_service_auth(
        auth_mode=settings.auth_mode,
        service_token=settings.service_token.get_secret_value(),
        hmac_secrets=settings.service_hmac_secrets.get_secret_value(),
    )


SCHEMA_CHECK_TIMEOUT_SECONDS = 10.0
database: db_infra.DatabaseRuntime | None = None
schema_guard = schema_infra.SchemaGuard()


def database_config() -> db_infra.DatabaseConfig:
    return db_infra.DatabaseConfig(
        dsn=settings.database_url.get_secret_value(),
        production=settings.is_production,
        pool_min=settings.db_pool_min,
        pool_max=settings.db_pool_max,
        ssl_mode=settings.db_ssl_mode,
        ssl_root_cert=settings.db_ssl_root_cert,
        statement_cache_size=settings.db_statement_cache_size,
        connect_timeout_seconds=float(settings.db_connect_timeout_seconds),
        command_timeout_seconds=float(settings.database_command_timeout_seconds),
        tcp_keepalives_idle_seconds=settings.db_tcp_keepalives_idle_seconds,
    )


def storage_policy() -> storage_infra.StoragePolicy:
    return storage_infra.StoragePolicy(
        allowed_origins=storage_infra.parse_allowed_origins(settings.storage_allowed_origins, settings.supabase_url),
        bucket=settings.supabase_storage_bucket,
        max_bytes=settings.max_document_bytes,
        timeout_seconds=float(settings.storage_download_timeout_seconds),
        ca_file=settings.storage_ca_file,
    )


async def _on_database_connected(pool: Any) -> None:
    global db_pool  # noqa: PLW0603 - process-wide pool, swapped by the reconnect callback
    db_pool = pool
    await schema_guard.refresh(pool, timeout_seconds=SCHEMA_CHECK_TIMEOUT_SECONDS)


async def startup() -> None:
    global database, supabase_client  # noqa: PLW0603 - process resources created at startup
    require_settings()
    try:
        allowed_origins = storage_infra.parse_allowed_origins(settings.storage_allowed_origins, settings.supabase_url)
    except ValueError:
        raise RuntimeError("AI_RAG_STORAGE_ALLOWED_ORIGINS is invalid.") from None
    supabase_client = None
    service_key = settings.supabase_service_key.get_secret_value()
    if settings.supabase_url and service_key:
        # Legacy download path for index requests without a signed URL (removed once the
        # backend sends one everywhere and SUPABASE_SERVICE_KEY is dropped from this service).
        supabase_client = create_client(settings.supabase_url, service_key)
    logger.info(
        "storage_access_configured",
        extra={"event": "storage_access_configured", "legacy_service_key": supabase_client is not None,
               "allowed_origin_count": len(allowed_origins)},
    )
    log_event(logger, logging.INFO, "service_prompt_templates", prompt_policy_version=IDM_PROMPT_POLICY_VERSION,
              prompt_templates_hash=prompt_templates_hash())
    config = database_config()
    logger.info(
        "database_configured",
        extra={"event": "database_configured", "ssl_mode": db_infra.effective_ssl_mode(config),
               "pool_min": config.pool_min, "pool_max": config.pool_max,
               "statement_cache_size": config.statement_cache_size},
    )
    schema_guard.reset()
    # Never crash-loops: an unreachable database leaves /readyz at 503 and retries in the background.
    database = db_infra.DatabaseRuntime(
        create_pool=db_infra.create_pool_factory(config),
        on_connected=_on_database_connected,
        retry_max_seconds=float(settings.db_connect_retry_max_seconds),
    )
    await database.start()


async def shutdown() -> None:
    global db_pool, database  # noqa: PLW0603 - process resources released at shutdown
    runtime, database = database, None
    if runtime is not None:
        await runtime.close()
    elif db_pool is not None:
        await db_pool.close()
    db_pool = None
    gemini_pool.client_pool.clear()
    concurrency.shutdown()
