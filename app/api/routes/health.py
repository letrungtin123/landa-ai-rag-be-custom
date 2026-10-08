"""Liveness, readiness, service identity and metrics routes."""

from __future__ import annotations

import asyncio
import logging

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse, Response

from app.api.deps import require_internal_token
from app.core import metrics
from app.core.config import settings
from app.core.errors import error_payload
from app.core.logging import SERVICE_LOGGER_NAME
from app.repositories import health as health_repository
from app.schemas.health import HealthResponse, ServiceMetaResponse
from app.services import runtime as service_runtime
from app.services.meta import API_VERSION, BUILD_SHA_PATTERN, SERVICE_NAME, service_contract_versions
from app.services.runtime import SCHEMA_CHECK_TIMEOUT_SECONDS

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def healthz() -> HealthResponse:
    """Liveness only: the process is serving requests. Never touches dependencies."""
    return HealthResponse(status="ok")


async def readyz() -> JSONResponse:
    """Readiness: started, not draining, database reachable and the schema check passed."""
    pool = service_runtime.db_pool
    ready = service_runtime.runtime_state.started and not service_runtime.runtime_state.draining and pool is not None
    code = "NOT_READY"
    if ready and pool is not None:
        try:
            async with asyncio.timeout(settings.readiness_db_timeout_ms / 1000):
                await health_repository.ping(pool)
        except Exception:
            logger.warning("readiness_database_unavailable", extra={"event": "readiness_database_unavailable"})
            ready = False
    if ready and pool is not None:
        schema = await service_runtime.schema_guard.current(pool, timeout_seconds=SCHEMA_CHECK_TIMEOUT_SECONDS)
        if not schema.ok:
            ready = False
            code = schema.code or "SCHEMA_CHECK_FAILED"
    if not ready:
        return JSONResponse(status_code=503, content=error_payload(code, "The service is not ready."))
    return JSONResponse(content={"status": "ready"})


async def service_meta() -> ServiceMetaResponse:
    """Build and contract identity so the backend can detect a version skew between servers."""
    build_sha = settings.build_sha.strip()
    return ServiceMetaResponse.model_validate({
        "service": SERVICE_NAME,
        "build_sha": build_sha if BUILD_SHA_PATTERN.fullmatch(build_sha) else "unknown",
        "api_version": API_VERSION,
        "contracts": service_contract_versions(),
        "capabilities": {
            "index_source_download_url": True,
            "legacy_storage_service_key": service_runtime.supabase_client is not None,
        },
        "database": {"state": service_runtime.database.state if service_runtime.database is not None else "idle"},
        "schema_check": service_runtime.schema_guard.summary(),
    })


async def metrics_endpoint() -> Response:
    body, content_type = metrics.render_metrics()
    return Response(content=body, media_type=content_type)


def register(app: FastAPI) -> None:
    """Declare these routes on ``app`` itself (see ``app.api.routes``)."""
    app.add_api_route("/healthz", healthz, methods=["GET"])
    app.add_api_route("/readyz", readyz, methods=["GET"])
    app.add_api_route("/v1/meta", service_meta, methods=["GET"], dependencies=[Depends(require_internal_token)])
    app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], dependencies=[Depends(require_internal_token)])
