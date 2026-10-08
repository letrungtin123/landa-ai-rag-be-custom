"""Response models of the liveness and service-identity routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str


class ServiceMetaResponse(BaseModel):
    """``/v1/meta``: build and contract identity (field order is the wire order)."""

    model_config = ConfigDict(extra="forbid")

    service: str
    build_sha: str
    api_version: str
    contracts: dict[str, Any]
    capabilities: dict[str, bool]
    database: dict[str, str]
    schema_check: dict[str, Any]
