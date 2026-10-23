"""Response models of the liveness and service-identity routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str


class InFlightRequests(BaseModel):
    """Requests being served (``/v1/meta`` itself and the health and metrics probes are not counted)."""

    model_config = ConfigDict(extra="forbid")

    total: int
    # ai_id, chat, knowledge_base, lesson_author_legacy and other, always all five.
    by_route_class: dict[str, int]


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
    # sha256 over the IDM prompt template sources (app.idm.prompt_manifest); changes with every prompt
    # edit, while contracts.idm_prompt_policy_version stays the contract value.
    prompt_templates_hash: str
    # A deploy restarts the process once total is 0, so no AI ID run is cut (audit 2026-10-24, O1).
    in_flight: InFlightRequests
