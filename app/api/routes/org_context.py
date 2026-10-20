"""Organization context routes (ORG_CONTEXT spec §4): the profile distillation the backend's distill worker calls."""

from __future__ import annotations

from typing import Annotated

import asyncpg
from fastapi import Depends, FastAPI

from app.api.deps import get_db, require_internal_token
from app.schemas.org_context import OrgContextDistillResponse, RagOrgContextDistillRequest
from app.services.org_context import distill as distill_service


async def lesson_author_org_context_distill(
    request: RagOrgContextDistillRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> OrgContextDistillResponse:
    return await distill_service.org_context_distill(request, pool)


def register(app: FastAPI) -> None:
    """Declare these routes on ``app`` itself (see ``app.api.routes``)."""
    app.add_api_route(
        distill_service.DISTILL_ROUTE,
        lesson_author_org_context_distill,
        methods=["POST"],
        dependencies=[Depends(require_internal_token)],
    )
