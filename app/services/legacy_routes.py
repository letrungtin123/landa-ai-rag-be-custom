"""Usage of the legacy AI ID V1 routes, so they can be removed once their traffic stops (audit F-05, roadmap 3.1).

Legacy means the lesson-author routes ``/proposal``, ``/blueprint`` and ``/chapter-checkpoint``, and the
orchestration-v2 generation routes when the request carries no IDM input (no ``idm`` on the course skeleton, no
``idm_module_context`` on the chapter shard, no ``idm_unit_brief`` in the unit contract). Each call adds one to
``ai_rag_legacy_route_calls_total{route}`` and logs one ``legacy_route_called`` event; the route template is the
only label and the only field, so neither carries tenant data or content.
"""

from __future__ import annotations

import logging
from typing import Final

from app.core.logging import SERVICE_LOGGER_NAME, log_event
from app.core.metrics import LEGACY_ROUTE_CALLS

logger = logging.getLogger(SERVICE_LOGGER_NAME)

PROPOSAL_ROUTE: Final = "/v1/lesson-author/proposal"
BLUEPRINT_ROUTE: Final = "/v1/lesson-author/blueprint"
CHAPTER_CHECKPOINT_ROUTE: Final = "/v1/lesson-author/chapter-checkpoint"
COURSE_SKELETON_ROUTE: Final = "/v1/lesson-author/orchestration-v2/course-skeleton"
CHAPTER_SHARD_ROUTE: Final = "/v1/lesson-author/orchestration-v2/chapter-shard"
UNIT_ROUTE: Final = "/v1/lesson-author/orchestration-v2/unit"
# The label values: a closed set, so the counter stays bounded.
LEGACY_ROUTES: Final = frozenset({
    PROPOSAL_ROUTE, BLUEPRINT_ROUTE, CHAPTER_CHECKPOINT_ROUTE, COURSE_SKELETON_ROUTE, CHAPTER_SHARD_ROUTE, UNIT_ROUTE,
})


def record_legacy_route_call(route: str) -> None:
    if route not in LEGACY_ROUTES:
        msg = f"not a legacy route: {route}"
        raise ValueError(msg)
    LEGACY_ROUTE_CALLS.labels(route=route).inc()
    log_event(logger, logging.INFO, "legacy_route_called", route=route)
