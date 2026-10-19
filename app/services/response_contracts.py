"""Observe whether the IDM generation answers fit their published response models, without ever changing them.

The ``/course-skeleton``, ``/chapter-shard`` and ``/unit`` routes answer the service dict exactly as before the typed
contracts (FastAPI's ``dict[str, Any]`` encoder, so the wire bytes are unchanged). Before the dict is returned,
the strict model of each route checks it without changing it. A payload that would not validate is only counted
(``ai_rag_response_contract_mismatch_total{route}``) and logged as one ``response_contract_mismatch`` event with
the route and the failing locations (field paths, never values); the check never raises (an unexpected error of the
check itself is counted the same way and reported as a handled exception).
"""

from __future__ import annotations

import logging
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from app.core.handled_exceptions import report_handled_exception
from app.core.logging import SERVICE_LOGGER_NAME, log_event
from app.core.metrics import RESPONSE_CONTRACT_MISMATCHES
from app.services.legacy_routes import CHAPTER_SHARD_ROUTE, COURSE_SKELETON_ROUTE, UNIT_ROUTE

logger = logging.getLogger(SERVICE_LOGGER_NAME)

# The label values: a closed set, so the counter stays bounded.
OBSERVED_ROUTES: Final = frozenset({COURSE_SKELETON_ROUTE, CHAPTER_SHARD_ROUTE, UNIT_ROUTE})
OTHER_ROUTE: Final = "other"
MAX_LOGGED_LOCATIONS: Final = 10
MAX_LOCATION_CHARS: Final = 200


def _location(loc: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in loc)[:MAX_LOCATION_CHARS]


def observe_response_contract(route: str, model: type[BaseModel], payload: Any) -> None:
    """Validate ``payload`` against ``model`` for observation only; ``payload`` is not changed."""

    label = route if route in OBSERVED_ROUTES else OTHER_ROUTE
    try:
        model.model_validate(payload)
    except ValidationError as error:
        details = error.errors(include_url=False, include_context=False, include_input=False)
        RESPONSE_CONTRACT_MISMATCHES.labels(route=label).inc()
        log_event(logger, logging.WARNING, "response_contract_mismatch", route=label, error_count=len(details),
                  locations=[_location(item["loc"]) for item in details[:MAX_LOGGED_LOCATIONS]])
    except Exception as error:  # observation only: a model bug must never touch a sent answer
        RESPONSE_CONTRACT_MISMATCHES.labels(route=label).inc()
        report_handled_exception(logger, error, stage="response_contract", outcome="RESPONSE_CONTRACT_CHECK_FAILED",
                                 route=label)
