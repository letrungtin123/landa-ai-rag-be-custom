from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.routes import chat as chat_routes
from app.api.routes import health as health_routes
from app.api.routes import kb as kb_routes
from app.api.routes import lesson_author_legacy as lesson_author_legacy_routes
from app.api.routes import orchestration_v2 as orchestration_v2_routes
from app.core.config import settings
from app.core.errors import AppError, app_error_handler, request_validation_error_handler, unhandled_error_handler
from app.core.lifespan import build_lifespan
from app.core.logging import configure_application_logging
from app.core.middleware import RequestBodyLimitMiddleware
from app.core.request_context import DisconnectCancellationMiddleware, RequestContextMiddleware
from app.schemas.common import AiUsage as AiUsage
from app.schemas.lesson_author import RagLessonAuthorCheckpointRequest as RagLessonAuthorCheckpointRequest
from app.schemas.orchestration_v2 import (
    RagLessonAuthorCourseSkeletonV2Request as RagLessonAuthorCourseSkeletonV2Request,
)
from app.services import runtime as service_runtime
from app.services.lesson_author.architecture_validation import (
    validate_v5_instructional_coherence as validate_v5_instructional_coherence,
)
from app.services.lesson_author.checkpoint import (
    build_lesson_author_checkpoint_result as build_lesson_author_checkpoint_result,
)
from app.services.lesson_author.proposal_validation import (
    semantic_learning_visible_text as semantic_learning_visible_text,
)
from app.services.lesson_author.staged.provider_schemas import (
    staged_component_payload_code as staged_component_payload_code,
)
from app.services.meta import API_VERSION
from app.services.provider import generate_content as generate_content

app = FastAPI(
    title="Internal AI RAG Service",
    version=API_VERSION,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None,
    # Resolved at call time: startup/shutdown are defined further down.
    lifespan=build_lifespan(
        state=service_runtime.runtime_state,
        startup=lambda: service_runtime.startup(),
        shutdown=lambda: service_runtime.shutdown(),
        grace_seconds=settings.shutdown_grace_seconds,
    ),
)


# add_middleware wraps outward: request context (outermost) -> body limit ->
# disconnect cancellation (buffers the already size-checked body) -> routes.
app.add_middleware(DisconnectCancellationMiddleware)


app.add_middleware(
    RequestBodyLimitMiddleware,
    max_request_bytes=settings.max_request_bytes,
    idm_max_request_bytes=settings.idm_max_request_bytes,
)


app.add_middleware(RequestContextMiddleware, state=service_runtime.runtime_state)


app.add_exception_handler(AppError, app_error_handler)


app.add_exception_handler(Exception, unhandled_error_handler)


app.add_exception_handler(RequestValidationError, request_validation_error_handler)


health_routes.register(app)


kb_routes.register(app)


chat_routes.register(app)


lesson_author_legacy_routes.register(app)
orchestration_v2_routes.register(app)


# Configure the package logger so every app.* module (app.idm, app.core.request_context, ...)
# emits through the JSON handler, not only this module.
configure_application_logging("app")


logger = logging.getLogger(__name__)
