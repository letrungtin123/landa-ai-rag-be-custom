"""FastAPI application factory of the landa-ai-rag service.

``app.main:app`` is the ASGI entrypoint (``python -m app``, PM2, the Docker image). The factory
only wires the application: middleware, error handlers and route registration. Route handlers
live in ``app.api.routes``, business logic in ``app.services``, all SQL in ``app.repositories``
and the process resources (database pool, limits, readiness state) in ``app.services.runtime``.
"""

from __future__ import annotations

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
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorCheckpointRequest
from app.schemas.orchestration_v2 import RagLessonAuthorCourseSkeletonV2Request
from app.services import runtime as service_runtime
from app.services.lesson_author.architecture_validation import validate_v5_instructional_coherence
from app.services.lesson_author.checkpoint import build_lesson_author_checkpoint_result
from app.services.lesson_author.proposal_validation import semantic_learning_visible_text
from app.services.lesson_author.staged.provider_schemas import staged_component_payload_code
from app.services.meta import API_VERSION
from app.services.provider import generate_content

# Besides ``app`` and ``create_app``, these names are re-exported for the Node backend's cross-language
# tests, which run ``python -c "from app.main import <name>"`` (and patch ``app.main.generate_content``)
# against this checkout. Remove a name only after the backend test imports it from its new module.
__all__ = [
    "AiUsage",
    "RagLessonAuthorCheckpointRequest",
    "RagLessonAuthorCourseSkeletonV2Request",
    "app",
    "build_lesson_author_checkpoint_result",
    "create_app",
    "generate_content",
    "semantic_learning_visible_text",
    "staged_component_payload_code",
    "validate_v5_instructional_coherence",
]

# Registration order is the order of app.routes (and of the OpenAPI document in development).
ROUTE_MODULES = (health_routes, kb_routes, chat_routes, lesson_author_legacy_routes, orchestration_v2_routes)


def create_app() -> FastAPI:
    application = FastAPI(
        title="Internal AI RAG Service",
        version=API_VERSION,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        # The lambdas resolve service_runtime.startup/shutdown at call time (tests replace them).
        lifespan=build_lifespan(
            state=service_runtime.runtime_state,
            startup=lambda: service_runtime.startup(),  # noqa: PLW0108
            shutdown=lambda: service_runtime.shutdown(),  # noqa: PLW0108
            grace_seconds=settings.shutdown_grace_seconds,
        ),
    )
    # add_middleware wraps outward: request context (outermost) -> body limit ->
    # disconnect cancellation (buffers the already size-checked body) -> routes.
    application.add_middleware(DisconnectCancellationMiddleware)
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_request_bytes=settings.max_request_bytes,
        idm_max_request_bytes=settings.idm_max_request_bytes,
    )
    application.add_middleware(RequestContextMiddleware, state=service_runtime.runtime_state)
    # Starlette types exception handlers by the base Exception; these accept their subclass.
    application.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    application.add_exception_handler(Exception, unhandled_error_handler)
    application.add_exception_handler(
        RequestValidationError, request_validation_error_handler,  # type: ignore[arg-type]
    )
    for routes in ROUTE_MODULES:
        routes.register(application)
    return application


app = create_app()


# Configure the package logger so every app.* module (app.idm, app.core.request_context and the
# services that log as "app.main") emits through the JSON handler.
configure_application_logging("app")
