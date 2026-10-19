"""In-process Prometheus metrics for the AI RAG service.

Labels are deliberately low-cardinality: route templates (never raw paths),
fixed stage/workload names and error codes. Tenant identifiers, document IDs
and model output never become label values.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry(auto_describe=True)

# Pipeline stage of the provider calls made in the current context (``provider_stage``): IDM stages
# (idm_w1_map ... idm_w6_judge), course_skeleton, chapter_shard, chat, embeddings. Calls made
# outside a labelled stage count as ``unlabelled``; a label that is not a short snake_case token
# counts as ``other`` so the label set stays bounded.
UNLABELLED_STAGE = "unlabelled"
INVALID_STAGE = "other"
STAGE_LABEL_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_provider_stage: ContextVar[str] = ContextVar("ai_rag_provider_stage", default=UNLABELLED_STAGE)


def stage_label(stage: str | None) -> str:
    if stage is None:
        return UNLABELLED_STAGE
    return stage if STAGE_LABEL_PATTERN.fullmatch(stage) else INVALID_STAGE


@contextmanager
def provider_stage(stage: str | None) -> Iterator[None]:
    """Label the provider calls made inside this block (and the tasks it starts) with ``stage``."""

    token = _provider_stage.set(stage_label(stage))
    try:
        yield
    finally:
        _provider_stage.reset(token)


def current_provider_stage() -> str:
    return _provider_stage.get()


HTTP_REQUESTS = Counter(
    "ai_rag_http_requests_total",
    "HTTP requests handled, by route template, method and status code.",
    ("route", "method", "status"),
    registry=REGISTRY,
)
HTTP_REQUEST_DURATION = Histogram(
    "ai_rag_http_request_duration_seconds",
    "HTTP request duration by route template and method.",
    ("route", "method"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
    registry=REGISTRY,
)
HTTP_CLIENT_DISCONNECTS = Counter(
    "ai_rag_http_client_disconnects_total",
    "Requests cancelled because the caller disconnected before the response.",
    ("route",),
    registry=REGISTRY,
)
PROVIDER_CALLS = Counter(
    "ai_rag_provider_calls_total",
    "Provider transport calls by operation, outcome and pipeline stage.",
    ("operation", "outcome", "stage"),
    registry=REGISTRY,
)
PROVIDER_CALL_DURATION = Histogram(
    "ai_rag_provider_call_duration_seconds",
    "Provider transport call duration by operation and pipeline stage.",
    ("operation", "stage"),
    buckets=(0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 180, 300),
    registry=REGISTRY,
)
PROVIDER_TOKENS = Counter(
    "ai_rag_provider_tokens_total",
    "Provider-reported or estimated tokens by operation, token kind and pipeline stage.",
    ("operation", "kind", "stage"),
    registry=REGISTRY,
)
HANDLED_EXCEPTIONS = Counter(
    "ai_rag_handled_exceptions_total",
    "Exceptions a broad handler turned into a fallback or a safe error code, by stage, exception type and outcome.",
    ("stage", "exception_type", "outcome"),
    registry=REGISTRY,
)
FALLBACKS = Counter(
    "ai_rag_fallbacks_total",
    "Responses that used deterministic or structured fallback content, by stage.",
    ("stage",),
    registry=REGISTRY,
)
LIMITER_IN_USE = Gauge(
    "ai_rag_limiter_in_use",
    "Concurrency slots currently held, by workload.",
    ("workload",),
    registry=REGISTRY,
)
LIMITER_REJECTIONS = Counter(
    "ai_rag_limiter_rejections_total",
    "Requests rejected because a workload limiter stayed saturated.",
    ("workload",),
    registry=REGISTRY,
)
DOCUMENT_LIMIT_REJECTIONS = Counter(
    "ai_rag_document_limit_rejections_total",
    "Untrusted documents rejected by a processing limit, by error code.",
    ("code",),
    registry=REGISTRY,
)
DEADLINE_EXCEEDED = Counter(
    "ai_rag_deadline_exceeded_total",
    "Requests stopped by their overall deadline, by route template.",
    ("route",),
    registry=REGISTRY,
)
# Traffic of the legacy AI ID V1 routes (app.services.legacy_routes), watched before they are removed.
LEGACY_ROUTE_CALLS = Counter(
    "ai_rag_legacy_route_calls_total",
    "Calls of the legacy AI ID V1 routes and of the orchestration-v2 routes without IDM input, by route template.",
    ("route",),
    registry=REGISTRY,
)


def render_metrics() -> tuple[bytes, str]:
    """Return the Prometheus text exposition and its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
