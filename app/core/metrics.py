"""In-process Prometheus metrics for the AI RAG service.

Labels are deliberately low-cardinality: route templates (never raw paths),
fixed stage/workload names and error codes. Tenant identifiers, document IDs
and model output never become label values.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry(auto_describe=True)

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
    "Provider transport calls by operation and outcome.",
    ("operation", "outcome"),
    registry=REGISTRY,
)
PROVIDER_CALL_DURATION = Histogram(
    "ai_rag_provider_call_duration_seconds",
    "Provider transport call duration by operation.",
    ("operation",),
    buckets=(0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 180, 300),
    registry=REGISTRY,
)
PROVIDER_TOKENS = Counter(
    "ai_rag_provider_tokens_total",
    "Provider-reported or estimated tokens by operation and token kind.",
    ("operation", "kind"),
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


def render_metrics() -> tuple[bytes, str]:
    """Return the Prometheus text exposition and its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
