"""The provider gateway of the service layer: the Gemini package ``app.infra.gemini`` under its old path.

Callers use ``provider.generate_content`` / ``provider.embed_texts`` through this module so a single
patch target (``app.services.provider.generate_content``) fakes the provider everywhere. The code moved
to ``app.infra.gemini`` in refactor wave 2b (step 2.4); the helpers below are re-exported for the old
imports. Patch a helper or a retry tunable in its own module (``app.infra.gemini.retry``, ...): a patch
of this module only reaches the callers that look the name up here.
"""

from app.infra.gemini.calls import (
    DEFAULT_EMBEDDING_MODEL,
    LEGACY_EMBEDDING_MODEL_ALIASES,
    embed_text_batch,
    embed_texts,
    embedding_batch_size,
    generate_content,
    get_embedding_values,
    normalize_embedding_model,
    require_provider_api_key,
)
from app.infra.gemini.errors import (
    NON_RETRYABLE_PROVIDER_ERROR_CODES,
    http_error_detail,
    is_non_retryable_provider_error,
    provider_http_error_status,
    safe_provider_error_diagnostics,
)
from app.infra.gemini.retry import (
    PROVIDER_TRANSIENT_MAX_ATTEMPTS,
    provider_retry_delay_seconds,
    provider_retry_hint_seconds,
)
from app.infra.gemini.telemetry import (
    emit_safe_provider_telemetry,
    normalize_provider_finish_reason,
    provider_response_telemetry,
)
from app.infra.gemini.transport import call_provider_with_timeout
from app.infra.gemini.usage import combine_usage, estimate_tokens, normalize_usage, usage_from_google_response

# Installs the provider slots of the process runtime before any call goes out.
from app.services import runtime as _runtime  # noqa: F401

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "LEGACY_EMBEDDING_MODEL_ALIASES",
    "NON_RETRYABLE_PROVIDER_ERROR_CODES",
    "PROVIDER_TRANSIENT_MAX_ATTEMPTS",
    "call_provider_with_timeout",
    "combine_usage",
    "embed_text_batch",
    "embed_texts",
    "embedding_batch_size",
    "emit_safe_provider_telemetry",
    "estimate_tokens",
    "generate_content",
    "get_embedding_values",
    "http_error_detail",
    "is_non_retryable_provider_error",
    "normalize_embedding_model",
    "normalize_provider_finish_reason",
    "normalize_usage",
    "provider_http_error_status",
    "provider_response_telemetry",
    "provider_retry_delay_seconds",
    "provider_retry_hint_seconds",
    "require_provider_api_key",
    "safe_provider_error_diagnostics",
    "usage_from_google_response",
]
