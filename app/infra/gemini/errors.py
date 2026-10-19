"""Classification of provider failures: HTTP status, content-free diagnostics, legacy retry codes.

The failures the gateway raises are ``app.core.errors.ProviderError`` with a code of
``app.core.error_codes``; this module reads them (and raw SDK errors) without exposing messages.
"""

from __future__ import annotations

import re
from typing import Any

from google.genai import errors as genai_errors

from app.core.errors import ServiceError

HTTP_BAD_REQUEST = 400
HTTP_STATUS_MIN = 400
HTTP_STATUS_MAX = 599


def provider_http_error_status(error: Exception) -> int | None:
    """SDK 1.0 APIError uses .code, not the HTTP wrapper's .status_code."""
    status = error.code if isinstance(error, genai_errors.APIError) else getattr(error, "status_code", None)
    return status if type(status) is int and HTTP_STATUS_MIN <= status <= HTTP_STATUS_MAX else None


def safe_provider_error_diagnostics(error: Exception) -> dict[str, Any]:
    """Classify locally; never emit exception messages, bodies, URLs or headers."""
    status = None if isinstance(error, ServiceError) else provider_http_error_status(error)
    details = getattr(error, "details", None)
    envelope = details if isinstance(details, dict) else {}
    error_body = envelope.get("error", envelope)
    error_body = error_body if isinstance(error_body, dict) else {}
    message_value = error_body.get("message")
    structured_details = error_body.get("details", [])
    generic_message = isinstance(message_value, str) and message_value.strip().casefold() in {
        "request contains an invalid argument.", "request contains an invalid argument", "invalid argument."
    }
    message = str(error).casefold()
    # Gemini may say "JSON schema"/"controlled generation" without naming
    # response_schema. Match diagnostic categories, never log the message.
    schema_error = status == HTTP_BAD_REQUEST and any(
        key in message for key in ("schema", "controlled generation", "constrained decoding"))
    compact_message = re.sub(r"[\s_]", "", message)
    markers = [code for code, needles in (
        ("MAX_ITEMS", ("maxitems", "maximumitems")), ("MIN_ITEMS", ("minitems", "minimumitems")),
        ("COMPLEXITY", ("toocomplex", "toomanystates", "nesting", "complexity")),
        ("UNSUPPORTED", ("unsupported", "notsupported", "unknownname")),
        ("POSITIVE_BOUND", ("greaterthan0", "greaterthanzero", "positiveinteger", "mustbepositive")),
        ("NULLABLE", ("nullable",)), ("ANY_OF", ("anyof",)), ("ONE_OF", ("oneof",)),
        ("MIN_LENGTH", ("minlength",)), ("MAX_LENGTH", ("maxlength",)),
        ("TOKEN_LIMIT", ("tokenlimit", "maxtokens", "maxoutputtokens")),
        ("THINKING_CONFIG", ("thinkingconfig", "includethoughts", "thinkingbudget")),
        ("ENUM", ("enum",)), ("PROPERTY_ORDERING", ("propertyordering",)),
        ("MODEL_UNSUPPORTED", ("modeldoesnotsupport", "modelisnotsupported")),
    ) if any(needle in compact_message for needle in needles)]
    constraint = "UNAVAILABLE"
    if schema_error:
        for code, constraint_markers in (
            ("MAX_ITEMS", ("max_items", "maxitems", "max items")),
            ("MIN_ITEMS", ("min_items", "minitems", "min items")),
            ("ARRAY_ITEMS", ("items",)),
            ("SCHEMA_COMPLEXITY", ("too complex", "too many states", "nesting")),
            ("UNSUPPORTED_FIELD", ("unknown name", "unsupported", "not supported")),
        ):
            if any(marker in message for marker in constraint_markers):
                constraint = code
                break
    provider_status = getattr(error, "status", None)
    known_statuses = {"INVALID_ARGUMENT", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED",
                      "PERMISSION_DENIED", "UNAUTHENTICATED", "NOT_FOUND", "FAILED_PRECONDITION"}
    # Type names are operational metadata, but never trust a dynamically created
    # exception type to be free of customer-controlled content.
    error_type = ServiceError.diagnostic_type if isinstance(error, ServiceError) else type(error).__name__
    known_types = {"ClientError", "ServerError", "APIError", "TimeoutError", "Timeout", "ReadTimeout", "ConnectTimeout",
                   "ConnectionError", "ConnectError", "RemoteProtocolError", "SSLError", "TypeError", "ValueError",
                   "ValidationError", "RuntimeError", "HTTPException", "AttributeError", "KeyError"}
    return {
        "provider_http_status": status,
        "provider_error_type": error_type if error_type in known_types else "OtherException",
        "provider_status": (provider_status if isinstance(provider_status, str) and provider_status in known_statuses
                            else "unavailable"),
        "provider_error_category": ("RESPONSE_SCHEMA_INVALID" if schema_error else "HTTP_ERROR" if status
                                    else "SDK_OR_TRANSPORT_ERROR"),
        "provider_schema_constraint": constraint,
        "provider_error_markers": markers,
        "provider_message_class": "GENERIC_INVALID_ARGUMENT" if generic_message else "REDACTED_OTHER",
        "provider_error_detail_count": len(structured_details) if isinstance(structured_details, list) else 0,
        "usage_source": "unavailable",
    }


NON_RETRYABLE_PROVIDER_ERROR_CODES = frozenset({
    "AI_PROVIDER_QUOTA_EXHAUSTED",
    # Classified out of AI_PROVIDER_QUOTA_EXHAUSTED; legacy flows keep treating both alike.
    "AI_PROVIDER_RATE_LIMITED",
    "AI_PROVIDER_UNAVAILABLE",
    "AI_PROVIDER_TIMEOUT",
    "AI_STAGED_LESSON_WORKFLOW_TIMEOUT",
})


def http_error_detail(error: ServiceError) -> dict[str, Any]:
    """The structured ``detail`` of a service error (``{"code", "message"}``), else ``{}``.

    Starlette types ``detail`` as ``str``; the service raises dict details, so it is read untyped.
    """
    detail: Any = error.detail
    return detail if isinstance(detail, dict) else {}


def is_non_retryable_provider_error(error: ServiceError) -> bool:
    detail = http_error_detail(error)
    return detail.get("code") in NON_RETRYABLE_PROVIDER_ERROR_CODES
