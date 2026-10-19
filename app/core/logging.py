from __future__ import annotations

import json
import logging
import re
import traceback
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Logger name of the HTTP service code split out of the former ``app/main.py`` (PRD-2).
# Log pipelines, alerts and tests select service events by ``logger == "app.main"``, so
# every module that used the module-level ``logger`` of ``app.main`` keeps this name.
SERVICE_LOGGER_NAME = "app.main"

REDACTED = "<redacted>"
SENSITIVE_FIELD_PATTERN = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|token|x-landa-[a-z0-9-]+)",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"AQ\.[0-9A-Za-z_\-]{20,}"),
    re.compile(
        r"((?:api[_-]?key|authorization|password|secret|token|x-landa-[a-z0-9-]+)\s*[:=]\s*)"
        r"([^\s,;]+)",
        re.IGNORECASE,
    ),
)
# Token *counts* (input_tokens, max_output_tokens, token_allowance, ...) share the "token" word
# with credentials; a number under such a key is a count, never a secret.
TOKEN_COUNT_FIELD_PATTERN = re.compile(r"tokens|token_(?:allowance|budget|count)", re.IGNORECASE)
STANDARD_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)
# Keys the formatter owns or the LogRecord reserves; a ``log_event`` field with one of these names
# gets a trailing underscore instead of overwriting it (or failing ``makeRecord``).
RESERVED_EVENT_FIELDS = STANDARD_LOG_RECORD_FIELDS | {"message", "asctime", "ts", "level", "logger", "request_id"}
MAX_EXCEPTION_FRAMES = 20
# Exception classes from these modules are code-defined, so their names are safe, bounded labels.
KNOWN_EXCEPTION_MODULES = (
    "builtins", "app", "pydantic", "pydantic_core", "google.genai", "httpx", "httpcore", "asyncio",
    "json", "asyncpg", "concurrent.futures", "ssl", "socket",
)
UNKNOWN_EXCEPTION_TYPE = "OtherException"
# Service and provider errors were fastapi.HTTPException until refactor wave 2b (step 2.4), an unknown
# module, so OtherException. They keep that label, and every subclass with them, so the handled-exception
# series of the owner's review (2026-10-18 to about 2026-10-25) stays comparable. Remove this case after
# that review.
_SERIES_STABLE_EXCEPTION_TYPES = frozenset({("app.core.errors", "ServiceError"), ("app.core.errors", "ProviderError")})
_LIBRARY_PATH_MARKERS = ("/site-packages/", "/dist-packages/")
_STDLIB_PATH_PATTERN = re.compile(r"/lib/python3\.\d+/|/Lib/")
# Observability fields every JSON line carries; they default to null and are
# filled from the record's ``extra`` when the caller supplies them (for example
# the request middleware's ``http_request_completed`` event).
REQUEST_OUTCOME_FIELDS = frozenset({"route", "duration_ms", "status"})
CAPTURE_GROUPS_WITH_SECRET = 2


def redact_secret_like_values(value: object) -> str:
    text = str(value)
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= CAPTURE_GROUPS_WITH_SECRET:
            text = pattern.sub(lambda match: f"{match.group(1)}{REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def _is_numeric_tree(value: Any) -> bool:
    """True for a number or null, or a dict/list/tuple whose every leaf is one."""

    if isinstance(value, (int, float)) or value is None:
        return True
    if isinstance(value, dict):
        return all(_is_numeric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_numeric_tree(item) for item in value)
    return False


def _is_token_count(key: str, value: Any) -> bool:
    # A container is kept only when it holds nothing but numbers: ``tokens={"access": "<secret>"}``
    # must be redacted whole, since its inner keys need not look sensitive.
    return TOKEN_COUNT_FIELD_PATTERN.search(key) is not None and _is_numeric_tree(value)


def _safe_value(key: str, value: Any) -> Any:
    if SENSITIVE_FIELD_PATTERN.search(key) and not _is_token_count(key, value):
        return REDACTED
    if isinstance(value, str):
        return redact_secret_like_values(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(item_key): _safe_value(str(item_key), item_value) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(key, item) for item in value]
    return redact_secret_like_values(value)


class SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secret_like_values(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: _safe_value(str(key), value) for key, value in record.args.items()
                }
            else:
                record.args = tuple(_safe_value("argument", value) for value in record.args)
        for key, value in list(record.__dict__.items()):
            if key not in STANDARD_LOG_RECORD_FIELDS:
                record.__dict__[key] = _safe_value(key, value)
        return True


def exception_type_label(error: BaseException) -> str:
    """A bounded, content-free name for ``error``'s class (metric label and log field).

    Only classes defined by the service, its tests, the standard library or the known client
    libraries are named; a dynamically created or third-party class is ``OtherException``.
    """

    cls = type(error)
    if any((base.__module__, base.__qualname__) in _SERIES_STABLE_EXCEPTION_TYPES for base in cls.__mro__):
        return UNKNOWN_EXCEPTION_TYPE
    module = cls.__module__ or ""
    if any(module == known or module.startswith(known + ".") for known in KNOWN_EXCEPTION_MODULES):
        return cls.__name__
    return UNKNOWN_EXCEPTION_TYPE


def _frame_module(filename: str) -> str:
    """Dotted module of a traceback frame without the host's directories (``app.idm.storyboard``)."""

    path = filename.replace("\\", "/")
    library = next((marker for marker in _LIBRARY_PATH_MARKERS if marker in path), None)
    # The innermost of the service's own package directories (``/app/app/idm`` in the image).
    package = max(("/app/", "/tests/"), key=path.rfind) if "/app/" in path or "/tests/" in path else None
    stdlib = _STDLIB_PATH_PATTERN.split(path)
    if library is not None:
        path = path.rsplit(library, 1)[1]
    elif package is not None:
        path = package.strip("/") + "/" + path.rsplit(package, 1)[1]
    elif len(stdlib) > 1:
        path = stdlib[-1]
    else:
        path = path.rsplit("/", 1)[-1]
    return path.removesuffix(".py").replace("/", ".")


def exception_frames(error: BaseException, *, limit: int = MAX_EXCEPTION_FRAMES) -> list[str]:
    """Where ``error`` was raised: ``module:function:line`` per frame, innermost last.

    A sanitised traceback: never the exception message, source lines or local values, which can
    carry prompts, provider output or personal data.
    """

    return [f"{_frame_module(frame.filename)}:{frame.name}:{frame.lineno}"
            for frame in traceback.extract_tb(error.__traceback__)[-limit:]]


def exception_fields(error: BaseException) -> dict[str, Any]:
    """Log fields of a handled exception: bounded type label and sanitised frames, plus its cause's."""

    fields: dict[str, Any] = {"exception_type": exception_type_label(error),
                              "exception_frames": exception_frames(error)}
    cause = error.__cause__ or error.__context__
    if cause is not None:
        fields["exception_cause_type"] = exception_type_label(cause)
        fields["exception_cause_frames"] = exception_frames(cause)
    return fields


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    /,
    *,
    family: str | None = None,
    **fields: Any,
) -> None:
    """Emit one structured event: the message names the event, every field is a top-level JSON field.

    ``family`` keeps the prefix of the former ``"<family> {json}"`` lines (``lesson_author_idm``,
    ``lesson_author_orchestration_v2``) at the start of the message, so searches by prefix keep
    matching. Values go through the redaction filter; never pass prompts, source text or provider
    output.
    """

    extra: dict[str, Any] = {"event": event}
    for key, value in fields.items():
        extra[f"{key}_" if key in RESERVED_EVENT_FIELDS or key == "event" else key] = value
    logger.log(level, f"{family} {event}" if family else event, extra=extra)


def log_payload(logger: logging.Logger, level: int, family: str, payload: Mapping[str, Any]) -> None:
    """``log_event`` for a payload that names its own event (``{"event": ..., **fields}``)."""

    fields = dict(payload)
    log_event(logger, level, str(fields.pop("event")), family=family, **fields)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = redact_secret_like_values(record.getMessage())
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": _safe_value("event", getattr(record, "event", None) or message.split(" ", 1)[0]),
            "message": message,
            "request_id": request_id_var.get(),
            # Outside a request (background work) the event's own correlation id still identifies the run.
            "correlation_id": correlation_id_var.get() or _safe_value("correlation_id",
                                                                       getattr(record, "correlation_id", None)),
            "route": None,
            "duration_ms": None,
            "status": None,
        }
        for key, value in record.__dict__.items():
            if key in STANDARD_LOG_RECORD_FIELDS:
                continue
            if key not in payload or key in REQUEST_OUTCOME_FIELDS:
                payload[key] = _safe_value(key, value)
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
            if record.exc_info[1] is not None:
                payload["exception_frames"] = exception_frames(record.exc_info[1])
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def configure_application_logging(name: str = "app.main") -> logging.Logger:
    application_logger = logging.getLogger(name)
    application_logger.setLevel(logging.INFO)
    application_logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.addFilter(SecretRedactionFilter())
    handler.setFormatter(JsonFormatter())
    application_logger.addHandler(handler)
    application_logger.propagate = False
    return application_logger
