from __future__ import annotations

import json
import logging
import re
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

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
STANDARD_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)
CAPTURE_GROUPS_WITH_SECRET = 2


def redact_secret_like_values(value: object) -> str:
    text = str(value)
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= CAPTURE_GROUPS_WITH_SECRET:
            text = pattern.sub(lambda match: f"{match.group(1)}{REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def _safe_value(key: str, value: Any) -> Any:
    if SENSITIVE_FIELD_PATTERN.search(key):
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
            "correlation_id": correlation_id_var.get(),
            "route": None,
            "duration_ms": None,
            "status": None,
        }
        for key, value in record.__dict__.items():
            if key not in STANDARD_LOG_RECORD_FIELDS and key not in payload:
                payload[key] = _safe_value(key, value)
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
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
