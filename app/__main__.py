"""Process entrypoint: ``python -m app``.

All bind/runtime options come from validated Settings so the same command
works under PM2, a container or a plain shell on any OS.
"""

from __future__ import annotations

import logging
from typing import Any

import uvicorn

from app.core.config import Settings, settings
from app.core.logging import JsonFormatter, SecretRedactionFilter

UVICORN_EVENT = "uvicorn_log"


class UvicornEventFilter(logging.Filter):
    """Names uvicorn's lines ``uvicorn_log`` (the formatter would take the message's first word, e.g. "Started")
    and drops ``color_message``, uvicorn's ANSI-coloured copy of the message."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "event", None):
            record.event = UVICORN_EVENT
        record.__dict__.pop("color_message", None)
        return True


def uvicorn_log_config() -> dict[str, Any]:
    """uvicorn's own loggers through the service's JSON formatter and secret filter (security audit 2026-10-24, L1).

    After the app has answered 500, Starlette re-raises the exception and ``uvicorn.error`` logs it; uvicorn's
    default formatter wrote the whole traceback with the exception's message (provider text, request values, a
    key in a URL) to the PM2 log. The JSON line carries the exception type and code frames only, never the message.
    """

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"redact_secrets": {"()": SecretRedactionFilter}, "uvicorn_event": {"()": UvicornEventFilter}},
        "formatters": {"json": {"()": JsonFormatter}},
        "handlers": {
            "default": {"class": "logging.StreamHandler", "formatter": "json",
                        "filters": ["uvicorn_event", "redact_secrets"], "stream": "ext://sys.stderr"},
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }


def uvicorn_options(value: Settings) -> dict[str, Any]:
    return {
        "host": value.host,
        "port": value.port,
        "workers": value.workers,
        "proxy_headers": value.proxy_headers,
        "forwarded_allow_ips": value.forwarded_allow_ips,
        "timeout_keep_alive": value.keep_alive_timeout_seconds,
        "timeout_graceful_shutdown": value.shutdown_grace_seconds,
        "log_level": "info",
        "log_config": uvicorn_log_config(),
        "access_log": False,
    }


def main() -> None:
    uvicorn.run("app.main:app", **uvicorn_options(settings))


if __name__ == "__main__":
    main()
