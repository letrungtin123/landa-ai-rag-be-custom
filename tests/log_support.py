"""Read structured log events (``app.core.logging.log_event``) captured by ``assertLogs``.

Events carry their fields as record attributes, not inside the message, so tests read
``assertLogs(...).records``: ``logged_events`` returns each event's own fields the way the JSON
line carries them, ``log_text`` renders the records exactly as the production JSON formatter writes
them (for "never logged" assertions over everything that reaches the log).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

from app.core.logging import STANDARD_LOG_RECORD_FIELDS, JsonFormatter

_RECORD_ONLY_FIELDS = STANDARD_LOG_RECORD_FIELDS | {"message", "asctime"}


def event_fields(record: logging.LogRecord) -> dict[str, Any]:
    """``{"event": ..., **fields}`` of one record, JSON-normalised (tuples become lists)."""

    fields = {key: value for key, value in record.__dict__.items() if key not in _RECORD_ONLY_FIELDS}
    normalised: dict[str, Any] = json.loads(json.dumps(fields, ensure_ascii=False, default=str))
    return normalised


def logged_events(records: Iterable[logging.LogRecord], event: str | None = None) -> list[dict[str, Any]]:
    """The structured events among ``records`` (only ``event`` when given), in log order."""

    return [event_fields(record) for record in records
            if hasattr(record, "event") and (event is None or record.__dict__["event"] == event)]


def event_lines(records: Iterable[logging.LogRecord]) -> list[str]:
    """Each structured event as ``<message prefix> <JSON of its fields, sorted keys>``: the text of the
    former ``"lesson_author_idm {json}"`` lines, so substring checks read every field of the event."""

    return [f"{record.getMessage().split(' ', 1)[0]} {json.dumps(event_fields(record), sort_keys=True)}"
            for record in records if hasattr(record, "event")]


def log_text(records: Iterable[logging.LogRecord]) -> str:
    """The records as the service's JSON log lines (message and every field)."""

    formatter = JsonFormatter()
    return "\n".join(formatter.format(record) for record in records)


__all__ = ["event_fields", "event_lines", "log_text", "logged_events"]
