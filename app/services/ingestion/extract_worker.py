"""Child process of the isolated document extraction: ``python -m app.services.ingestion.extract_worker``.

``app.services.ingestion.extract_isolated`` starts one per document (``app.core.child_process``). It reads one JSON
request from stdin (``{"path", "file_name", "settings"}``: the service's settings without its secrets, so the limits
are the service's own), runs ``extract_sections`` and writes one reply to its stdout, a pickle of plain values only
(tuples, lists, dicts, str, int, float, bool, None) that the service reads without resolving any class:

- ``("ok", [(text, page, section, metadata), ...])``
- ``("app_error", code, http_status, safe_message)``: a processing limit (``DOCUMENT_LIMIT_EXCEEDED``)
- ``("value_error", message)``: an unreadable, unsupported or empty document (what the extractors raise)
- ``("memory",)``: an allocation failed at the memory limit
- ``("error", exception class name)``: any other failure of the parsers (for the service's log line)

What the libraries print goes to the null device: the reply is the only output on the pipe.
"""

from __future__ import annotations

import json
import os
import pickle
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from app.core.config import Settings, settings
from app.core.errors import AppError
from app.services.ingestion.extract import ExtractedSection, extract_sections

Extractor = Callable[[Path, str], list[ExtractedSection]]


def _contain_crashes() -> None:
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        # No error dialog for a crash or a missing drive (the job object does the same when the cap is in place).
        ctypes.WinDLL("kernel32").SetErrorMode(0x0001 | 0x0002 | 0x8000)
    elif sys.platform == "linux":
        import ctypes  # noqa: PLC0415 - Linux only

        # Killed with the service even when the service itself is killed (PR_SET_PDEATHSIG).
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)


def _reply_stream() -> IO[bytes]:
    """The pipe to the service, kept for the reply only; file descriptor 1 (``print``, C libraries) goes nowhere."""

    reply = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    return reply


def apply_settings(values: dict[str, Any]) -> None:
    for name, value in values.items():
        if name in Settings.model_fields:
            setattr(settings, name, value)


def extraction_reply(extract: Extractor, path: Path, file_name: str) -> tuple[Any, ...]:
    try:
        sections = extract(path, file_name)
        return "ok", [(section.text, section.page, section.section, section.metadata) for section in sections]
    except MemoryError:
        return ("memory",)
    except AppError as error:
        return "app_error", error.code, error.http_status, error.safe_message
    except ValueError as error:
        return "value_error", str(error)
    except Exception as error:  # every parser failure becomes a reply; the service maps it to its error code
        return "error", type(error).__name__


def encode_reply(reply: tuple[Any, ...]) -> bytes:
    try:
        return pickle.dumps(reply, protocol=pickle.HIGHEST_PROTOCOL)
    except MemoryError:
        return pickle.dumps(("memory",), protocol=pickle.HIGHEST_PROTOCOL)


def main(extract: Extractor = extract_sections, *, request_stream: IO[bytes] | None = None,
         reply_stream: IO[bytes] | None = None) -> int:
    """Serve one request (the streams are the process's own unless a test passes others)."""

    if reply_stream is None:
        _contain_crashes()
        reply_stream = _reply_stream()
    request = json.loads((request_stream or sys.stdin.buffer).read())
    apply_settings(request["settings"])
    reply_stream.write(encode_reply(extraction_reply(extract, Path(request["path"]), str(request["file_name"]))))
    reply_stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
