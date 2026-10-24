"""Document extraction in a fresh child process per document (``AI_RAG_EXTRACTION_EXECUTOR=isolated``, audit M2).

With the ``thread`` executor a hostile or broken file can keep a thread forever (the deadline only stops the
await), crash the whole service (MuPDF is C code) or take its memory. Here each document is parsed by
``app.services.ingestion.extract_worker`` in its own interpreter (``ConcurrencyRuntime.run_in_child``): killed at the
caller's deadline, capped at ``AI_RAG_EXTRACTION_MEMORY_LIMIT_BYTES``, and its crash is only that document's error.
The sections are the ones ``extract_sections`` returns in the service, value for value.

Failures keep the codes Node already maps: the index deadline still ends as ``INDEX_DEADLINE_EXCEEDED`` (the child
is killed by the cancelled await), the memory limit and an oversized reply are ``DOCUMENT_LIMIT_EXCEEDED``
(``DocumentLimitError``), a crash or an unreadable reply is ``INDEX_DOCUMENT_FAILED`` (``IsolatedExtractionError``),
and the parsers' own errors come back as the same ``DocumentLimitError`` / ``ValueError`` they raise in the service.
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Final

from pydantic import SecretStr

from app.core.child_process import ChildOutcome
from app.core.concurrency import ConcurrencyRuntime
from app.core.config import settings
from app.core.document_limits import UnsupportedDocumentTypeError
from app.core.error_codes import APP_ERROR_CODES
from app.core.errors import AppError, DocumentLimitError
from app.core.logging import log_event
from app.services.ingestion import extract as extraction
from app.services.ingestion.extract import ExtractedSection

logger = logging.getLogger(__name__)

WORKER_MODULE = "app.services.ingestion.extract_worker"
# The child takes its settings from the request: "production" keeps it from reading a local .env file (and its
# secrets) when it loads app.core.config.
WORKER_ENVIRONMENT: Final = {"AI_RAG_ENV": "production"}
MEMORY_LIMIT_MESSAGE: Final = "Document extraction exceeded the memory limit."
# A reply holds the text (at most AI_RAG_MAX_EXTRACTED_CHARS, up to 4 UTF-8 bytes each) and its metadata, which
# repeats some of it (headings, the visual prompt): anything past this bound is not an honest reply.
_REPLY_BYTES_PER_CHAR = 8
_REPLY_OVERHEAD_BYTES = 64 * 1024 * 1024
_MAX_MESSAGE_CHARS = 4_000
_TYPE_LABEL_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
UNKNOWN_CHILD_ERROR = "OtherException"


class IsolatedExtractionError(RuntimeError):
    """The child ended without a usable reply, or with a parser error the service has no class for.

    ``error_type`` names what happened in the child (the parser's exception type, ``ChildProcessCrashed`` or
    ``ChildReplyInvalid``) for the ``rag_index_failed`` log line.
    """

    def __init__(self, error_type: str) -> None:
        super().__init__("Document extraction failed in its child process.")
        self.error_type = error_type


class _PlainValues(pickle.Unpickler):
    """Reads tuples, lists, dicts, sets, str, bytes, numbers, bool and None; any class reference is refused, so a
    reply cannot make the service import or call anything."""

    def find_class(self, module: str, name: str, /) -> Any:
        raise pickle.UnpicklingError("The extraction reply may only hold plain values.")


def extraction_request(path: Path, file_name: str) -> bytes:
    """The worker's request: the document and every setting of the service except its secrets."""

    values = {name: value for name, value in settings.model_dump().items() if not isinstance(value, SecretStr)}
    return json.dumps({"path": str(path.resolve()), "file_name": file_name, "settings": values}).encode("utf-8")


def max_reply_bytes() -> int:
    return settings.max_extracted_chars * _REPLY_BYTES_PER_CHAR + _REPLY_OVERHEAD_BYTES


async def extract_document(concurrency: ConcurrencyRuntime, path: Path) -> list[ExtractedSection]:
    """``extract_sections(path, path.name)`` on the configured extraction executor."""

    if concurrency.extraction_executor_kind == "isolated":
        outcome = await concurrency.run_in_child(
            WORKER_MODULE, extraction_request(path, path.name),
            memory_limit_bytes=settings.extraction_memory_limit_bytes, max_output_bytes=max_reply_bytes(),
            environment=dict(WORKER_ENVIRONMENT),
        )
        return sections_from_outcome(outcome)
    return await concurrency.run_extraction(extraction.extract_sections, path, path.name)


def _child_failed(outcome: ChildOutcome, reason: str) -> None:
    log_event(logger, logging.WARNING, "document_extraction_child_failed", reason=reason,
              exit_code=outcome.returncode, killed=outcome.killed)


def sections_from_outcome(outcome: ChildOutcome) -> list[ExtractedSection]:
    """The sections of a finished child, or the exception the service raises for its failure."""

    if outcome.memory_limit_reached:
        # Whatever the child answered after an allocation failed at the cap (MemoryError, a parser error, a crash).
        _child_failed(outcome, "memory_limit")
        raise DocumentLimitError(MEMORY_LIMIT_MESSAGE)
    if outcome.output_too_large:
        _child_failed(outcome, "reply_too_large")
        raise DocumentLimitError()
    if outcome.returncode != 0 or outcome.killed:
        _child_failed(outcome, "crashed")
        raise IsolatedExtractionError("ChildProcessCrashed")
    try:
        reply = _PlainValues(io.BytesIO(outcome.output)).load()
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError, IndexError, KeyError, MemoryError):
        reply = None
    sections = _reply_sections(reply)
    if sections is None:
        _child_failed(outcome, "reply_invalid")
        raise IsolatedExtractionError("ChildReplyInvalid")
    return sections


def _reply_sections(reply: Any) -> list[ExtractedSection] | None:
    """The sections of an ``ok`` reply; the parser's error of an error reply is raised; None for a malformed one."""

    if not isinstance(reply, tuple) or not reply:
        return None
    kind, rest = reply[0], reply[1:]
    if kind == "ok" and len(rest) == 1:
        return _sections(rest[0])
    if kind == "memory" and not rest:
        raise DocumentLimitError(MEMORY_LIMIT_MESSAGE)
    if kind == "app_error" and len(rest) == 3:  # noqa: PLR2004 - code, status, message
        error = _app_error(*rest)
        if error is not None:
            raise error
    elif kind == "value_error" and len(rest) == 1 and isinstance(rest[0], str):
        raise ValueError(rest[0][:_MAX_MESSAGE_CHARS])
    elif kind == "error" and len(rest) == 1 and isinstance(rest[0], str):
        raise IsolatedExtractionError(rest[0] if _TYPE_LABEL_RE.fullmatch(rest[0]) else UNKNOWN_CHILD_ERROR)
    return None


def _app_error(code: Any, http_status: Any, message: Any) -> AppError | None:
    """The registered AppError the parser raised (its class for the codes the service tests with isinstance)."""

    registered = APP_ERROR_CODES.get(code) if isinstance(code, str) else None
    if registered is None or http_status != registered.http_status or not isinstance(message, str):
        return None
    safe_message = message[:_MAX_MESSAGE_CHARS]
    if code == "DOCUMENT_LIMIT_EXCEEDED":
        return DocumentLimitError(safe_message)
    if code == "DOCUMENT_TYPE_UNSUPPORTED":
        return UnsupportedDocumentTypeError()
    return AppError(code=code, http_status=registered.http_status, safe_message=safe_message)


def _sections(value: Any) -> list[ExtractedSection] | None:
    if not isinstance(value, list):
        return None
    sections: list[ExtractedSection] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 4:  # noqa: PLR2004 - text, page, section, metadata
            return None
        text, page, section, metadata = item
        if (not isinstance(text, str) or not (page is None or type(page) is int)
                or not (section is None or isinstance(section, str)) or not isinstance(metadata, dict)):
            return None
        sections.append(ExtractedSection(text=text, page=page, section=section, metadata=metadata))
    # The child checked the limit; a reply past it is not one an honest child sends.
    if sum(len(section.text) for section in sections) > settings.max_extracted_chars:
        raise DocumentLimitError()
    return sections
