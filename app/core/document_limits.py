from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from app.core.errors import AppError, DocumentLimitError

OOXML_SUFFIXES = frozenset({".docx", ".pptx", ".xlsx"})
# An XML part: byte-order-mark bytes (UTF-8, UTF-16, UTF-32 in either order), white space and NUL bytes, then "<"
# (or nothing else in the first bytes). Pictures, media and fonts start with binary signatures.
_XML_SNIFF_BYTES = 64
_XML_START_RE = re.compile(rb"[\x00\t\n\r \xef\xbb\xbf\xfe\xff]*(?:<|\Z)")
# Suffixes the index route has an extractor for. Anything else is rejected before any work.
INDEX_DOCUMENT_SUFFIXES = frozenset({".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".xls", ".txt", ".md", ".csv"})


class UnsupportedDocumentTypeError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_TYPE_UNSUPPORTED",
            http_status=422,
            safe_message="The document type is not supported for indexing.",
        )


def assert_document_size(size_bytes: int, *, maximum: int) -> None:
    if size_bytes < 0 or size_bytes > maximum:
        raise DocumentLimitError()


def document_source_suffix(storage_path: str) -> str:
    """Return the extractor suffix of a stored document, or raise for an unsupported type.

    The storage path is built by the backend from the uploaded file name and always keeps the
    real extension; the display name may have none (KB articles), so it is not used for routing.
    """
    suffix = PurePosixPath(storage_path.replace("\\", "/")).suffix.lower()
    if suffix not in INDEX_DOCUMENT_SUFFIXES:
        raise UnsupportedDocumentTypeError()
    return suffix


def assert_tenant_storage_path(storage_path: str, tenant_id: str) -> None:
    normalized = storage_path.replace("\\", "/").lstrip("/")
    if (
        not normalized.startswith(f"{tenant_id}/")
        or ".." in normalized.split("/")
        or "://" in normalized
    ):
        raise DocumentLimitError("Document storage ownership could not be verified.")


def _starts_like_xml(archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> bool:
    """True when the part's first bytes are those of an XML document: "<" after byte-order-mark bytes, white space or
    NUL bytes (UTF-16/32), or nothing but those. The name and the declared content type do not count: a package can
    give an XML part any name and still have it parsed as XML."""

    try:
        with archive.open(entry) as stream:
            head = stream.read(_XML_SNIFF_BYTES)
    except (OSError, EOFError, RuntimeError, NotImplementedError, zipfile.BadZipFile, zlib.error):
        return True  # unreadable here, unreadable for the parser too: count it against the smaller budget
    return _XML_START_RE.match(head) is not None


def validate_ooxml_archive(
    path: Path,
    *,
    max_uncompressed_bytes: int,
    max_xml_bytes: int,
    max_entries: int,
    max_compression_ratio: float,
    suffix: str | None = None,
) -> None:
    """Zip-bomb and memory guard. ``suffix`` must be the one the extractor is routed by (defaults to the path's).

    ``max_uncompressed_bytes`` bounds every part; ``max_xml_bytes`` the XML parts, which the parsers hold as whole
    trees at about 15 times their size (media about twice).
    """
    effective_suffix = (path.suffix if suffix is None else suffix).lower()
    if effective_suffix not in OOXML_SUFFIXES:
        return
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                raise DocumentLimitError()
            total = 0
            xml_total = 0
            for entry in entries:
                total += entry.file_size
                if total > max_uncompressed_bytes:
                    raise DocumentLimitError()
                if entry.file_size <= 0:
                    continue
                if entry.compress_size <= 0:
                    raise DocumentLimitError()
                if entry.file_size / entry.compress_size > max_compression_ratio:
                    raise DocumentLimitError()
                if _starts_like_xml(archive, entry):
                    xml_total += entry.file_size
                    if xml_total > max_xml_bytes:
                        raise DocumentLimitError()
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise DocumentLimitError("Document archive is invalid or unsafe.") from error


def run_limited_subprocess(
    command: Sequence[str],
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[bytes]:
    # An if-statement on sys.platform (not os.name) so mypy type-checks it on Windows and Linux.
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(  # noqa: S603 - command is server-owned and contains no shell.
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        if os.name == "nt":
            subprocess.run(  # noqa: S603 - fixed Windows process-tree command.
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],  # noqa: S607 - OS utility.
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
            # start_new_session=True made the child a group leader: kill the whole group. A group
            # that already exited is fine; the timeout is still the error to report.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(-process.pid, kill_signal)
        process.kill()
        process.wait()
        raise DocumentLimitError("Document conversion exceeded the configured timeout.") from error
    if process.returncode != 0:
        raise RuntimeError("Document conversion failed.")
    return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)
