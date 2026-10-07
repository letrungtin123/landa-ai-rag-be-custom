from __future__ import annotations

import os
import signal
import subprocess
import zipfile
from collections.abc import Sequence
from pathlib import Path

from app.core.errors import DocumentLimitError

OOXML_SUFFIXES = frozenset({".docx", ".pptx", ".xlsx"})


def assert_document_size(size_bytes: int, *, maximum: int) -> None:
    if size_bytes < 0 or size_bytes > maximum:
        raise DocumentLimitError()


def assert_tenant_storage_path(storage_path: str, tenant_id: str) -> None:
    normalized = storage_path.replace("\\", "/").lstrip("/")
    if (
        not normalized.startswith(f"{tenant_id}/")
        or ".." in normalized.split("/")
        or "://" in normalized
    ):
        raise DocumentLimitError("Document storage ownership could not be verified.")


def validate_ooxml_archive(
    path: Path,
    *,
    max_uncompressed_bytes: int,
    max_entries: int,
    max_compression_ratio: float,
) -> None:
    if path.suffix.lower() not in OOXML_SUFFIXES:
        return
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                raise DocumentLimitError()
            total = 0
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
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise DocumentLimitError("Document archive is invalid or unsafe.") from error


def run_limited_subprocess(
    command: Sequence[str],
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[bytes]:
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
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
            os.kill(-process.pid, kill_signal)
        process.kill()
        process.wait()
        raise DocumentLimitError("Document conversion exceeded the configured timeout.") from error
    if process.returncode != 0:
        raise RuntimeError("Document conversion failed.")
    return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)
