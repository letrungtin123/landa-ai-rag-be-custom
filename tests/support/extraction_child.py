"""Misbehaving extraction children for the isolated-executor tests: ``python -m tests.support.extraction_child``.

The real worker (``app.services.ingestion.extract_worker.main``) serves the request. A document whose first line is
``#child:<behaviour>`` makes the "parser" misbehave that way instead of extracting; any other document is extracted
for real. The child writes its process id next to the document (``<name>.pid``) first.
"""

from __future__ import annotations

import faulthandler
import os
import time
from pathlib import Path

from app.services.ingestion import extract_worker
from app.services.ingestion.extract import ExtractedSection, extract_sections

ALLOCATION_BYTES = 16 * 1024 * 1024


def _allocate_until_refused() -> None:
    held: list[bytearray] = []
    while True:
        held.append(bytearray(ALLOCATION_BYTES))


def misbehave(path: Path, file_name: str) -> list[ExtractedSection]:
    first_line = path.read_bytes()[:64].split(b"\n", 1)[0]
    behaviour = first_line.removeprefix(b"#child:").decode("ascii", "replace") if first_line[:7] == b"#child:" else ""
    path.with_suffix(".pid").write_text(str(os.getpid()), encoding="utf-8")
    if behaviour == "sleep":
        while True:
            time.sleep(60)
    if behaviour == "allocate":
        _allocate_until_refused()  # MemoryError at the limit: the worker answers "memory"
    if behaviour == "allocate_then_crash":
        try:
            _allocate_until_refused()
        except MemoryError:
            faulthandler._read_null()  # a parser in C that crashes when its allocation fails
    if behaviour == "exit":
        os._exit(70)
    if behaviour == "segfault":
        faulthandler._read_null()  # a real access violation, as a crash of the parser would be
    return extract_sections(path, file_name)


if __name__ == "__main__":
    raise SystemExit(extract_worker.main(misbehave))
