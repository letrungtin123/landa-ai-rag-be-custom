"""Isolated document extraction (audit 2026-10-24, M2): one fresh child process per document.

The children are real processes of this interpreter (``app.services.ingestion.extract_worker``, or the misbehaving
``tests.support.extraction_child``); nothing leaves the machine.
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import json
import os
import pickle
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic import SecretStr

from app.core import child_process, metrics
from app.core.child_process import ChildOutcome, child_environment
from app.core.concurrency import ConcurrencyRuntime
from app.core.config import settings
from app.core.errors import AppError, DocumentLimitError
from app.infra.gemini.usage import AiUsage
from app.services import runtime as runtime_service
from app.services.ingestion import extract as extraction
from app.services.ingestion import extract_isolated, extract_worker
from app.services.ingestion import index as index_service
from app.services.ingestion.extract import ExtractedSection
from app.services.ingestion.extract_isolated import MEMORY_LIMIT_MESSAGE, IsolatedExtractionError
from tests.unit.services.test_characterization_ingestion import (
    INDEX_ID,
    KB_ID,
    TENANT_ID,
    FakeDb,
    FakeEmbedder,
    document_row,
    index_request,
    write_docx,
    write_pdf,
    write_pptx,
    write_xlsx,
)

CHILD_MODULE = "tests.support.extraction_child"
STILL_ACTIVE = 259
TEST_MEMORY_LIMIT = 512 * 1024 * 1024


def process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def cpu_slots_in_use() -> float:
    return metrics.REGISTRY.get_sample_value("ai_rag_limiter_in_use", {"workload": "cpu"}) or 0.0


class IsolatedTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def runtime(self, cpu_workers: int = 1) -> ConcurrencyRuntime:
        runtime = ConcurrencyRuntime(provider_limit=1, index_limit=1, cpu_workers=cpu_workers,
                                     acquire_timeout_seconds=0.5, extraction_executor="isolated")
        self.addCleanup(runtime.shutdown)
        return runtime

    def behaviour(self, name: str) -> Path:
        path = self.dir / f"{name}.txt"
        path.write_bytes(f"#child:{name}\nbody".encode())
        return path

    def isolated(self, runtime: ConcurrencyRuntime, path: Path) -> Any:
        """The sections, or the exception the isolated extraction raised."""

        try:
            return asyncio.run(extract_isolated.extract_document(runtime, path))
        except Exception as error:
            return error


class IsolatedExtractionParityTests(IsolatedTestCase):
    def documents(self) -> list[Path]:
        writers = {"source.pdf": write_pdf, "guide.docx": write_docx, "deck.pptx": write_pptx,
                   "controls.xlsx": write_xlsx}
        for name, writer in writers.items():
            writer(self.dir / name)
        (self.dir / "notes.txt").write_text("Ghi chú an toàn\n\nĐoạn hai: kiểm soát rủi ro.", encoding="utf-8")
        (self.dir / "notes.md").write_text("# Title\n\nPlain markdown body", encoding="utf-8")
        (self.dir / "table.csv").write_text("level,owner\n1,team", encoding="utf-8")
        return sorted(path for path in self.dir.iterdir() if path.suffix != ".pid")

    def test_sections_are_those_of_the_thread_executor_for_every_format(self) -> None:
        runtime = self.runtime()
        for path in self.documents():
            with self.subTest(document=path.name):
                expected = extraction.extract_sections(path, path.name)
                got = self.isolated(runtime, path)
                self.assertEqual(got, expected)
                # Same values of the same types (a tuple is not a list, 1 is not 1.0): the reprs are equal too.
                self.assertEqual(repr(got), repr(expected))
                self.assertTrue(all(type(section) is ExtractedSection for section in got))

    def test_parser_errors_come_back_as_the_errors_the_thread_executor_raises(self) -> None:
        runtime = self.runtime()
        broken = self.dir / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4 not really a pdf")
        blank = self.dir / "blank.pdf"
        write_pdf(blank, pages=0)
        not_zip = self.dir / "fake.docx"
        not_zip.write_bytes(b"not a zip archive")
        pages = self.dir / "pages.pdf"
        write_pdf(pages, pages=3)
        for path in (blank, not_zip):
            with self.subTest(document=path.name):
                try:
                    extraction.extract_sections(path, path.name)
                except (ValueError, DocumentLimitError) as error:
                    expected: Exception = error
                got = self.isolated(runtime, path)
                self.assertIs(type(got), type(expected))
                self.assertEqual(str(got), str(expected))
        # The service's settings reach the child: the page limit set here applies there.
        with patch.object(settings, "max_document_pages", 2):
            self.assertIsInstance(self.isolated(runtime, pages), DocumentLimitError)
        self.assertEqual(len(self.isolated(runtime, pages)), 3)
        crashed = self.isolated(runtime, broken)
        self.assertIsInstance(crashed, IsolatedExtractionError)
        self.assertEqual(crashed.error_type, "FileDataError")


class IsolatedExtractionContainmentTests(IsolatedTestCase):
    def test_a_child_that_never_returns_is_killed_at_the_deadline_and_frees_its_slot(self) -> None:
        runtime = self.runtime(cpu_workers=1)
        sleeper = self.behaviour("sleep")
        after = self.dir / "after.txt"
        after.write_text("The next document", encoding="utf-8")
        before = cpu_slots_in_use()

        async def scenario() -> tuple[float, list[ExtractedSection]]:
            started = time.perf_counter()
            with self.assertRaises(TimeoutError):
                async with asyncio.timeout(3.0):
                    await extract_isolated.extract_document(runtime, sleeper)
            elapsed = time.perf_counter() - started
            # One CPU slot and a 0.5 s acquire wait: the next document runs only if the slot was given back.
            return elapsed, await extract_isolated.extract_document(runtime, after)

        with patch.object(extract_isolated, "WORKER_MODULE", CHILD_MODULE):
            elapsed, sections = asyncio.run(scenario())
        pid = int(sleeper.with_suffix(".pid").read_text(encoding="utf-8"))
        self.assertFalse(process_alive(pid))
        self.assertLess(elapsed, 3.0 + 2.0)
        self.assertEqual([section.text for section in sections], ["The next document"])
        self.assertEqual(cpu_slots_in_use(), before)

    def test_a_child_past_the_memory_limit_is_a_document_limit(self) -> None:
        runtime = self.runtime()
        with patch.object(extract_isolated, "WORKER_MODULE", CHILD_MODULE), \
                patch.object(settings, "extraction_memory_limit_bytes", TEST_MEMORY_LIMIT):
            outcome = self.isolated(runtime, self.behaviour("allocate"))
        self.assertIsInstance(outcome, DocumentLimitError)
        self.assertEqual((outcome.code, outcome.safe_message), ("DOCUMENT_LIMIT_EXCEEDED", MEMORY_LIMIT_MESSAGE))

    @unittest.skipUnless(sys.platform == "win32", "the job object reports the failed allocation (Windows only)")
    def test_a_crash_after_an_allocation_failed_at_the_limit_is_a_document_limit(self) -> None:
        runtime = self.runtime()
        with patch.object(extract_isolated, "WORKER_MODULE", CHILD_MODULE), \
                patch.object(settings, "extraction_memory_limit_bytes", TEST_MEMORY_LIMIT), \
                self.assertLogs(extract_isolated.logger, "WARNING") as logs:
            outcome = self.isolated(runtime, self.behaviour("allocate_then_crash"))
        self.assertIsInstance(outcome, DocumentLimitError)
        self.assertEqual(logs.records[-1].reason, "memory_limit")

    def test_the_service_survives_a_child_that_crashes(self) -> None:
        runtime = self.runtime()
        healthy = self.dir / "healthy.txt"
        healthy.write_text("Still working", encoding="utf-8")
        for behaviour in ("exit", "segfault"):
            with self.subTest(behaviour=behaviour), patch.object(extract_isolated, "WORKER_MODULE", CHILD_MODULE), \
                    self.assertLogs(extract_isolated.logger, "WARNING") as logs:
                outcome = self.isolated(runtime, self.behaviour(behaviour))
                self.assertIsInstance(outcome, IsolatedExtractionError)
                self.assertEqual(outcome.error_type, "ChildProcessCrashed")
                record = logs.records[-1]
                self.assertEqual((record.event, record.reason), ("document_extraction_child_failed", "crashed"))
                self.assertNotEqual(record.exit_code, 0)
                if behaviour == "exit":
                    self.assertEqual(record.exit_code, 70)
                self.assertEqual([section.text for section in self.isolated(runtime, healthy)], ["Still working"])

    def test_without_the_memory_cap_the_child_still_runs_and_dies_at_the_deadline(self) -> None:
        class Unavailable:
            def __init__(self, _limit: int) -> None:
                raise OSError("no job objects on this host")

        class Refused:
            closed = False

            def __init__(self, _limit: int) -> None:
                pass

            def assign(self, _process: Any) -> None:
                raise PermissionError("cannot assign")

            def close(self) -> None:
                Refused.closed = True

        runtime = self.runtime()
        healthy = self.dir / "healthy.txt"
        healthy.write_text("Still working", encoding="utf-8")
        sleeper = self.behaviour("sleep")

        async def deadline() -> None:
            with self.assertRaises(TimeoutError):
                async with asyncio.timeout(3.0):
                    await extract_isolated.extract_document(runtime, sleeper)

        with patch.object(extract_isolated, "WORKER_MODULE", CHILD_MODULE), \
                self.assertLogs(child_process.logger, "WARNING") as logs:
            with patch.object(child_process, "_MemoryCap", Refused):
                self.assertEqual([section.text for section in self.isolated(runtime, healthy)], ["Still working"])
            with patch.object(child_process, "_MemoryCap", Unavailable):
                asyncio.run(deadline())
        self.assertTrue(Refused.closed)
        self.assertEqual([record.event for record in logs.records], ["child_process_memory_cap_unavailable"] * 2)
        self.assertFalse(process_alive(int(sleeper.with_suffix(".pid").read_text(encoding="utf-8"))))

    def test_shutdown_kills_the_running_children(self) -> None:
        runtime = self.runtime()
        sleeper = self.behaviour("sleep")
        pid_file = sleeper.with_suffix(".pid")

        async def scenario() -> None:
            task = asyncio.ensure_future(extract_isolated.extract_document(runtime, sleeper))
            for _ in range(200):
                if pid_file.exists():
                    break
                await asyncio.sleep(0.05)
            runtime.shutdown()
            with self.assertRaises(IsolatedExtractionError):
                await asyncio.wait_for(task, 10)

        with patch.object(extract_isolated, "WORKER_MODULE", CHILD_MODULE):
            asyncio.run(scenario())
        self.assertFalse(process_alive(int(pid_file.read_text(encoding="utf-8"))))


class IsolatedIndexTests(IsolatedTestCase):
    def run_index(self, content: bytes) -> tuple[Any, FakeDb, list[Any]]:
        db = FakeDb(document=document_row(name="upload.txt", file_path=f"{TENANT_ID}/{KB_ID}/upload.txt",
                                          content=None))
        with patch("app.services.provider.embed_texts", new=FakeEmbedder()), \
                patch("app.services.ingestion.storage.download_storage_object", return_value=content), \
                patch.object(runtime_service, "concurrency", self.runtime()), \
                patch.object(extract_isolated, "WORKER_MODULE", CHILD_MODULE), \
                self.assertLogs(index_service.logger, level="INFO") as logs:
            try:
                outcome: Any = asyncio.run(index_service.index_document(index_request(), pool=db))
            except Exception as error:
                outcome = error
        return outcome, db, logs.records

    def test_a_crashed_child_fails_the_index_with_the_extraction_failure_code(self) -> None:
        outcome, db, records = self.run_index(b"#child:segfault\n")
        self.assertEqual(outcome, {"status": "error", "chunk_count": 0, "usage": AiUsage().model_dump(),
                                   "error_reason": "INDEX_DOCUMENT_FAILED"})
        self.assertEqual(db.args_for("mark_index_error"), [(INDEX_ID, "INDEX_DOCUMENT_FAILED", TENANT_ID, KB_ID)])
        failed = [record for record in records if getattr(record, "event", None) == "rag_index_failed"]
        self.assertEqual(failed[-1].error_type, "ChildProcessCrashed")

    def test_a_child_past_the_index_deadline_is_the_deadline_code(self) -> None:
        with patch.object(settings, "index_deadline_ms", 3_000):
            outcome, db, _ = self.run_index(b"#child:sleep\n")
        self.assertEqual(outcome["error_reason"], "INDEX_DEADLINE_EXCEEDED")
        self.assertEqual(db.args_for("mark_index_error"), [(INDEX_ID, "INDEX_DEADLINE_EXCEEDED", TENANT_ID, KB_ID)])

    def test_a_child_past_the_memory_limit_is_a_document_limit_rejection(self) -> None:
        with patch.object(settings, "extraction_memory_limit_bytes", TEST_MEMORY_LIMIT):
            outcome, db, _ = self.run_index(b"#child:allocate\n")
        self.assertIsInstance(outcome, DocumentLimitError)
        self.assertEqual(db.args_for("mark_index_error"), [(INDEX_ID, "DOCUMENT_LIMIT_EXCEEDED", TENANT_ID, KB_ID)])


def outcome_of(output: bytes = b"", *, returncode: int = 0, too_large: bool = False, memory: bool = False,
               killed: bool = False) -> ChildOutcome:
    return ChildOutcome(returncode=returncode, output=output, output_too_large=too_large,
                        memory_limit_reached=memory, killed=killed)


def reply(*values: Any) -> bytes:
    return pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL)


class ReplyDecodingTests(unittest.TestCase):
    def decoded(self, outcome: ChildOutcome) -> Any:
        try:
            return extract_isolated.sections_from_outcome(outcome)
        except Exception as error:
            return error

    def assert_invalid(self, outcome: ChildOutcome) -> None:
        with self.assertLogs(extract_isolated.logger, "WARNING"):
            error = self.decoded(outcome)
        self.assertIsInstance(error, IsolatedExtractionError)
        self.assertEqual(error.error_type, "ChildReplyInvalid")

    def test_a_reply_never_resolves_a_class(self) -> None:
        # GLOBAL "tests.support.never_imported boom": a default unpickler would import the module.
        crafted = b"\x80\x04" + b"ctests.support.never_imported\nboom\n" + b"\x85."
        self.assert_invalid(outcome_of(crafted))
        self.assertNotIn("tests.support.never_imported", sys.modules)
        self.assert_invalid(outcome_of(reply("ok", [(Path("text"), None, None, {})])))

    def test_malformed_replies_are_invalid(self) -> None:
        for output in (b"", b"not a pickle", reply("unknown"), reply("ok"), reply("ok", "text"),
                       reply("ok", [("text", None, None)]), reply("ok", [("text", True, None, {})]),
                       reply("ok", [("text", 1.0, None, {})]), reply("ok", [("text", None, 3, {})]),
                       reply("ok", [("text", None, None, [])]), reply("app_error", "NOT_A_CODE", 422, "x"),
                       reply("app_error", "DOCUMENT_LIMIT_EXCEEDED", 500, "x"), reply("value_error", 3),
                       pickle.dumps(["ok", []])):
            with self.subTest(output=output[:40]):
                self.assert_invalid(outcome_of(output))

    def test_error_replies_raise_the_parser_errors(self) -> None:
        limit = self.decoded(outcome_of(reply("app_error", "DOCUMENT_LIMIT_EXCEEDED", 422, "Too many pages.")))
        self.assertIsInstance(limit, DocumentLimitError)
        self.assertEqual(limit.safe_message, "Too many pages.")
        unsupported = self.decoded(outcome_of(reply("app_error", "DOCUMENT_TYPE_UNSUPPORTED", 422, "x")))
        self.assertEqual(unsupported.code, "DOCUMENT_TYPE_UNSUPPORTED")
        other = self.decoded(outcome_of(reply("app_error", "STORAGE_DOWNLOAD_FAILED", 502, "x")))
        self.assertIs(type(other), AppError)
        value = self.decoded(outcome_of(reply("value_error", "OCR_REQUIRED: no text")))
        self.assertIs(type(value), ValueError)
        self.assertEqual(str(value), "OCR_REQUIRED: no text")
        self.assertEqual(self.decoded(outcome_of(reply("error", "FileDataError"))).error_type, "FileDataError")
        self.assertEqual(self.decoded(outcome_of(reply("error", "not a name!"))).error_type, "OtherException")
        memory = self.decoded(outcome_of(reply("memory")))
        self.assertEqual((type(memory), memory.safe_message), (DocumentLimitError, MEMORY_LIMIT_MESSAGE))

    def test_what_the_parent_saw_decides_before_the_reply(self) -> None:
        sections = reply("ok", [("text", 1, None, {"k": (1, 2)})])
        self.assertEqual(self.decoded(outcome_of(sections)), [ExtractedSection("text", 1, None, {"k": (1, 2)})])
        with self.assertLogs(extract_isolated.logger, "WARNING"):
            self.assertIsInstance(self.decoded(outcome_of(sections, memory=True)), DocumentLimitError)
            self.assertIsInstance(self.decoded(outcome_of(b"", too_large=True)), DocumentLimitError)
            self.assertIsInstance(self.decoded(outcome_of(sections, killed=True)), IsolatedExtractionError)
            self.assertIsInstance(self.decoded(outcome_of(sections, returncode=1)), IsolatedExtractionError)
        with patch.object(settings, "max_extracted_chars", 1_000):
            self.assertIsInstance(self.decoded(outcome_of(reply("ok", [("x" * 1_001, None, None, {})]))),
                                  DocumentLimitError)

    def test_the_request_carries_the_settings_but_no_secret(self) -> None:
        with patch.object(settings, "database_url", SecretStr("postgresql://user:hidden-password@db/x")), \
                patch.object(settings, "max_document_pages", 77):
            request = json.loads(extract_isolated.extraction_request(Path("/tmp/a.pdf"), "a.pdf"))  # noqa: S108
        # The child runs in another working directory: the path is sent absolute (review of wave 3, A5).
        self.assertEqual((request["path"], request["file_name"]), (str(Path("/tmp/a.pdf").resolve()), "a.pdf"))  # noqa: S108
        self.assertEqual(request["settings"]["max_document_pages"], 77)
        self.assertNotIn("database_url", request["settings"])
        self.assertNotIn("hidden-password", json.dumps(request))
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://hidden", "PATH": "x"}):
            environment = child_environment({"AI_RAG_ENV": "production"})
        self.assertNotIn("DATABASE_URL", environment)
        self.assertEqual(environment["AI_RAG_ENV"], "production")


class WorkerTests(unittest.TestCase):
    def serve(self, extract: Any, settings_values: dict[str, Any] | None = None) -> Any:
        request = json.dumps({"path": "doc.txt", "file_name": "doc.txt", "settings": settings_values or {}})
        replies = io.BytesIO()
        self.assertEqual(extract_worker.main(extract, request_stream=io.BytesIO(request.encode()),
                                             reply_stream=replies), 0)
        return pickle.loads(replies.getvalue())  # noqa: S301 - the test's own reply

    def test_every_outcome_is_one_plain_reply(self) -> None:
        def ok(_path: Path, name: str) -> list[ExtractedSection]:
            return [ExtractedSection(text=name, page=2, section="S", metadata={"a": [1]})]

        def raising(error: BaseException) -> Any:
            def extract(_path: Path, _name: str) -> list[ExtractedSection]:
                raise error
            return extract

        self.assertEqual(self.serve(ok), ("ok", [("doc.txt", 2, "S", {"a": [1]})]))
        self.assertEqual(self.serve(raising(MemoryError())), ("memory",))
        self.assertEqual(self.serve(raising(DocumentLimitError("Big."))),
                         ("app_error", "DOCUMENT_LIMIT_EXCEEDED", 422, "Big."))
        self.assertEqual(self.serve(raising(ValueError("OCR_REQUIRED: x"))), ("value_error", "OCR_REQUIRED: x"))
        self.assertEqual(self.serve(raising(KeyError("k"))), ("error", "KeyError"))

    def test_the_request_settings_apply_in_the_worker(self) -> None:
        seen: list[int] = []

        def extract(_path: Path, _name: str) -> list[ExtractedSection]:
            seen.append(settings.max_document_pages)
            return []

        with patch.object(settings, "max_document_pages", settings.max_document_pages):
            self.serve(extract, {"max_document_pages": 9, "not_a_setting": 1})
        self.assertEqual(seen, [9])
        self.assertFalse(hasattr(settings, "not_a_setting"))
        self.assertEqual(extract_worker.encode_reply(("memory",)), reply("memory"))


if __name__ == "__main__":
    unittest.main()
