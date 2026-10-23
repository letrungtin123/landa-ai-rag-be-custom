"""A cancelled index request closes its index row (security audit 2026-10-24, O2).

A client disconnect or a service stop cancels the request's task. ``except Exception`` never sees a cancellation,
so the row stayed "running" and the next request for the document embedded everything again.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import patch

from app.services.ingestion import index as index_service
from tests.unit.services.test_characterization_ingestion import INDEX_ID, FakeDb, document_row, index_request


class BlockingEmbedder:
    """Stands for a long provider call: the request is waiting in it when the caller goes away."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self.started.set()
        await asyncio.Event().wait()


class SlowWriteDb(FakeDb):
    """The error write takes a moment and records when it has finished."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.writing = asyncio.Event()
        self.finished: list[tuple[Any, ...]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        result = await super().execute(sql, *args)
        if "error_reason = $2" in sql:
            self.writing.set()
            await asyncio.sleep(0.05)
            self.finished.append(args)
        return result


class IndexCancellationTests(unittest.TestCase):
    def run_cancelled(self, db: FakeDb, *, cancels: int = 1) -> list[str]:
        async def scenario() -> None:
            embedder = BlockingEmbedder()
            with patch("app.services.provider.embed_texts", new=embedder):
                task = asyncio.create_task(index_service.index_document(index_request(), pool=db))
                await asyncio.wait_for(embedder.started.wait(), 5)
                task.cancel()
                if cancels > 1 and isinstance(db, SlowWriteDb):
                    await asyncio.wait_for(db.writing.wait(), 5)
                    task.cancel()  # a second cancellation while the row is being written
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0.1)  # the shielded write finishes on its own

        with self.assertLogs(index_service.logger, level="INFO") as logs:
            asyncio.run(scenario())
        return [getattr(record, "event", "") for record in logs.records]

    def test_the_row_is_marked_failed_and_the_cancellation_goes_on(self) -> None:
        db = FakeDb(document=document_row())
        events = self.run_cancelled(db)
        self.assertEqual(db.args_for("mark_index_error"), [(INDEX_ID, "INDEX_CANCELLED")])
        self.assertEqual(events[-1], "rag_index_cancelled")
        self.assertEqual(db.args_for("insert_chunk"), [])

    def test_a_second_cancellation_does_not_stop_the_write(self) -> None:
        db = SlowWriteDb(document=document_row())
        self.run_cancelled(db, cancels=2)
        self.assertEqual(db.finished, [(INDEX_ID, "INDEX_CANCELLED")])

    def test_a_hung_write_gives_up_after_its_timeout(self) -> None:
        class HungWriteDb(FakeDb):
            async def execute(self, sql: str, *args: Any) -> str:
                if "error_reason = $2" in sql:
                    await asyncio.Event().wait()
                return await super().execute(sql, *args)

        db = HungWriteDb(document=document_row())
        with patch.object(index_service, "CANCEL_RECORD_TIMEOUT_SECONDS", 0.05):
            events = self.run_cancelled(db)
        self.assertEqual(events[-1], "broad_exception_handled")
        self.assertNotIn("rag_index_cancelled", events)

    def test_a_failed_write_is_reported_and_the_cancellation_still_goes_on(self) -> None:
        db = FakeDb(document=document_row(), errors={"error_reason = $2": OSError("database gone")})
        events = self.run_cancelled(db)
        self.assertEqual(events[-1], "broad_exception_handled")
        self.assertNotIn("rag_index_cancelled", events)

    def test_a_request_cancelled_before_its_row_writes_nothing(self) -> None:
        class SlowDocumentDb(FakeDb):
            def __init__(self) -> None:
                super().__init__(document=document_row())
                self.loading = asyncio.Event()

            async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
                self.loading.set()
                await asyncio.Event().wait()
                return None

        db = SlowDocumentDb()

        async def scenario() -> None:
            task = asyncio.create_task(index_service.index_document(index_request(), pool=db))
            await db.loading.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        self.assertEqual(db.args_for("mark_index_error"), [])


if __name__ == "__main__":
    unittest.main()
