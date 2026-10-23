"""A re-index of an unchanged document reuses its active index: no embedding call, no new index row.

Offline: the recording fake pool of the ingestion characterization tests, with the document's active index
answered per lookup (the first check once the source is read, the re-check once the fingerprint is known).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock, patch

from app.core.errors import DocumentLimitError
from app.schemas.common import AiUsage
from app.schemas.kb import INDEX_DOCUMENT_RESPONSE, INDEX_SKIP_REASON_UNCHANGED, RagIndexRequest
from app.services.ingestion import fingerprint as fingerprint_module
from app.services.ingestion import index as index_service
from app.services.ingestion.fingerprint import (
    StoredChunk,
    index_fingerprint,
    index_pipeline_versions,
    stored_chunk_values,
)
from tests.contract.test_route_snapshots import replay
from tests.unit.services.test_characterization_ingestion import (
    DOC_ID,
    INDEX_ID,
    KB_ID,
    TENANT_ID,
    FakeDb,
    FakeEmbedder,
    document_row,
    index_request,
    write_docx,
)

ACTIVE_INDEX_ID = "66666666-6666-4666-8666-666666666666"
OTHER_INDEX_ID = "77777777-7777-4777-8777-777777777777"
BODY = "Chương 1. An toàn điện\n\n1.1 Cắt nguồn điện\nNgắt cầu dao tổng trước khi bảo trì thiết bị điện."
ZERO_USAGE = AiUsage().model_dump()
REBUILD_LABELS = ["supersede_running_indexes", "next_index_version", "insert_index_row", "lock_index_row",
                  "insert_chunk", "insert_structure_nodes", "delete_structure_nodes", "deactivate_other_indexes",
                  "activate_index", "delete_indexes"]


def active_index(fingerprint: str | None, chunk_count: int, *, index_id: str = ACTIVE_INDEX_ID,
                 running: bool = False) -> dict[str, Any]:
    return {"index_id": index_id, "content_sha256": fingerprint, "chunk_count": chunk_count, "index_running": running}


class ActiveIndexDb(FakeDb):
    """``FakeDb`` whose ``load_active_index`` answers come from ``active``, one per lookup (then ``None``)."""

    def __init__(self, *, active: list[dict[str, Any] | None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.active = list(active)

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        if "FROM rag_document_indexes active" in " ".join(sql.split()):
            self._record("fetchrow", sql, args)
            return self.active.pop(0) if self.active else None
        return await super().fetchrow(sql, *args)


def run_index(db: FakeDb, request: RagIndexRequest | None = None, *, embedder: FakeEmbedder | None = None,
              download: bytes | None = None) -> tuple[Any, FakeEmbedder, list[str]]:
    """Run the index service; an exception is returned instead of raised. Returns the log events too."""
    embedder = embedder or FakeEmbedder()
    downloader = Mock(return_value=download, side_effect=None if download is not None
                      else AssertionError("download not expected"))
    with patch("app.services.provider.embed_texts", new=embedder), \
            patch("app.services.ingestion.storage.download_storage_object", new=downloader), \
            unittest.TestCase().assertLogs(index_service.logger, level="INFO") as logs:
        try:
            outcome: Any = asyncio.run(index_service.index_document(request or index_request(), pool=db))
        except Exception as error:
            outcome = error
    return outcome, embedder, [getattr(record, "event", None) or record.getMessage() for record in logs.records]


def learn(row: dict[str, Any], *, download: bytes | None = None,
          request: RagIndexRequest | None = None) -> tuple[dict[str, Any], str]:
    """Index ``row`` on a document without an index; returns the answer and the stored fingerprint."""
    db = FakeDb(document=row)
    result, _, _ = run_index(db, request, download=download)
    [(_, _, fingerprint, _, _, _)] = db.args_for("activate_index")
    return result, fingerprint


class SkipUnchangedIndexTests(unittest.TestCase):
    maxDiff = None

    def assert_skipped(self, db: FakeDb, result: Any, learned: dict[str, Any], embedder: FakeEmbedder,
                       events: list[str]) -> None:
        self.assertEqual(result, {**learned, "usage": ZERO_USAGE, "skipped": True,
                                  "skip_reason": INDEX_SKIP_REASON_UNCHANGED})
        self.assertEqual(embedder.calls, [])
        # Only reads: no index row is created, superseded, marked or deleted.
        self.assertEqual(db.labels(), ["load_document", "load_active_index", "load_active_index"])
        self.assertEqual(db.events, [])
        self.assertIn("rag_index_skipped", events)
        self.assertNotIn("rag_index_embedded", events)

    def test_unchanged_text_document_is_skipped_without_provider_call_or_index_row(self) -> None:
        row = document_row(content=BODY)
        learned, fingerprint = learn(row)
        chunk_count = learned["chunk_count"]
        db = ActiveIndexDb(document=row, active=[active_index(fingerprint, chunk_count)] * 2)
        result, embedder, events = run_index(db)
        self.assert_skipped(db, result, learned, embedder, events)
        self.assertEqual(db.args_for("load_active_index"),
                         [(TENANT_ID, KB_ID, DOC_ID, "gemini-embedding-001", 768)] * 2)

    def test_unchanged_file_document_is_skipped_after_reading_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_docx(Path(directory) / "guide.docx")
            raw = (Path(directory) / "guide.docx").read_bytes()
        row = document_row(name="guide.docx", type="docx", file_path=f"{TENANT_ID}/{KB_ID}/guide.docx", content=None)
        learned, fingerprint = learn(row, download=raw)
        db = ActiveIndexDb(document=row, active=[active_index(fingerprint, learned["chunk_count"])] * 2)
        result, embedder, events = run_index(db, download=raw)
        self.assert_skipped(db, result, learned, embedder, events)
        self.assertGreater(result["structure_node_count"], 0)

    def test_changed_content_is_reindexed(self) -> None:
        _, old_fingerprint = learn(document_row(content=BODY))
        row = document_row(content=BODY + "\n\n1.2 Khoá và gắn thẻ\nMỗi người gắn khoá cá nhân.")
        db = ActiveIndexDb(document=row, active=[active_index(old_fingerprint, 1)] * 2)
        result, embedder, events = run_index(db)
        self.assertEqual(result["status"], "learned")
        self.assertNotIn("skipped", result)
        self.assertEqual(len(embedder.calls), 1)
        self.assertEqual(db.labels(), ["load_document", "load_active_index", "load_active_index", *REBUILD_LABELS])
        [(_, _, new_fingerprint, _, _, _)] = db.args_for("activate_index")
        self.assertNotEqual(new_fingerprint, old_fingerprint)
        self.assertNotIn("rag_index_skipped", events)

    def test_renamed_document_is_reindexed_because_chunks_store_its_name(self) -> None:
        _, fingerprint = learn(document_row(content=BODY))
        db = ActiveIndexDb(document=document_row(content=BODY, name="Renamed"),
                           active=[active_index(fingerprint, 1)] * 2)
        result, embedder, _ = run_index(db)
        self.assertEqual((result["status"], len(embedder.calls)), ("learned", 1))
        chunk_rows = db.chunk_rows()
        self.assertEqual(json.loads(chunk_rows[0][10])["source_name"], "Renamed")

    def test_same_content_with_another_pipeline_version_is_reindexed(self) -> None:
        row = document_row(content=BODY)
        _, fingerprint = learn(row)
        for name in ("INDEX_FINGERPRINT_VERSION", "STRUCTURED_EXTRACTION_VERSION", "PARSER_VERSION",
                     "CHUNKER_VERSION", "SOURCE_EVIDENCE_PROPAGATION_VERSION", "EMBEDDING_TASK_TYPE"):
            with self.subTest(version=name), patch.object(fingerprint_module, name, "next-version"):
                db = ActiveIndexDb(document=row, active=[active_index(fingerprint, 1)] * 2)
                result, embedder, _ = run_index(db)
                self.assertEqual((result["status"], len(embedder.calls)), ("learned", 1))
                self.assertNotIn("skipped", result)
                self.assertIn("insert_index_row", db.labels())

    def test_same_content_with_another_embedding_model_is_reindexed(self) -> None:
        row = document_row(content=BODY)
        _, fingerprint = learn(row)
        # The lookup is scoped to the request's model and size, so another model finds no active index...
        db = ActiveIndexDb(document=row, active=[None])
        result, embedder, _ = run_index(db, index_request(embedding_model="gemini-embedding-002"))
        self.assertEqual((result["status"], len(embedder.calls)), ("learned", 1))
        self.assertEqual(db.args_for("load_active_index")[0][3:], ("gemini-embedding-002", 768))
        # ...and its fingerprint differs anyway.
        _, other = learn(row, request=index_request(embedding_model="gemini-embedding-002"))
        self.assertNotEqual(other, fingerprint)

    def test_force_rebuilds_without_looking_up_the_active_index(self) -> None:
        row = document_row(content=BODY)
        _, fingerprint = learn(row)
        db = ActiveIndexDb(document=row, active=[active_index(fingerprint, 1)] * 2)
        result, embedder, events = run_index(db, index_request(force=True))
        self.assertEqual((result["status"], len(embedder.calls)), ("learned", 1))
        self.assertEqual(db.labels(), ["load_document", *REBUILD_LABELS])
        self.assertEqual(db.args_for("activate_index")[0][2], fingerprint)
        self.assertNotIn("rag_index_skipped", events)

    def test_a_running_index_disables_the_skip_and_is_superseded_as_before(self) -> None:
        row = document_row(content=BODY)
        _, fingerprint = learn(row)
        db = ActiveIndexDb(document=row, active=[active_index(fingerprint, 1, running=True)])
        result, embedder, _ = run_index(db)
        self.assertEqual((result["status"], len(embedder.calls)), ("learned", 1))
        # The row is taken before extraction, superseding the running index, exactly as without an active index.
        self.assertEqual(db.labels(), ["load_document", "load_active_index", *REBUILD_LABELS])

    def test_an_index_started_during_the_check_supersedes_this_request(self) -> None:
        row = document_row(content=BODY)
        _, fingerprint = learn(row)
        stale = active_index("0" * 64, 1)
        rechecks = {
            "now_running": active_index("0" * 64, 1, running=True),
            "same_fingerprint_but_running": active_index(fingerprint, 1, running=True),
            "replaced_by_another_index": active_index("1" * 64, 1, index_id=OTHER_INDEX_ID),
            "indexes_deleted": None,
        }
        for name, recheck in rechecks.items():
            with self.subTest(case=name):
                db = ActiveIndexDb(document=row, active=[stale, recheck])
                result, embedder, events = run_index(db)
                # The same answer a superseded run gives, without the embedding call it used to pay for.
                self.assertEqual(result, {"status": "error", "chunk_count": 0, "usage": ZERO_USAGE,
                                          "error_reason": "INDEX_DOCUMENT_FAILED"})
                self.assertEqual(embedder.calls, [])
                self.assertEqual(db.labels(), ["load_document", "load_active_index", "load_active_index"])
                self.assertEqual(events[-1], "rag_index_failed")

    def test_another_index_that_already_holds_the_same_content_is_reused(self) -> None:
        row = document_row(content=BODY)
        learned, fingerprint = learn(row)
        db = ActiveIndexDb(document=row, active=[active_index("0" * 64, 1),
                                                 active_index(fingerprint, 1, index_id=OTHER_INDEX_ID)])
        result, embedder, events = run_index(db)
        self.assert_skipped(db, result, learned, embedder, events)

    def test_chunk_count_must_match_as_well(self) -> None:
        row = document_row(content=BODY)
        _, fingerprint = learn(row)
        db = ActiveIndexDb(document=row, active=[active_index(fingerprint, 2)] * 2)
        result, embedder, _ = run_index(db)
        self.assertEqual((result["status"], len(embedder.calls)), ("learned", 1))
        self.assertNotIn("skipped", result)

    def test_failures_before_the_index_row_answer_as_before_and_write_nothing(self) -> None:
        # A reuse candidate takes its row only after the fingerprint; a failure before that answers exactly
        # what a run that took its row first answers, and leaves the document's indexes untouched.
        cases = {
            "corrupt_pdf": (document_row(name="broken.pdf", file_path=f"{TENANT_ID}/{KB_ID}/broken.pdf",
                                         content=None), b"%PDF-1.4 garbage"),
            "unsafe_ooxml": (document_row(name="x.docx", file_path=f"{TENANT_ID}/{KB_ID}/x.docx", content=None),
                             b"not zip"),
        }
        for name, (row, download) in cases.items():
            with self.subTest(case=name):
                direct_db = FakeDb(document=row)
                direct, _, _ = run_index(direct_db, download=download)
                deferred_db = ActiveIndexDb(document=row, active=[active_index("0" * 64, 1)])
                deferred, embedder, events = run_index(deferred_db, download=download)
                if isinstance(direct, Exception):
                    self.assertIsInstance(direct, DocumentLimitError)
                    self.assertIs(type(deferred), type(direct))
                    self.assertEqual((deferred.code, deferred.http_status), (direct.code, direct.http_status))
                else:
                    self.assertEqual(deferred, direct)
                self.assertEqual(direct_db.args_for("mark_index_error"),
                                 [(INDEX_ID, "INDEX_DOCUMENT_FAILED" if name == "corrupt_pdf"
                                   else "DOCUMENT_LIMIT_EXCEEDED", TENANT_ID, KB_ID)])
                self.assertEqual(deferred_db.labels(), ["load_document", "load_active_index"])
                self.assertEqual(embedder.calls, [])
                self.assertEqual(events[-1], "rag_index_failed")

    def test_provider_failure_after_the_deferred_row_is_marked_on_that_row(self) -> None:
        row = document_row(content=BODY + " thay đổi")
        db = ActiveIndexDb(document=row, active=[active_index("0" * 64, 1)] * 2)
        result, embedder, _ = run_index(db, embedder=FakeEmbedder(drop=1))
        self.assertEqual(result, {"status": "error", "chunk_count": 0, "usage": ZERO_USAGE,
                                  "error_reason": "INDEX_DOCUMENT_FAILED"})
        self.assertEqual(len(embedder.calls), 1)
        self.assertEqual(db.args_for("mark_index_error"), [(INDEX_ID, "INDEX_DOCUMENT_FAILED", TENANT_ID, KB_ID)])


class IndexFingerprintTests(unittest.TestCase):
    CHUNKS: ClassVar[list[StoredChunk]] = [(0, "Body", "hash-0", 1, None, None, '{"source_name": "Doc"}'),
                                           (1, "More", "hash-1", 1, 2, "Section", '{"source_name": "Doc"}')]
    STRUCTURE: ClassVar[dict[str, Any]] = {"nodes": [{"source_ref": "src-001", "title": "Chapter"}], "confidence": 0.5}

    def fingerprint(self, **overrides: Any) -> str:
        arguments: dict[str, Any] = {"embedding_model": "gemini-embedding-001", "embedding_dimensions": 768,
                                     "structure": self.STRUCTURE, "chunks": self.CHUNKS, **overrides}
        return index_fingerprint(**arguments)

    def test_fingerprint_is_stable_and_covers_every_stored_value(self) -> None:
        base = self.fingerprint()
        self.assertEqual(base, self.fingerprint(structure=json.loads(json.dumps(self.STRUCTURE))))
        self.assertRegex(base, r"^[0-9a-f]{64}$")
        variants = {
            "model": {"embedding_model": "gemini-embedding-002"},
            "dimensions": {"embedding_dimensions": 1536},
            "structure": {"structure": {**self.STRUCTURE, "confidence": 0.6}},
            "chunk_order": {"chunks": list(reversed(self.CHUNKS))},
            "chunk_dropped": {"chunks": self.CHUNKS[:1]},
            "page": {"chunks": [self.CHUNKS[0], (*self.CHUNKS[1][:4], 3, *self.CHUNKS[1][5:])]},
            "metadata": {"chunks": [self.CHUNKS[0], (*self.CHUNKS[1][:6], '{"source_name": "Other"}')]},
        }
        for name, override in variants.items():
            with self.subTest(changed=name):
                self.assertNotEqual(self.fingerprint(**override), base)

    def test_versions_and_chunk_settings_are_part_of_the_fingerprint(self) -> None:
        base = self.fingerprint()
        self.assertEqual(set(index_pipeline_versions()), {
            "fingerprint", "extraction", "structure_parser", "chunker", "evidence_contract", "chunk_max_chars",
            "chunk_overlap_chars", "embedding_task_type",
        })
        for name in ("INDEX_FINGERPRINT_VERSION", "STRUCTURED_EXTRACTION_VERSION", "PARSER_VERSION",
                     "CHUNKER_VERSION", "SOURCE_EVIDENCE_PROPAGATION_VERSION", "EMBEDDING_TASK_TYPE"):
            with self.subTest(version=name), patch.object(fingerprint_module, name, "next-version"):
                self.assertNotEqual(self.fingerprint(), base)
        settings = fingerprint_module.settings
        for setting in ("chunk_max_chars", "chunk_overlap_chars"):
            with self.subTest(setting=setting), patch.object(settings, setting, getattr(settings, setting) + 1):
                self.assertNotEqual(self.fingerprint(), base)

    def test_stored_chunk_values_follow_the_chunk_insert_columns(self) -> None:
        chunk = {"content": "Body", "content_hash": "h", "token_count": 3, "page": 4, "section": "S",
                 "metadata": {"source_ref": "src-001"}}
        [values] = stored_chunk_values({"name": "Doc", "type": "pdf"}, [chunk])
        self.assertEqual(values[:6], (0, "Body", "h", 3, 4, "S"))
        self.assertEqual(json.loads(values[6]), {"source_name": "Doc", "document_type": "pdf", "source_ref": "src-001"})


class SkipResponseWireTests(unittest.TestCase):
    """The skipped answer through the real route (auth, validation, response model, serialization)."""

    def case(self, active: dict[str, Any] | None) -> dict[str, Any]:
        rules: list[dict[str, Any]] = [
            {"match": "FROM kb_documents", "method": "fetchrow", "rows": [{
                "id": DOC_ID, "tenant_id": TENANT_ID, "kb_id": KB_ID, "type": "text", "name": "an-toan.txt",
                "status": "learned", "source_info": None, "file_path": None, "content": BODY}]},
            {"match": "COALESCE(MAX(version), 0) + 1", "method": "fetchval", "value": 2},
            {"match": "INSERT INTO rag_document_indexes", "method": "fetchval", "value": INDEX_ID},
            {"match": "FOR UPDATE", "method": "fetchval", "value": "running"},
        ]
        if active is not None:
            rules.insert(1, {"match": "FROM rag_document_indexes active", "method": "fetchrow", "rows": [active]})
        return {"request": {"method": "POST", "path": "/v1/kb/documents/index", "json": {
            "tenant_id": TENANT_ID, "kb_id": KB_ID, "document_id": DOC_ID,
            "embedding_model": "gemini-embedding-001", "embedding_dimensions": 768,
            "api_key": "route-snapshot-provider-key"}},
            "db": rules}

    def test_skipped_answer_on_the_wire(self) -> None:
        learned = replay(self.case(None))
        self.assertEqual(learned["status"], 200)
        self.assertNotIn("skipped", learned["body"])
        self.assertNotIn("skip_reason", learned["body"])
        # mark_index_learned(index_id, document_id, fingerprint, chunk_count, tenant_id, kb_id)
        [stored] = [call["args"][2] for call in learned["db_calls"]
                    if call["method"] == "execute" and len(call["args"]) == 6 and call["args"][0] == INDEX_ID]
        skipped = replay(self.case(active_index(stored, learned["body"]["chunk_count"])))
        self.assertEqual(skipped["status"], 200)
        self.assertEqual(skipped["provider_calls"], [])
        self.assertEqual(skipped["body"], {**learned["body"], "usage": ZERO_USAGE, "skipped": True,
                                           "skip_reason": "INDEX_UNCHANGED"})
        self.assertEqual(list(skipped["body"])[-2:], ["skipped", "skip_reason"])
        self.assertEqual([call["method"] for call in skipped["db_calls"]], ["fetchrow"] * 3)
        INDEX_DOCUMENT_RESPONSE.validate_python(skipped["body"])


if __name__ == "__main__":
    unittest.main()
