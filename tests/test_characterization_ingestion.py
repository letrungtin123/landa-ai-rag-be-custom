"""Characterization tests locking CURRENT ingestion behavior in app.main (extraction, indexing, deletion).

Offline only: a recording fake replaces the asyncpg pool, embed_texts/download_storage_object are patched at
the app.main boundary, and source files are generated at runtime in temp dirs. "Characterized:" comments flag
current behavior that may be worth revisiting."""
from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import asyncpg
import pymupdf
from docx import Document
from fastapi import HTTPException
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches
from pydantic import ValidationError

from app import main
from app.core.errors import AppError, DocumentLimitError
from app.repositories import indexing as index_repository
from app.schemas.common import AiUsage
from app.schemas.kb import RagDeleteDocumentRequest, RagDeleteKbRequest, RagIndexRequest
from app.services import provider as provider_service
from app.services.ingestion import chunking
from app.services.ingestion import documents as documents_service
from app.services.ingestion import extract as extraction
from app.services.ingestion import index as index_service
from app.services.ingestion.chunking import SOURCE_EVIDENCE_PROPAGATION_VERSION
from app.services.ingestion.extract import STRUCTURED_EXTRACTION_VERSION, ExtractedSection
from app.services.retrieval import source_coverage
from app.source_structure import PARSER_VERSION

TENANT_ID = "11111111-1111-4111-8111-111111111111"
KB_ID = "22222222-2222-4222-8222-222222222222"
DOC_ID = "33333333-3333-4333-8333-333333333333"
INDEX_ID = "44444444-4444-4444-8444-444444444444"
OTHER_TENANT_ID = "55555555-5555-4555-8555-555555555555"
API_KEY = "offline-characterization-key"
FAKE_SOFFICE = r"C:\fake\LibreOffice\program\soffice.exe"

# Ordered (fragment, label) pairs; the first fragment found in a statement labels it.
SQL_LABELS = (
    ("FROM kb_documents", "load_document"), ("error_reason = $2", "mark_index_error"),
    ("AND status = 'running'", "supersede_running_indexes"), ("COALESCE(MAX(version), 0) + 1", "next_index_version"),
    ("INSERT INTO rag_document_indexes", "insert_index_row"), ("FOR UPDATE", "lock_index_row"),
    ("INSERT INTO rag_chunks", "insert_chunk"), ("INSERT INTO rag_document_structure_nodes", "insert_structure_nodes"),
    ("DELETE FROM rag_document_structure_nodes", "delete_structure_nodes"),
    ("SET is_active = false", "deactivate_other_indexes"), ("SET status = 'learned'", "activate_index"),
    ("DELETE FROM rag_document_indexes", "delete_indexes"),
)


class FakeDb:
    """Recording asyncpg Pool *and* Connection stand-in: ``calls`` = (method, normalized SQL, args),
    ``events`` = (begin|commit|rollback, depth); ``structure_table=False`` simulates a missing table."""

    def __init__(self, *, document: dict[str, Any] | None = None, structure_table: bool = True,
                 index_status: str | None = "running", next_version: int = 3,
                 errors: dict[str, Exception] | None = None) -> None:
        self.document, self.structure_table, self.index_status = document, structure_table, index_status
        self.next_version, self.errors = next_version, dict(errors or {})
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.events: list[tuple[str, int]] = []
        self.depth = self.acquired = 0

    def _record(self, method: str, sql: str, args: tuple[Any, ...]) -> str:
        text = " ".join(sql.split())
        self.calls.append((method, text, args))
        if "rag_document_structure_nodes" in text and not self.structure_table:
            raise asyncpg.exceptions.UndefinedTableError('relation "rag_document_structure_nodes" does not exist')
        for fragment, error in self.errors.items():
            if fragment in text:
                raise error
        return text

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        return self.document if "FROM kb_documents" in self._record("fetchrow", sql, args) else None

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._record("fetch", sql, args)
        return []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        text = self._record("fetchval", sql, args)
        if "COALESCE(MAX(version), 0) + 1" in text:
            return self.next_version
        if "INSERT INTO rag_document_indexes" in text:
            return INDEX_ID
        return self.index_status if "FOR UPDATE" in text else None

    async def execute(self, sql: str, *args: Any) -> str:
        self._record("execute", sql, args)
        return "OK"

    async def executemany(self, sql: str, args: Any) -> None:
        self._record("executemany", sql, (list(args),))

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[FakeDb]:
        self.acquired += 1
        yield self

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[FakeDb]:
        self.depth += 1
        self.events.append(("begin", self.depth))
        try:
            yield self
        except BaseException:
            self.events.append(("rollback", self.depth))
            raise
        else:
            self.events.append(("commit", self.depth))
        finally:
            self.depth -= 1

    def labels(self) -> list[str]:
        return [next((label for fragment, label in SQL_LABELS if fragment in sql), sql[:40])
                for _, sql, _ in self.calls]

    def args_for(self, label: str) -> list[tuple[Any, ...]]:
        return [args for (_, _, args), name in zip(self.calls, self.labels(), strict=True) if name == label]


class FakeEmbedder:
    """Async replacement for app.main.embed_texts that never touches a provider."""

    def __init__(self, *, drop: int = 0, error: Exception | None = None) -> None:
        self.drop, self.error = drop, error
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, api_key: Any, model: str, contents: list[str], *,
                       task_type: str | None = None, output_dimensionality: int = 768) -> Any:
        self.calls.append({"api_key": api_key, "model": model, "contents": list(contents),
                           "task_type": task_type, "output_dimensionality": output_dimensionality})
        if self.error is not None:
            raise self.error
        vectors = [[0.5, -0.25] + [0.0] * 766 for _ in contents[: max(0, len(contents) - self.drop)]]
        return vectors, AiUsage(embeddingTokens=7 * len(contents), totalTokens=7 * len(contents))


def document_row(**overrides: Any) -> dict[str, Any]:
    row = {"id": DOC_ID, "tenant_id": TENANT_ID, "kb_id": KB_ID, "type": "text", "name": "Safety notes",
           "status": "pending", "source_info": None, "file_path": None, "content": "Short body text."}
    return {**row, **overrides}


def index_request(**overrides: Any) -> RagIndexRequest:
    payload = {"tenant_id": TENANT_ID, "kb_id": KB_ID, "document_id": DOC_ID,
               "embedding_model": "gemini-embedding-001", "api_key": API_KEY}
    return RagIndexRequest(**{**payload, **overrides})


THREE_PARAGRAPHS = "\n\n".join(f"Paragraph {n}: " + "safety control evidence " * 6 for n in (1, 2, 3))


# ---- runtime fixture builders (nothing is committed) -----------------------------

def write_pdf(path: Path, pages: int | None = None) -> None:
    """pages=None: rich 3-page document; pages=0: one blank page; pages=n: n one-line pages."""
    document = pymupdf.open()
    if pages is not None:
        for number in range(max(1, pages)):
            page = document.new_page()
            if pages:
                page.insert_text((72, 72), f"Bounded page {number + 1}")
    else:
        page = document.new_page(width=600, height=800)
        page.insert_text((60, 80), "Chapter One Overview", fontsize=24)
        page.insert_text((60, 130), "Body text line about safety controls.", fontsize=11)
        page.insert_text((60, 150), "Second body line with more words here.", fontsize=11)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 240, 160), False)
        pixmap.clear_with(160)
        page = document.new_page(width=600, height=800)
        page.insert_text((60, 80), "Image situation", fontsize=20)
        page.insert_text((60, 130), "Please observe the scenario?", fontsize=12)
        page.insert_image(pymupdf.Rect(100, 180, 500, 500), stream=pixmap.tobytes("png"))
        document.new_page()  # trailing blank page
    document.save(path)
    document.close()


def write_docx(path: Path) -> None:
    document = Document()
    document.add_heading("Muc 1 Gioi thieu", level=1)
    document.add_paragraph("Doan van ban mo dau.")
    document.add_paragraph("")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text, table.cell(1, 0).text = "Cap", "Kiem soat", "1"
    document.add_heading("Muc 2", level=2)
    document.save(path)


def write_pptx(path: Path) -> None:
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Slide Title A"
    slide.placeholders[1].text = "Bullet body"
    slide.notes_slide.notes_text_frame.text = "Speaker note text"
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(1)).table
    table.cell(0, 0).text, table.cell(0, 1).text, table.cell(1, 0).text = "H1", "H2", "v1"
    presentation.slides.add_slide(presentation.slide_layouts[6])  # blank slide
    presentation.save(path)


def write_xlsx(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Controls"
    sheet.append(["Level", None, "Owner"])
    sheet.append(["1", "", "=1+1"])
    workbook.create_sheet("Empty")
    workbook.save(path)


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)


# ---- 1. extraction -------------------------------------------------------------

class ExtractorCharacterizationTests(TempDirTestCase):
    def test_extract_pdf_uses_pymupdf_pages_headings_and_visual_regions(self) -> None:
        path = self.dir / "source.pdf"
        write_pdf(path)
        sections = extraction.extract_pdf(path)
        self.assertEqual([section.page for section in sections], [1, 2])  # blank page dropped
        first, second = sections
        self.assertTrue(first.text.startswith("Chapter One Overview\n\nBody text line"))
        self.assertEqual(first.metadata["heading_candidates"], [{"title": "Chapter One Overview", "level": 1}])
        self.assertEqual((first.metadata["extraction_version"], first.metadata["content_kinds"],
                          first.metadata["reading_order"]),
                         (STRUCTURED_EXTRACTION_VERSION, ["text"], "top_to_bottom_left_to_right"))
        self.assertNotIn("visual_regions", first.metadata)
        self.assertEqual(second.metadata["visual_region_count"], 1)
        self.assertEqual(second.metadata["visual_regions"][0]["region_kind"], "embedded_image")
        self.assertEqual(second.metadata["visual_prompt_text"], "Please observe the scenario?")

    def test_extract_pdf_page_limit_ocr_required_and_corrupt_file(self) -> None:
        path = self.dir / "pages.pdf"
        write_pdf(path, pages=3)
        with patch.object(main.settings, "max_document_pages", 2), self.assertRaises(DocumentLimitError):
            extraction.extract_pdf(path)
        blank = self.dir / "blank.pdf"
        write_pdf(blank, pages=0)
        with self.assertRaises(ValueError) as caught:
            extraction.extract_pdf(blank)  # pymupdf finds nothing, the pypdf fallback finds nothing
        self.assertTrue(str(caught.exception).startswith("OCR_REQUIRED:"))
        broken = self.dir / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4 not really a pdf")
        with self.assertRaises(pymupdf.FileDataError):  # RuntimeError subclass; no pypdf fallback
            extraction.extract_pdf(broken)
        # Characterized: the failed open keeps the file handle alive until GC; on Windows the
        # file cannot be deleted before a collection (see the corrupt-PDF index test below).
        gc.collect()

    def test_extract_pdf_falls_back_to_pypdf_when_pymupdf_is_unavailable(self) -> None:
        path = self.dir / "fallback.pdf"
        write_pdf(path, pages=2)
        with patch.dict(sys.modules, {"pymupdf": None}):  # makes `import pymupdf` raise ImportError
            sections = extraction.extract_pdf(path)
            self.assertEqual([(s.page, s.text) for s in sections], [(1, "Bounded page 1"), (2, "Bounded page 2")])
            self.assertEqual(sections[0].metadata, {})  # pypdf path carries no structured metadata
            with patch.object(main.settings, "max_document_pages", 1), self.assertRaises(DocumentLimitError):
                extraction.extract_pdf(path)

    def test_extract_docx_keeps_document_order_tables_and_heading_styles(self) -> None:
        path = self.dir / "guide.docx"
        write_docx(path)
        [section] = extraction.extract_docx(path)
        self.assertEqual(section.text, "Muc 1 Gioi thieu\n\nDoan van ban mo dau.\n\n[TABLE]\nRow 1: Cap | Kiem soat\n"
                                       "Row 2: 1 | \n\nMuc 2")
        self.assertIsNone(section.page)
        self.assertEqual(section.metadata, {
            "extraction_version": STRUCTURED_EXTRACTION_VERSION, "content_kinds": ["table", "text"],
            "table_count": 1, "reading_order": "document_order",
            "heading_candidates": [{"title": "Muc 1 Gioi thieu", "level": 1}, {"title": "Muc 2", "level": 2}],
        })

    def test_extract_pptx_slides_tables_notes_and_slide_limit(self) -> None:
        path = self.dir / "deck.pptx"
        write_pptx(path)
        first, second = extraction.extract_pptx(path)  # blank third slide dropped
        self.assertEqual([(s.page, s.section) for s in (first, second)], [(1, "Slide 1"), (2, "Slide 2")])
        self.assertEqual(first.text, "Slide Title A\n\nBullet body\n\n[SPEAKER NOTES]\nSpeaker note text")
        self.assertEqual((first.metadata["content_kinds"], first.metadata["notes_included"]),
                         (["speaker_notes", "text"], True))
        self.assertEqual(first.metadata["heading_candidates"], [{"title": "Slide Title A", "level": 1}])
        self.assertEqual(second.text, "[TABLE]\nRow 1: H1 | H2\nRow 2: v1 |")
        # Characterized: a table-only slide still reports "text" among its content kinds.
        self.assertEqual((second.metadata["content_kinds"], second.metadata["table_count"]), (["table", "text"], 1))
        with patch.object(main.settings, "max_document_pages", 2), self.assertRaises(DocumentLimitError):
            extraction.extract_pptx(path)  # 3 slides including the blank one

    def test_extract_xlsx_renders_coordinates_formula_text_and_cell_limit(self) -> None:
        path = self.dir / "sheet.xlsx"
        write_xlsx(path)
        [section] = extraction.extract_xlsx(path)  # empty sheet dropped
        # Characterized: data_only=False indexes formula text ("=1+1"), not computed values.
        self.assertEqual(section.text, "[TABLE]\nRow 1: A1=Level | C1=Owner\nRow 2: A2=1 | C2==1+1")
        self.assertEqual((section.section, section.page), ("Controls", None))
        self.assertEqual(section.metadata["reading_order"], "row_major_with_coordinates")
        self.assertEqual(section.metadata["heading_candidates"], [{"title": "Controls", "level": 1}])
        with patch.object(main.settings, "max_xlsx_cells", 5), self.assertRaises(DocumentLimitError):
            extraction.extract_xlsx(path)  # 2 rows x 3 cols = 6 cells

    def test_extract_xls_reads_xlrd_sheets_and_enforces_cell_limit(self) -> None:
        rows = [["Level", "", "Owner"], [1.0, "", "Supervisor"]]
        sheets = [SimpleNamespace(name="Levels", nrows=2, ncols=3, row_values=lambda index: list(rows[index])),
                  SimpleNamespace(name="Blank", nrows=0, ncols=0, row_values=lambda index: [])]
        path = self.dir / "legacy.xls"
        path.write_bytes(b"xls placeholder")
        with patch("xlrd.open_workbook", return_value=SimpleNamespace(sheets=lambda: sheets)) as opener:
            [section] = extraction.extract_xls(path)
            opener.assert_called_once_with(str(path))
            # Characterized: xlrd numeric cells are rendered as floats ("1.0").
            self.assertEqual(section.text, "[TABLE]\nRow 1: A1=Level | C1=Owner\nRow 2: A2=1.0 | C2=Supervisor")
            self.assertEqual((section.section, section.metadata["content_kinds"]), ("Levels", ["table"]))
            with patch.object(main.settings, "max_xlsx_cells", 5), self.assertRaises(DocumentLimitError):
                extraction.extract_xls(path)  # nrows * ncols = 6 is checked before reading values

    def test_extract_doc_converts_with_libreoffice_then_reads_docx(self) -> None:
        path = self.dir / "legacy.doc"
        path.write_bytes(b"\xd0\xcf\x11\xe0 legacy doc bytes")
        seen: dict[str, Any] = {}

        def convert(command: list[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[bytes]:
            outdir = Path(command[command.index("--outdir") + 1])
            seen.update(command=command, timeout=timeout_seconds, outdir=outdir)
            write_docx(outdir / f"{Path(command[-1]).stem}.docx")
            return subprocess.CompletedProcess(command, 0, b"", b"")

        with patch("app.services.ingestion.extract.shutil.which",
                   side_effect=lambda name: FAKE_SOFFICE if name == "soffice" else None), \
                patch("app.services.ingestion.extract.run_limited_subprocess", side_effect=convert):
            [section] = extraction.extract_doc(path)
        command = seen["command"]
        self.assertEqual(command[:2] + command[3:], [FAKE_SOFFICE, "--headless", "--convert-to", "docx", "--outdir",
                                                     str(seen["outdir"]), str(path)])
        self.assertTrue(command[2].startswith("-env:UserInstallation=file:"))
        self.assertEqual(seen["timeout"], main.settings.libreoffice_timeout_seconds)
        self.assertFalse(seen["outdir"].exists())  # conversion workspace is cleaned up
        self.assertTrue(section.text.startswith("Muc 1 Gioi thieu"))
        self.assertEqual(section.metadata["reading_order"], "document_order")

    def test_extract_doc_byte_fallback_and_conversion_failures(self) -> None:
        path = self.dir / "legacy.doc"
        readable = "Readable legacy body text. " * 10
        path.write_bytes(b"\x00\x01\x02" + readable.encode("ascii") + b"\x03\x7f")  # controls -> spaces
        no_output = Mock(return_value=None)
        with patch("app.services.ingestion.extract.shutil.which", return_value=FAKE_SOFFICE), \
                patch("app.services.ingestion.extract.run_limited_subprocess", new=no_output):
            [section] = extraction.extract_doc(path)  # converter produced nothing -> byte decode fallback
        no_output.assert_called_once()
        self.assertEqual((section.text, section.page, section.metadata), (readable.strip(), None, {}))
        with patch("app.services.ingestion.extract.shutil.which", return_value=None):
            self.assertEqual(extraction.extract_doc(path)[0].text, readable.strip())
            path.write_bytes(b"too short")
            with self.assertRaises(ValueError):
                extraction.extract_doc(path)
        for error in (RuntimeError("Document conversion failed."), DocumentLimitError("timeout")):
            with self.subTest(error=type(error).__name__), \
                    patch("app.services.ingestion.extract.shutil.which", return_value=FAKE_SOFFICE), \
                    patch("app.services.ingestion.extract.run_limited_subprocess", side_effect=error), \
                    self.assertRaises(type(error)):
                extraction.extract_doc(path)  # conversion errors propagate; no byte fallback

    def test_extract_plain_text_decoding(self) -> None:
        path = self.dir / "notes.txt"
        path.write_bytes(b"caf\xe9  \t latte\n\n\n\nend")
        # cp1258 fallback decodes 0xE9; runs of spaces/tabs and 3+ newlines collapse.
        self.assertEqual(extraction.extract_plain_text(path)[0].text, "café latte\n\nend")
        path.write_bytes("\ufeffBOM body".encode())
        # Characterized: utf-8 (not utf-8-sig) wins first, so the BOM survives clean_text.
        self.assertEqual(extraction.extract_plain_text(path)[0].text, "\ufeffBOM body")
        path.write_bytes(b"line one\r\nline two")
        # Characterized: CRLF line endings are not normalized.
        self.assertEqual(extraction.extract_plain_text(path)[0].text, "line one\r\nline two")


class ExtractSectionsRoutingTests(TempDirTestCase):
    def test_routes_by_lowercased_file_name_suffix(self) -> None:
        targets = {".pdf": "extract_pdf", ".docx": "extract_docx", ".doc": "extract_doc", ".pptx": "extract_pptx",
                   ".xlsx": "extract_xlsx", ".xls": "extract_xls"}
        extracted = [ExtractedSection(text="x"), ExtractedSection(text=" \n ")]
        for suffix, target in targets.items():
            path = self.dir / f"source{suffix}"
            path.write_bytes(b"placeholder")
            with self.subTest(suffix=suffix), patch("app.services.ingestion.extract.validate_ooxml_archive"), \
                    patch(f"app.services.ingestion.extract.{target}", return_value=extracted) as extractor:
                sections = extraction.extract_sections(path, f"SOURCE{suffix.upper()}")
                extractor.assert_called_once_with(path)
                self.assertEqual([s.text for s in sections], ["x"])  # blank sections dropped
        for suffix in (".txt", ".md", ".csv"):
            path = self.dir / f"plain{suffix}"
            path.write_bytes(b"a,b\n1,2")
            with self.subTest(suffix=suffix):
                self.assertEqual(extraction.extract_sections(path, path.name)[0].text, "a,b\n1,2")

    def test_unknown_suffix_empty_content_and_size_limit(self) -> None:
        path = self.dir / "payload.exe"
        path.write_bytes(b"MZ")
        for file_name, expected in (("payload.exe", ".exe"), ("README", "README")):
            with self.subTest(file_name=file_name), self.assertRaises(ValueError) as caught:
                extraction.extract_sections(path, file_name)
            self.assertIn(expected, str(caught.exception))
        empty = self.dir / "empty.txt"
        empty.write_bytes(b" \n\t ")
        with self.assertRaises(ValueError):
            extraction.extract_sections(empty, empty.name)
        big = self.dir / "big.txt"
        big.write_bytes(b"12345")
        with patch.object(main.settings, "max_document_bytes", 4), \
                patch("app.services.ingestion.extract.extract_plain_text") as extractor, \
                self.assertRaises(DocumentLimitError):
            extraction.extract_sections(big, big.name)
        extractor.assert_not_called()

    def test_ooxml_archive_guard_runs_on_the_routing_suffix(self) -> None:
        bogus = self.dir / "bogus.docx"
        bogus.write_bytes(b"not a zip archive")
        with self.assertRaises(DocumentLimitError) as caught:
            extraction.extract_sections(bogus, bogus.name)
        self.assertEqual(caught.exception.safe_message, "Document archive is invalid or unsafe.")
        real = self.dir / "real.docx"
        write_docx(real)
        with patch.object(main.settings, "max_ooxml_entries", 1), self.assertRaises(DocumentLimitError):
            extraction.extract_sections(real, real.name)
        # Changed 2026-10-08 (SEP-1, F2): the guard used the *path* suffix while routing used file_name,
        # so a mismatched pair reached the zip parser unguarded. Both now use the routing suffix.
        disguised = self.dir / "disguised.bin"
        disguised.write_bytes(real.read_bytes())
        with patch.object(main.settings, "max_ooxml_entries", 1), self.assertRaises(DocumentLimitError):
            extraction.extract_sections(disguised, "real.docx")
        # The reverse pair is routed to the plain-text reader, which never decompresses: no guard needed.
        with patch.object(main.settings, "max_ooxml_entries", 1):
            self.assertTrue(extraction.extract_sections(real, "real.txt")[0].text.startswith("PK"))

    def test_index_document_temp_path(self) -> None:
        root = self.dir.resolve()
        for name, suffix in ((None, ".txt"), ("A.PDF", ".pdf"), ("x.exe", ".txt"), ("../../y.docx", ".docx")):
            with self.subTest(name=name):
                self.assertEqual(extraction.index_document_temp_path(str(self.dir), DOC_ID, name),
                                 root / f"{DOC_ID}{suffix}")
        with self.assertRaises(ValueError):
            extraction.index_document_temp_path(str(self.dir), "../not-a-uuid", "x.pdf")


# ---- 2. index_document ---------------------------------------------------------

class IndexDocumentCharacterizationTests(unittest.TestCase):
    def run_index(self, db: FakeDb, *, request: RagIndexRequest | None = None,
                  embedder: FakeEmbedder | None = None,
                  download: bytes | None = None) -> tuple[Any, FakeEmbedder, Mock, list[Any]]:
        """Run the endpoint coroutine; a raised exception is returned instead of propagating."""
        embedder = embedder or FakeEmbedder()
        downloader = Mock(return_value=download, side_effect=None if download is not None
                          else AssertionError("download not expected"))
        with patch("app.services.provider.embed_texts", new=embedder), \
                patch("app.services.ingestion.storage.download_storage_object", new=downloader), \
                self.assertLogs(main.logger, level="INFO") as logs:
            try:
                outcome: Any = asyncio.run(index_service.index_document(request or index_request(), pool=db))
            except Exception as error:
                outcome = error
        return outcome, embedder, downloader, logs.records

    def assert_failed(self, db: FakeDb, result: Any, reason: str) -> None:
        self.assertEqual(result, {"status": "error", "chunk_count": 0, "usage": AiUsage().model_dump(),
                                  "error_reason": reason})
        self.assertEqual(db.args_for("mark_index_error"), [(INDEX_ID, reason)])
        self.assertEqual(db.args_for("insert_chunk"), [])

    def test_content_only_document_is_learned_with_structure_table_missing(self) -> None:
        db = FakeDb(document=document_row(content=THREE_PARAGRAPHS), structure_table=False)
        with patch.object(main.settings, "chunk_max_chars", 200), patch.object(main.settings, "chunk_overlap_chars", 0):
            result, embedder, downloader, records = self.run_index(db)
        downloader.assert_not_called()
        # Changed 2026-10-08 (SEP-1 #7): the three chunk rows go in one executemany (same SQL, same
        # columns, values and order) inside the same transaction instead of one execute per chunk.
        self.assertEqual(db.labels(), [
            "load_document", "supersede_running_indexes", "next_index_version", "insert_index_row",
            "lock_index_row", "insert_chunk", "insert_structure_nodes",
            "delete_structure_nodes", "deactivate_other_indexes", "activate_index", "delete_indexes",
        ])
        self.assertEqual([method for method, sql, _ in db.calls if "INSERT INTO rag_chunks" in sql], ["executemany"])
        # Savepoint for the optional structure table rolls back; the outer index transaction commits.
        self.assertEqual(db.events, [("begin", 1), ("commit", 1), ("begin", 1), ("begin", 2), ("rollback", 2),
                                     ("commit", 1)])
        self.assertEqual(db.acquired, 2)
        self.assertEqual(db.args_for("load_document"), [(DOC_ID, TENANT_ID, KB_ID)])
        self.assertEqual(db.args_for("supersede_running_indexes"), [(DOC_ID,)])
        self.assertEqual(db.args_for("insert_index_row"), [(TENANT_ID, KB_ID, DOC_ID, 3, "gemini-embedding-001")])
        self.assertEqual(db.args_for("lock_index_row"), [(INDEX_ID, DOC_ID)])
        [(chunk_args,)] = db.args_for("insert_chunk")
        self.assertEqual([args[:5] for args in chunk_args], [(TENANT_ID, KB_ID, DOC_ID, INDEX_ID, n) for n in range(3)])
        contents = [args[5] for args in chunk_args]
        self.assertTrue(all(content.startswith(f"Paragraph {n + 1}:") for n, content in enumerate(contents)))
        self.assertEqual([args[6] for args in chunk_args], [hashlib.sha256(c.encode()).hexdigest() for c in contents])
        self.assertEqual([(args[8], args[9]) for args in chunk_args], [(None, None)] * 3)
        metadata = json.loads(chunk_args[0][10])
        self.assertEqual((metadata["source_name"], metadata["document_type"]), ("Safety notes", "text"))
        self.assertEqual(metadata["structured_evidence_contract_version"], SOURCE_EVIDENCE_PROPAGATION_VERSION)
        self.assertTrue(chunk_args[0][11].startswith("[0.50000000,-0.25000000,0.00000000,"))
        content_sha = hashlib.sha256("\n\n".join(contents).encode()).hexdigest()
        self.assertEqual(db.args_for("activate_index"), [(INDEX_ID, DOC_ID, content_sha, 3)])
        self.assertEqual(db.args_for("deactivate_other_indexes"), [(DOC_ID, INDEX_ID)])
        self.assertEqual(db.args_for("delete_indexes"), [(DOC_ID, INDEX_ID)])
        self.assertEqual((result["status"], result["chunk_count"]), ("learned", 3))
        self.assertEqual((result["structure_source"], result["structure_node_count"], result["structure_confidence"]),
                         ("semantic_inferred", 1, 0.35))
        self.assertEqual(result["usage"], {"inputTokens": 0, "outputTokens": 0, "embeddingTokens": 21,
                                           "totalTokens": 21})
        diagnostics = result["diagnostics"]
        self.assertEqual((diagnostics["file_name"], diagnostics["raw_bytes"]),
                         ("Safety notes", len(THREE_PARAGRAPHS.encode())))
        self.assertEqual((diagnostics["chunk_count"], diagnostics["warnings"]), (3, []))
        [call] = embedder.calls
        self.assertEqual(provider_service.require_provider_api_key(call["api_key"]), API_KEY)
        self.assertEqual((call["model"], call["task_type"], call["output_dimensionality"], call["contents"]),
                         ("gemini-embedding-001", "RETRIEVAL_DOCUMENT", 768, contents))
        self.assertEqual([getattr(r, "event", None) or r.getMessage() for r in records], [
            "rag_index_started", "rag_index_extracted", "rag_index_chunked", "rag_index_embedded",
            "rag_document_structure_nodes is not deployed; using chunk metadata fallback",
            "rag_document_structure_nodes is not deployed; skipped old structure cleanup",
            "rag_index_completed",
        ])

    def test_file_document_downloads_to_temp_path_and_persists_structure_nodes(self) -> None:
        storage_path = f"{TENANT_ID}/{KB_ID}/guide.docx"
        with tempfile.TemporaryDirectory() as directory:
            write_docx(Path(directory) / "guide.docx")
            raw = (Path(directory) / "guide.docx").read_bytes()
        db = FakeDb(document=document_row(name="guide.docx", type="docx", file_path=storage_path, content=None))
        spy = Mock(wraps=extraction.extract_sections)
        with patch("app.services.ingestion.extract.extract_sections", new=spy):
            result, _, downloader, records = self.run_index(db, download=raw)
        downloader.assert_called_once_with(storage_path)
        temp_path, file_name = spy.call_args.args
        self.assertEqual((temp_path.name, file_name), (f"{DOC_ID}.docx", f"{DOC_ID}.docx"))
        self.assertFalse(temp_path.exists())
        self.assertEqual((result["status"], result["structure_source"], result["structure_confidence"]),
                         ("learned", "heading_inferred", 0.9))
        self.assertEqual((result["diagnostics"]["raw_bytes"], result["diagnostics"]["extracted_heading_count"]),
                         (len(raw), 2))
        [(node_rows,)] = db.args_for("insert_structure_nodes")
        self.assertEqual(len(node_rows), result["structure_node_count"])
        self.assertEqual(node_rows[0][:9], (TENANT_ID, KB_ID, DOC_ID, INDEX_ID, "src-001", None, 1, "chapter",
                                            "Muc 1 Gioi thieu"))
        self.assertEqual(node_rows[1][4:7], ("src-002", "src-001", 2))
        self.assertEqual(node_rows[0][14], PARSER_VERSION)
        self.assertEqual(json.loads(node_rows[0][15])["structure_source"], "heading_inferred")
        self.assertEqual(db.args_for("delete_structure_nodes"), [(TENANT_ID, KB_ID, DOC_ID, INDEX_ID)])
        self.assertEqual(db.events, [("begin", 1), ("commit", 1), ("begin", 1), ("begin", 2), ("commit", 2),
                                     ("commit", 1)])
        self.assertIn("rag_index_downloaded", [getattr(r, "event", None) for r in records])

    def test_unsupported_suffix_is_rejected_before_any_index_row(self) -> None:
        # Changed 2026-10-08 (SEP-1, F1): an unknown suffix used to be indexed as ".txt". The extractor is
        # now chosen from the stored object's extension (kb_documents.file_path) and anything without an
        # extractor is a 422 DOCUMENT_TYPE_UNSUPPORTED: nothing is downloaded and no index row is created.
        for file_path in (f"{TENANT_ID}/{KB_ID}/payload.exe", f"{TENANT_ID}/{KB_ID}/no-extension"):
            with self.subTest(file_path=file_path):
                db = FakeDb(document=document_row(name="payload.exe", file_path=file_path))
                outcome, embedder, downloader, _ = self.run_index(db)
                self.assertIsInstance(outcome, AppError)
                self.assertEqual((outcome.code, outcome.http_status), ("DOCUMENT_TYPE_UNSUPPORTED", 422))
                downloader.assert_not_called()
                self.assertEqual((db.labels(), embedder.calls), (["load_document"], []))

    def test_extension_comes_from_the_storage_path_and_legacy_model_is_normalized(self) -> None:
        # A KB article's display name has no extension; its stored ".md" object is still routed correctly.
        db = FakeDb(document=document_row(name="Article title", file_path=f"{TENANT_ID}/kb-articles/a-1.md"))
        result, embedder, _, _ = self.run_index(db, request=index_request(embedding_model=" text-embedding-004 "),
                                                download=b"# Title\n\nPlain markdown body")
        self.assertEqual(result["status"], "learned")
        [(chunk_rows,)] = db.args_for("insert_chunk")
        self.assertEqual(chunk_rows[0][5], "# Title\n\nPlain markdown body")
        self.assertEqual(db.args_for("insert_index_row")[0][4], "gemini-embedding-001")  # legacy alias mapped
        self.assertEqual(embedder.calls[0]["model"], "gemini-embedding-001")

    def test_document_limit_errors_propagate_and_only_parse_time_limits_touch_the_index(self) -> None:
        # Changed 2026-10-08 (SEP-1, F3): ownership and size are checked before the index row is created,
        # so these rejections no longer supersede the running index. Only a limit found while parsing the
        # downloaded file (the OOXML guard) still marks the row it created.
        cases = {
            "foreign_tenant_path": (document_row(file_path=f"{OTHER_TENANT_ID}/{KB_ID}/x.txt"), None),
            "traversal_path": (document_row(file_path=f"{TENANT_ID}/../{OTHER_TENANT_ID}/x.txt"), None),
            "download_too_large": (document_row(file_path=f"{TENANT_ID}/{KB_ID}/x.txt"), b"x" * 2048),
            "content_too_large": (document_row(content="y" * 2048), None),
            "unsafe_ooxml": (document_row(name="x.docx", file_path=f"{TENANT_ID}/{KB_ID}/x.docx"), b"not zip"),
        }
        for name, (row, download) in cases.items():
            with self.subTest(case=name), patch.object(main.settings, "max_document_bytes", 1024):
                db = FakeDb(document=row)
                outcome, embedder, downloader, records = self.run_index(db, download=download)
                self.assertIsInstance(outcome, DocumentLimitError)
                self.assertEqual(outcome.code, "DOCUMENT_LIMIT_EXCEEDED")
                parsed = name == "unsafe_ooxml"
                self.assertEqual(len(db.args_for("insert_index_row")), int(parsed))
                expected_marks = [(INDEX_ID, "DOCUMENT_LIMIT_EXCEEDED")] if parsed else []
                self.assertEqual(db.args_for("mark_index_error"), expected_marks)
                self.assertEqual(embedder.calls, [])
                self.assertEqual(downloader.called, name in {"download_too_large", "unsafe_ooxml"})
                failed = [r for r in records if getattr(r, "event", None) == "rag_index_failed"]
                self.assertEqual(failed[0].error_code, "DOCUMENT_LIMIT_EXCEEDED")

    def test_embedding_count_mismatch_returns_error_payload(self) -> None:
        db = FakeDb(document=document_row())
        result, embedder, _, records = self.run_index(db, embedder=FakeEmbedder(drop=1))
        self.assert_failed(db, result, "INDEX_DOCUMENT_FAILED")
        self.assertEqual(len(embedder.calls), 1)
        self.assertNotIn("lock_index_row", db.labels())
        self.assertEqual(db.events, [("begin", 1), ("commit", 1)])
        self.assertEqual((records[-1].event, records[-1].error_type), ("rag_index_failed", "ValueError"))

    def test_superseded_index_rolls_back_and_returns_error_payload(self) -> None:
        for status in ("error", None):
            with self.subTest(status=status):
                db = FakeDb(document=document_row(), index_status=status)
                result, _, _, _ = self.run_index(db)
                self.assert_failed(db, result, "INDEX_DOCUMENT_FAILED")
                self.assertEqual(db.events, [("begin", 1), ("commit", 1), ("begin", 1), ("rollback", 1)])
                self.assertEqual(db.labels()[-2:], ["lock_index_row", "mark_index_error"])

    def test_non_retryable_failures_return_error_payload_instead_of_raising(self) -> None:
        app_error = AppError(code="CUSTOM_APP_ERROR", http_status=409, safe_message="custom")
        provider_rejected = HTTPException(status_code=400, detail={"code": "AI_PROVIDER_REQUEST_REJECTED"})
        for name, error, reason in (("app_error_code_kept", app_error, "CUSTOM_APP_ERROR"),
                                    ("provider_4xx", provider_rejected, "INDEX_DOCUMENT_FAILED")):
            with self.subTest(case=name):
                db = FakeDb(document=document_row())
                result, used, _, _ = self.run_index(db, embedder=FakeEmbedder(error=error))
                self.assert_failed(db, result, reason)
                self.assertEqual(len(used.calls), 1)

    def test_transient_provider_failures_raise_a_retryable_503_with_the_provider_code(self) -> None:
        # Changed 2026-10-08 (SEP-1, F5): a provider 503/429/504 during embedding used to become HTTP 200 +
        # status "error" (INDEX_DOCUMENT_FAILED), so the backend could not tell it was retryable. The index
        # row is still marked failed (with the provider code) and the route answers 503 with that code.
        cases = {
            "unavailable": (HTTPException(503, detail={"code": "AI_PROVIDER_UNAVAILABLE", "message": "down"}),
                            "AI_PROVIDER_UNAVAILABLE"),
            "rate_limited": (HTTPException(503, detail={"code": "AI_PROVIDER_RATE_LIMITED"}),
                             "AI_PROVIDER_RATE_LIMITED"),
            "quota": (HTTPException(503, detail={"code": "AI_PROVIDER_QUOTA_EXHAUSTED"}),
                      "AI_PROVIDER_QUOTA_EXHAUSTED"),
            "timeout": (HTTPException(504, detail={"code": "AI_PROVIDER_TIMEOUT"}), "AI_PROVIDER_TIMEOUT"),
            "raw_429_without_code": (HTTPException(429, detail="slow down"), "AI_PROVIDER_UNAVAILABLE"),
            "service_busy": (AppError(code="SERVICE_BUSY", http_status=503, safe_message="busy"), "SERVICE_BUSY"),
        }
        for name, (error, code) in cases.items():
            with self.subTest(case=name):
                db = FakeDb(document=document_row())
                outcome, used, _, records = self.run_index(db, embedder=FakeEmbedder(error=error))
                self.assertIsInstance(outcome, AppError)
                self.assertEqual((outcome.code, outcome.http_status), (code, 503))
                self.assertEqual(db.args_for("mark_index_error"), [(INDEX_ID, code)])
                self.assertEqual((db.args_for("insert_chunk"), len(used.calls)), ([], 1))
                self.assertEqual(records[-1].error_code, code)
                if name == "unavailable":  # the provider's own safe message is kept
                    self.assertEqual(outcome.safe_message, "down")

    def test_corrupt_pdf_upload_returns_error_payload(self) -> None:
        # SEC-7: PDFs are parsed from memory, so a failed open never pins the temp file.
        # The real parser error is logged and no copy of the upload stays on disk (all OSes).
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch, \
                patch.object(tempfile, "tempdir", scratch):
            db = FakeDb(document=document_row(name="broken.pdf", file_path=f"{TENANT_ID}/{KB_ID}/broken.pdf"))
            result, embedder, _, records = self.run_index(db, download=b"%PDF-1.4 garbage")
            gc.collect()
            leftovers = list(Path(scratch).iterdir())
        self.assert_failed(db, result, "INDEX_DOCUMENT_FAILED")
        self.assertEqual(embedder.calls, [])
        self.assertEqual(records[-1].error_type, "FileDataError")
        self.assertEqual(leftovers, [])

    def test_pre_index_http_errors_touch_no_index_rows(self) -> None:
        self.assertNotIn(API_KEY, repr(index_request()))  # api_key is a SecretStr
        # Changed 2026-10-08 (SEP-1, F4): whitespace-only content is a 400 here; it used to create an index
        # row and only then fail with INDEX_DOCUMENT_FAILED.
        cases = ((index_request(embedding_dimensions=1536), document_row(), 400, []),
                 (index_request(), None, 404, ["load_document"]),
                 (index_request(), document_row(content=""), 400, ["load_document"]),
                 (index_request(), document_row(content=None), 400, ["load_document"]),
                 (index_request(), document_row(content="  \n\t "), 400, ["load_document"]))
        for request, row, status, labels in cases:
            with self.subTest(status=status, row=row):
                db = FakeDb(document=row)
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(index_service.index_document(request, pool=db))
                self.assertEqual((caught.exception.status_code, db.labels()), (status, labels))


# ---- 3/4. persistence helpers, deletion and diagnostics -------------------------

class PersistenceHelperTests(unittest.TestCase):
    def test_load_document_404_and_start_index_row(self) -> None:
        # PRD-2: the repository returns None for a missing document and the index service answers
        # the 404 (pinned end to end by test_pre_index_http_errors_touch_no_index_rows).
        db = FakeDb(document=None)
        self.assertIsNone(asyncio.run(index_repository.load_document(db, TENANT_ID, KB_ID, DOC_ID)))
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(index_service.index_document(index_request(), pool=FakeDb(document=None)))
        self.assertEqual(caught.exception.status_code, 404)
        [(method, sql, args)] = db.calls
        self.assertEqual((method, args), ("fetchrow", (DOC_ID, TENANT_ID, KB_ID)))
        self.assertIn("WHERE id = $1::uuid AND tenant_id = $2::uuid AND kb_id = $3::uuid", sql)
        db = FakeDb(next_version=7)
        self.assertEqual(asyncio.run(index_repository.start_index_row(db, document_row(), "model-x")), INDEX_ID)
        self.assertEqual(db.labels(), ["supersede_running_indexes", "next_index_version", "insert_index_row"])
        self.assertEqual(db.args_for("insert_index_row"), [(TENANT_ID, KB_ID, DOC_ID, 7, "model-x")])
        self.assertIn("'running', false, $5, 768, now()", db.calls[2][1])  # dimensions are hard-coded
        self.assertEqual(db.events, [("begin", 1), ("commit", 1)])

    def test_mark_index_error_noop_without_index_and_truncates_reason(self) -> None:
        db = FakeDb()
        asyncio.run(index_repository.mark_index_error(db, None, "ignored"))
        asyncio.run(index_repository.mark_index_error(db, "", "ignored"))
        self.assertEqual(db.calls, [])
        asyncio.run(index_repository.mark_index_error(db, INDEX_ID, "r" * 1500))
        [(method, sql, args)] = db.calls
        self.assertEqual((method, args[0], len(args[1])), ("execute", INDEX_ID, 1000))
        self.assertIn("WHERE id = $1::uuid AND status = 'running'", sql)

    def test_structure_node_persistence_and_cleanup_variants(self) -> None:
        structure = {"nodes": [{"source_ref": "src-001", "title": "Root", "level": 0, "order": 0, "confidence": 0},
                               "not-a-node"], "confidence": 0.8, "structure_source": "toc"}

        def persist(db: FakeDb, value: dict[str, Any]) -> bool:
            return asyncio.run(index_repository.persist_structure_nodes_if_available(db, document_row(), INDEX_ID,
                                                                                     value))

        db = FakeDb()
        self.assertTrue(persist(db, structure))
        [(rows,)] = db.args_for("insert_structure_nodes")
        # level 0 -> 1, missing node_type -> "section", confidence 0 -> structure confidence,
        # missing parser_version -> "source-structure-v2" (older than main.PARSER_VERSION).
        metadata = json.dumps({"structure_source": "toc", "parser_version": None, "warnings": [],
                               "chapter_authority": None})
        self.assertEqual(rows, [(TENANT_ID, KB_ID, DOC_ID, INDEX_ID, "src-001", None, 1, "section", "Root", None,
                                 None, None, 0, 0.8, "source-structure-v2", metadata)])
        self.assertEqual(db.events, [("begin", 1), ("commit", 1)])
        empty = FakeDb()
        self.assertFalse(persist(empty, {"nodes": ["x"]}))
        self.assertEqual((empty.calls, empty.events), ([], []))
        missing = FakeDb(structure_table=False)
        self.assertFalse(persist(missing, structure))
        self.assertEqual(missing.events, [("begin", 1), ("rollback", 1)])
        # delete_previous_structure_nodes_if_available: tolerated when missing, other errors propagate.
        for table_present in (True, False):
            with self.subTest(table_present=table_present):
                db = FakeDb(structure_table=table_present)
                self.assertIsNone(asyncio.run(index_repository.delete_previous_structure_nodes_if_available(
                    db, document_row(), INDEX_ID)))
                [(_, sql, args)] = db.calls
                self.assertIn("AND index_id <> $4::uuid", sql)
                self.assertEqual(args, (TENANT_ID, KB_ID, DOC_ID, INDEX_ID))
        db = FakeDb(errors={"DELETE FROM rag_document_structure_nodes":
                            asyncpg.exceptions.InsufficientPrivilegeError("denied")})
        with self.assertRaises(asyncpg.exceptions.InsufficientPrivilegeError):
            asyncio.run(index_repository.delete_previous_structure_nodes_if_available(db, document_row(), INDEX_ID))

    def test_build_index_diagnostics_warnings(self) -> None:
        empty = chunking.build_index_diagnostics([], [], raw_bytes=0, file_name="deck.PPTX")
        # Characterized: the PPTX warning says speaker notes are not extracted, but extract_pptx does.
        self.assertEqual(empty["warnings"], [
            "PPTX_TEXT_SHAPES_ONLY; CHARTS_IMAGES_AND_SPEAKER_NOTES_ARE_NOT_EXTRACTED",
            "EXTRACTION_EMPTY", "INDEX_CONTENT_EMPTY", "STRUCTURED_EVIDENCE_REVISION_INCONSISTENT",
        ])
        sections = [ExtractedSection(text="Same paragraph.", page=2),
                    ExtractedSection(text="Same paragraph.", page=1)]
        chunks = chunking.build_chunks(sections)
        diagnostics = chunking.build_index_diagnostics(sections, chunks, raw_bytes=10, file_name="a.pdf")
        self.assertEqual(diagnostics["warnings"], ["PDF_TEXT_LAYER_ONLY; OCR_OR_EMBEDDED_IMAGE_TEXT_IS_NOT_EXTRACTED",
                                                   "DUPLICATE_CHUNKS_DEDUPLICATED"])
        self.assertEqual((diagnostics["chunk_count"], diagnostics["candidate_chunk_count"],
                          diagnostics["deduplicated_chunk_count"], diagnostics["extracted_pages"]), (1, 2, 1, [1, 2]))
        self.assertEqual(diagnostics["source_evidence_revision"], chunks[0]["metadata"]["source_evidence_revision"])


class DeleteEndpointTests(unittest.TestCase):
    def test_delete_document_and_kb_scope_by_tenant(self) -> None:
        document_request = RagDeleteDocumentRequest(tenant_id=TENANT_ID, kb_id=KB_ID, document_id=DOC_ID)
        kb_request = RagDeleteKbRequest(tenant_id=TENANT_ID, kb_id=KB_ID)
        with self.assertRaises(ValidationError):  # ids are validated as UUIDs before any SQL runs
            RagDeleteKbRequest(tenant_id="tenant-a", kb_id=KB_ID)
        cases = (
            ("document", lambda db: documents_service.delete_document(document_request, pool=db), (TENANT_ID, KB_ID,
                                                                                                   DOC_ID)),
            ("kb", lambda db: documents_service.delete_kb(kb_request, pool=db), (TENANT_ID, KB_ID)),
        )
        for name, call, expected_args in cases:
            for table_present in (True, False):
                with self.subTest(target=name, table_present=table_present):
                    db = FakeDb(structure_table=table_present)
                    self.assertEqual(asyncio.run(call(db)), {"deleted": True})
                    self.assertEqual(db.labels(), ["delete_structure_nodes", "delete_indexes"])
                    self.assertEqual([(method, args) for method, _, args in db.calls],
                                     [("execute", expected_args), ("execute", expected_args)])
                    sqls = [sql for _, sql, _ in db.calls]
                    self.assertTrue(all("WHERE tenant_id = $1::uuid AND kb_id = $2::uuid" in sql for sql in sqls))
                    self.assertEqual(name == "document", all("document_id = $3::uuid" in sql for sql in sqls))
                    # Characterized: rag_chunks are never deleted explicitly (presumably FK cascade).
                    self.assertFalse(any("rag_chunks" in sql for sql in sqls))
            with self.subTest(target=name, error="other_postgres_error"):
                db = FakeDb(errors={"DELETE FROM rag_document_structure_nodes":
                                    asyncpg.exceptions.InsufficientPrivilegeError("denied")})
                with self.assertRaises(asyncpg.exceptions.InsufficientPrivilegeError):
                    asyncio.run(call(db))
                self.assertEqual(db.labels(), ["delete_structure_nodes"])


class SourceFactTableOfContentsTests(unittest.TestCase):
    """Facts are re-extracted from stored chunk text for every source snapshot (and legacy coverage manifest).

    Changed 2026-10-08 (QC course 234653, D7): a chunk that mentioned "table of contents" was dropped
    whole, so a TOC page lost its positioning statement, target learners and expected competencies.
    Now only the TOC title, its entries and bare chapter labels are dropped. Re-indexing is not needed:
    the next snapshot of an indexed document already gets the kept lines.
    """

    PAGE = (
        "MỤC LỤC & ĐỊNH VỊ CHƯƠNG TRÌNH\n"
        "TUYÊN BỐ ĐỊNH VỊ CỐT LÕI (POSITIONING STATEMENT)\n"
        "MODUN 1 là cánh cổng thay đổi hệ điều hành tư duy của người lãnh đạo.\n"
        "Đối tượng học viên: CEO, Founder, thành viên HĐQT và C-level của doanh nghiệp SME Việt Nam.\n"
        "Năng lực 1: Nhận diện bốn lực đẩy buộc doanh nghiệp phải thay đổi.\n"
        "CẤU TRÚC KHOÁ HỌC (TABLE OF CONTENTS)\n"
        "1. Why change?........ 3\n"
        "2. The BIC legacy........ 4\n"
        "CHƯƠNG 03\n"
        "The best-in-class house\n"
        "Hiểu kiến trúc ngôi nhà năng lực tám cấu phần.\n"
        "CHƯƠNG 04\n"
    )

    def test_toc_lines_are_dropped_and_the_rest_of_the_page_is_kept(self) -> None:
        facts = source_coverage.extract_source_coverage_facts(self.PAGE)
        self.assertIn("MODUN 1 là cánh cổng thay đổi hệ điều hành tư duy của người lãnh đạo.", facts)
        self.assertTrue(any(fact.startswith("Đối tượng học viên: CEO") for fact in facts))
        self.assertIn("Năng lực 1: Nhận diện bốn lực đẩy buộc doanh nghiệp phải thay đổi.", facts)
        self.assertIn("Hiểu kiến trúc ngôi nhà năng lực tám cấu phần.", facts)
        joined = "\n".join(facts)
        for dropped in ("TABLE OF CONTENTS", "MỤC LỤC", "Why change?", "........", "CHƯƠNG 03", "CHƯƠNG 04"):
            self.assertNotIn(dropped, joined)

    def test_a_page_that_is_only_a_table_of_contents_still_yields_no_facts(self) -> None:
        self.assertEqual(source_coverage.extract_source_coverage_facts(
            "TABLE OF CONTENTS\n1. Introduction........ 3\n2. Safety rules........ 7\nChapter 3 Practice 12"), [])
        self.assertEqual(source_coverage.extract_source_coverage_facts(
            "Nội dung chương trình\nChương 1: Tổng quan\nChương 2: Thực hành\nPhần II"), [])

    def test_a_sentence_that_mentions_the_program_content_is_content(self) -> None:
        # Previously the whole chunk was dropped because it contained "nội dung chương trình".
        text = "Nội dung chương trình được thiết kế cho nhân viên mới.\n1. Rửa tay trước khi vào ca làm việc."
        self.assertEqual(source_coverage.extract_source_coverage_facts(text),
                         ["Nội dung chương trình được thiết kế cho nhân viên mới.",
                          "1. Rửa tay trước khi vào ca làm việc."])
        self.assertEqual(source_coverage.extract_source_coverage_facts(
                             "1. Rửa tay.\n2. Đeo găng tay trước khi làm việc."),
                         ["1. Rửa tay.", "2. Đeo găng tay trước khi làm việc."])

    def test_chunks_without_the_previous_trigger_phrases_are_unchanged(self) -> None:
        # "Mục lục" alone never dropped a chunk; it is only recognised as a TOC title inside a triggered chunk.
        self.assertEqual(source_coverage.extract_source_coverage_facts("MỤC LỤC\nChương 1: Tổng quan an toàn lao động"),
                         ["MỤC LỤC", "Chương 1: Tổng quan an toàn lao động"])


if __name__ == "__main__":
    unittest.main()
