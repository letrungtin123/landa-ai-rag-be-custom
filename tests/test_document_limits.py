from __future__ import annotations

import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import fitz

from app.core.document_limits import (
    assert_document_size,
    assert_tenant_storage_path,
    run_limited_subprocess,
    validate_ooxml_archive,
)
from app.core.errors import DocumentLimitError
from app.main import extract_pdf, settings


class DocumentLimitTests(unittest.TestCase):
    def test_document_size_limit(self) -> None:
        assert_document_size(10, maximum=10)
        with self.assertRaises(DocumentLimitError):
            assert_document_size(11, maximum=10)

    def test_storage_path_must_belong_to_request_tenant(self) -> None:
        assert_tenant_storage_path("tenant-a/kb/document.pdf", "tenant-a")
        for unsafe in ("tenant-b/kb/document.pdf", "tenant-a/../tenant-b/file.pdf", "https://host/file"):
            with self.subTest(unsafe=unsafe), self.assertRaises(DocumentLimitError):
                assert_tenant_storage_path(unsafe, "tenant-a")

    def test_ooxml_zip_bomb_ratio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bomb.docx"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("word/document.xml", b"A" * 100_000)
            with self.assertRaises(DocumentLimitError):
                validate_ooxml_archive(
                    path,
                    max_uncompressed_bytes=200_000,
                    max_entries=10,
                    max_compression_ratio=10,
                )

    def test_pdf_page_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pages.pdf"
            document = fitz.open()
            for _ in range(3):
                document.new_page().insert_text((72, 72), "Bounded page")
            document.save(path)
            document.close()
            original = settings.max_document_pages
            settings.max_document_pages = 2
            try:
                with self.assertRaises(DocumentLimitError):
                    extract_pdf(path)
            finally:
                settings.max_document_pages = original

    def test_subprocess_timeout_kills_process_tree(self) -> None:
        process = Mock()
        process.pid = 1234
        process.communicate.side_effect = subprocess.TimeoutExpired(["soffice"], 1)
        process.wait.return_value = -9
        with (
            patch("app.core.document_limits.subprocess.Popen", return_value=process),
            patch("app.core.document_limits.subprocess.run") as kill_tree,
            self.assertRaises(DocumentLimitError),
        ):
            run_limited_subprocess(["soffice", "--headless"], timeout_seconds=1)
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        if kill_tree.called:
            self.assertIn("taskkill", kill_tree.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
