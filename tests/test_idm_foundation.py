from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app import main


class IdmFoundationTests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_errors_never_echo_request_secrets(self) -> None:
        secret = "idm0-provider-secret-must-not-echo"
        routes = (
            "/v1/chat",
            "/v1/kb/documents/index",
            "/v1/lesson-author/proposal",
            "/v1/lesson-author/blueprint",
        )
        main.app.dependency_overrides[main.require_internal_token] = lambda: None
        main.app.dependency_overrides[main.get_db] = lambda: None
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test",
            ) as client:
                for route in routes:
                    with self.subTest(route=route):
                        response = await client.post(route, json={
                            "api_key": secret,
                            "private_body": "private-source-content",
                        })
                        self.assertEqual(response.status_code, 422)
                        self.assertNotIn(secret, response.text)
                        self.assertNotIn("private-source-content", response.text)
        finally:
            main.app.dependency_overrides.pop(main.require_internal_token, None)
            main.app.dependency_overrides.pop(main.get_db, None)

    async def test_index_temp_path_uses_document_id_and_safe_extractor_suffix(self) -> None:
        document_id = "00000000-0000-4000-8000-000000000001"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            for original_name, expected_suffix in (
                ("../../x.py", ".txt"),
                ("C:/x/y.pdf", ".pdf"),
                ("/abs/source.docx", ".docx"),
            ):
                with self.subTest(original_name=original_name):
                    path = main.index_document_temp_path(temp_dir, document_id, original_name)
                    self.assertEqual(path.parent, root)
                    self.assertEqual(path.name, f"{document_id}{expected_suffix}")

            pdf_path = main.index_document_temp_path(temp_dir, document_id, "../../source.pdf")
            pdf_path.write_bytes(b"offline fixture")
            with patch("app.main.extract_pdf", return_value=[main.ExtractedSection(text="parsed")]) as extractor:
                sections = main.extract_sections(pdf_path, pdf_path.name)
            extractor.assert_called_once_with(pdf_path)
            self.assertEqual(sections[0].text, "parsed")


if __name__ == "__main__":
    unittest.main()
