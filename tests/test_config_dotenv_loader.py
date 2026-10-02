from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.config import load_runtime_dotenv


class RuntimeDotenvLoaderTests(unittest.TestCase):
    def test_local_loader_sanitizes_nul_bytes_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_bytes(b"AI_RAG_TEST_NUL=local\x00-value\n")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AI_RAG_TEST_NUL", None)
                self.assertTrue(load_runtime_dotenv(path, allow_nul_sanitization=True))
                self.assertEqual(os.environ.get("AI_RAG_TEST_NUL"), "local-value")
                os.environ.pop("AI_RAG_TEST_NUL", None)

    def test_production_style_loader_fails_closed_on_nul_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env.production"
            path.write_bytes(b"AI_RAG_TEST_NUL=invalid\x00-value\n")
            with self.assertRaisesRegex(ValueError, "embedded null character"):
                load_runtime_dotenv(path, allow_nul_sanitization=False)


if __name__ == "__main__":
    unittest.main()
