from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.config import load_runtime_dotenv, load_runtime_environment


class RuntimeDotenvLoaderTests(unittest.TestCase):
    def test_loader_rejects_nul_sanitization_compatibility_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_bytes(b"AI_RAG_TEST_NUL=local\x00-value\n")
            with self.assertRaisesRegex(RuntimeError, "NUL sanitization"):
                load_runtime_dotenv(path, allow_nul_sanitization=True)

    def test_production_style_loader_fails_closed_on_nul_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env.production"
            path.write_bytes(b"AI_RAG_TEST_NUL=invalid\x00-value\n")
            with self.assertRaisesRegex(ValueError, "embedded null character"):
                load_runtime_dotenv(path, allow_nul_sanitization=False)

    def test_production_ignores_all_dotenv_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app_root = root / "landa-ai-rag"
            backend_root = root / "landa-backend"
            app_root.mkdir()
            backend_root.mkdir()
            (app_root / ".env").write_text(
                "AI_RAG_SERVICE_TOKEN=stale-development-token\n",
                encoding="utf-8",
            )
            (backend_root / ".env.production").write_text(
                "AI_RAG_SERVICE_TOKEN=production-token\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"NODE_ENV": "production"}, clear=True):
                environment = load_runtime_environment(app_root=app_root, repo_root=root)
                self.assertEqual(environment, "production")
                self.assertIsNone(os.environ.get("AI_RAG_SERVICE_TOKEN"))

    def test_production_preserves_process_injected_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app_root = root / "landa-ai-rag"
            backend_root = root / "landa-backend"
            app_root.mkdir()
            backend_root.mkdir()
            (app_root / ".env").write_text(
                "AI_RAG_SERVICE_TOKEN=stale-development-token\n",
                encoding="utf-8",
            )
            (backend_root / ".env.production").write_text(
                "AI_RAG_SERVICE_TOKEN=production-file-token\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"NODE_ENV": "production", "AI_RAG_SERVICE_TOKEN": "process-token"},
                clear=True,
            ):
                environment = load_runtime_environment(app_root=app_root, repo_root=root)
                self.assertEqual(environment, "production")
                self.assertEqual(os.environ.get("AI_RAG_SERVICE_TOKEN"), "process-token")


if __name__ == "__main__":
    unittest.main()
