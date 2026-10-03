from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.config import load_runtime_dotenv, load_runtime_environment


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

    def test_production_ignores_ai_local_env_and_uses_backend_production(self) -> None:
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
                self.assertEqual(os.environ.get("AI_RAG_SERVICE_TOKEN"), "production-token")

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
