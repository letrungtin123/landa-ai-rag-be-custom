from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from fastapi.routing import APIRoute

from app.api.deps import require_internal_token
from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"


class ArchitectureLayerTests(unittest.TestCase):
    def test_application_does_not_reference_backend_files(self) -> None:
        offenders = []
        for path in APP_ROOT.rglob("*.py"):
            if "landa-backend" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_application_has_no_print_calls(self) -> None:
        offenders = []
        for path in APP_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                    offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_all_non_health_routes_declare_internal_auth_dependency(self) -> None:
        # Checked on the assembled application: every API route outside liveness/readiness
        # carries the internal-auth dependency, wherever the route module declares it.
        routes = [route for route in app.routes if isinstance(route, APIRoute)]
        offenders = [
            route.path
            for route in routes
            if route.path not in {"/healthz", "/readyz"}
            and require_internal_token not in {dependency.call for dependency in route.dependant.dependencies}
        ]
        self.assertEqual(offenders, [])
        # Guard against the check silently finding nothing after a move.
        self.assertGreaterEqual(len(routes), 15)

    def test_idm_package_never_imports_web_framework_or_service_module(self) -> None:
        # §19.2: app/idm receives its runtime by injection and stays framework free.
        forbidden = ("fastapi", "starlette", "app.main", "app.api", "asyncpg", "httpx", "requests")
        offenders = []
        for path in (APP_ROOT / "idm").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                offenders.extend(f"{path.name}:{name}" for name in names
                                 if any(name == item or name.startswith(item + ".") for item in forbidden))
        self.assertEqual(offenders, [])

    def test_document_queries_keep_tenant_filter_and_safe_error_code(self) -> None:
        repository = (APP_ROOT / "repositories" / "indexing.py").read_text(encoding="utf-8")
        load_document = re.search(
            r"async def load_document\(.*?(?=\nasync def |\n@app\.|\Z)",
            repository,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(load_document)
        self.assertIn("tenant_id", load_document.group(0))
        service = (APP_ROOT / "services" / "ingestion" / "index.py").read_text(encoding="utf-8")
        self.assertIn('"INDEX_DOCUMENT_FAILED"', service)
        self.assertNotIn("mark_index_error(pool, index_id, str(", service)


if __name__ == "__main__":
    unittest.main()
