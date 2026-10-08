from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

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
        # Routes are declared on ``app`` (app/main.py) or on a route module's ``router`` (PRD-2).
        paths = [APP_ROOT / "main.py", *sorted((APP_ROOT / "api" / "routes").glob("*.py"))]
        offenders = []
        declared = 0
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                        continue
                    if not isinstance(decorator.func.value, ast.Name) or decorator.func.value.id not in {
                        "app", "router",
                    }:
                        continue
                    if decorator.func.attr not in {"get", "post", "put", "patch", "delete"}:
                        continue
                    declared += 1
                    route = (
                        decorator.args[0].value
                        if decorator.args and isinstance(decorator.args[0], ast.Constant)
                        else ""
                    )
                    if route in {"/healthz", "/readyz"}:
                        continue
                    dependencies = next(
                        (keyword.value for keyword in decorator.keywords if keyword.arg == "dependencies"),
                        None,
                    )
                    if dependencies is None or "require_internal_token" not in ast.unparse(dependencies):
                        offenders.append(f"{path.name}:{route}:{node.lineno}")
        self.assertEqual(offenders, [])
        # Guard against the scan silently finding nothing after a move.
        self.assertGreaterEqual(declared, 15)

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
        source = (APP_ROOT / "main.py").read_text(encoding="utf-8")
        load_document = re.search(
            r"async def load_document\(.*?(?=\nasync def |\n@app\.)",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(load_document)
        self.assertIn("tenant_id", load_document.group(0))
        self.assertIn('"INDEX_DOCUMENT_FAILED"', source)
        self.assertNotIn("await mark_index_error(pool, index_id, str(", source)


if __name__ == "__main__":
    unittest.main()
