from __future__ import annotations

import ast
import hashlib
import re
import unittest
from pathlib import Path

from fastapi.routing import APIRoute

from app import hashing, lesson_author_orchestration_v2, source_evidence_bundle
from app.api.deps import require_internal_token
from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
# Framework-free domain modules (§19.2 "module thuần").
PURE_MODULE_PATTERNS = (
    "source_*.py", "instructional_*.py", "lesson_quality.py", "semantic_review.py", "workflows/*.py",
    "assessment_*.py", "learner_content_purity.py", "ordered_learning_content.py", "media_brief.py",
    "component_capabilities.py", "lesson_author_*.py", "lesson_content_observation.py", "lesson_prompt_policy.py",
)
MAX_ROUTE_HANDLER_LINES = 30


def imported_modules(path: Path) -> list[tuple[str, set[str]]]:
    """(module, imported names) of every import statement in ``path`` (absolute imports only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, set()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, {alias.name for alias in node.names}))
    return found


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

    def test_routes_are_declared_only_in_api_route_modules(self) -> None:
        # PRD-2: app/main.py is the app factory; route handlers live in app/api/routes.
        offenders = [
            f"{route.path}:{route.endpoint.__module__}"
            for route in app.routes
            if isinstance(route, APIRoute) and not route.endpoint.__module__.startswith("app.api.routes.")
        ]
        self.assertEqual(offenders, [])

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

    def test_layers_import_only_what_the_layer_table_allows(self) -> None:
        # §19.2 dependency table. ``fastapi.HTTPException`` is the one web-framework name the
        # service layer still uses: it is the provider/service error type of the code moved out of
        # app/main.py, and turning it into AppError would change error bodies (correlation_id).
        layers: dict[str, tuple[list[Path], tuple[str, ...]]] = {
            "services": (sorted((APP_ROOT / "services").rglob("*.py")),
                         ("starlette", "app.api", "app.main")),
            "repositories": (sorted((APP_ROOT / "repositories").rglob("*.py")),
                             ("fastapi", "starlette", "app.api", "app.main", "app.services", "app.idm")),
            "schemas": (sorted((APP_ROOT / "schemas").rglob("*.py")),
                        ("fastapi", "starlette", "app.api", "app.main", "app.services", "app.repositories",
                         "app.infra")),
            "pure": (sorted(path for pattern in PURE_MODULE_PATTERNS for path in APP_ROOT.glob(pattern)),
                     ("fastapi", "starlette", "app.api", "app.main", "app.services", "app.repositories",
                      "app.infra", "asyncpg", "httpx", "requests")),
        }
        offenders = []
        for layer, (paths, forbidden) in layers.items():
            self.assertTrue(paths, layer)
            for path in paths:
                for name, imported in imported_modules(path):
                    if any(name == item or name.startswith(item + ".") for item in forbidden):
                        offenders.append(f"{layer}:{path.relative_to(APP_ROOT).as_posix()}:{name}")
                    if layer == "services" and name.split(".")[0] == "fastapi" and imported != {"HTTPException"}:
                        offenders.append(f"{layer}:{path.relative_to(APP_ROOT).as_posix()}:{name}:{imported}")
        self.assertEqual(offenders, [])

    def test_no_application_module_imports_app_main(self) -> None:
        # app/__main__.py names "app.main:app" as a string for uvicorn; nothing imports the factory module.
        offenders = [
            f"{path.relative_to(APP_ROOT).as_posix()}:{name}"
            for path in sorted(APP_ROOT.rglob("*.py"))
            for name, imported in imported_modules(path)
            if name == "app.main" or (name == "app" and "main" in imported)
        ]
        self.assertEqual(offenders, [])

    def test_file_sizes_follow_std5(self) -> None:
        # PRD-2 exit gate: app/main.py is the app factory only; moved legacy files stay <= 2,500 lines.
        sizes = {path.relative_to(APP_ROOT).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
                 for path in APP_ROOT.rglob("*.py")}
        self.assertLessEqual(sizes["main.py"], 200)
        self.assertEqual({name: lines for name, lines in sizes.items() if lines > 2500}, {})

    def test_route_handlers_stay_thin(self) -> None:
        # Route modules only bind HTTP to services: no handler body grows into business logic.
        offenders = []
        for path in sorted((APP_ROOT / "api" / "routes").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno is not None:
                    length = node.end_lineno - node.lineno + 1
                    if length > MAX_ROUTE_HANDLER_LINES:
                        offenders.append(f"{path.name}:{node.name}:{length}")
        self.assertEqual(offenders, [])

    def test_canonical_hash_has_one_implementation(self) -> None:
        # STD-7: contract and evidence hashes share app.hashing.canonical_hash (public names kept).
        offenders = [
            f"{path.relative_to(APP_ROOT).as_posix()}:{node.lineno}"
            for path in sorted(APP_ROOT.rglob("*.py"))
            if path != APP_ROOT / "hashing.py"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name in {"canonical_hash", "_canonical_hash"}
        ]
        self.assertEqual(offenders, [])
        self.assertIs(lesson_author_orchestration_v2.canonical_hash, hashing.canonical_hash)
        self.assertIs(source_evidence_bundle.canonical_hash, hashing.canonical_hash)
        # The encoding the Node mirror reproduces: sorted keys, compact separators, raw UTF-8.
        self.assertEqual(hashing.canonical_hash({"b": [1, "đ"], "a": None}),
                         hashlib.sha256('{"a":null,"b":[1,"đ"]}'.encode()).hexdigest())

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
