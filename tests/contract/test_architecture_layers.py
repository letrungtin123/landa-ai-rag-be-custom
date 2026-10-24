from __future__ import annotations

import ast
import hashlib
import importlib
import re
import unittest
from pathlib import Path
from typing import NamedTuple

from fastapi.routing import APIRoute

from app import hashing
from app.api.deps import require_internal_token
from app.content import source_evidence_bundle
from app.contracts.orchestration_v2 import blueprint as orchestration_v2_blueprint
from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
# Framework-free domain modules (§19.2 "module thuần").
PURE_MODULE_PATTERNS = (
    "source_*.py", "instructional_*.py", "lesson_quality.py", "semantic_review.py", "workflows/*.py",
    "assessment_*.py", "media_brief.py", "component_capabilities.py", "lesson_author_*.py",
    "lesson_content_observation.py", "lesson_prompt_policy.py",
    # The shared core moved verbatim out of the modules above in refactor wave 2.
    "content/*.py", "contracts/**/*.py",
)
MAX_ROUTE_HANDLER_LINES = 30
EVENT_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


# STD-5 (spec §19): files <= 800 lines and functions <= 120 lines in the code held to the full standard, which is
# every Python file of the strict paths of scripts/check.sh (``strict_paths``; test_quality_gate_scripts.py keeps
# check.ps1 equal to it); code moved verbatim in PRD-2 stays <= 2,500 lines per file.
CHECK_SH = REPO_ROOT / "scripts" / "check.sh"
MAX_STRICT_FILE_LINES = 800
MAX_STRICT_FUNCTION_LINES = 120
MAX_MOVED_FILE_LINES = 2_500


class Allowance(NamedTuple):
    lines: int
    reason: str
    target: str


# Strict-code offenders at their size when the STD-5 limits started to cover them: the first three in refactor
# wave 1 (2026-10-18), the others in refactor wave 3b (2026-10-24), when the rule grew from part of the strict paths
# to all of them (mostly code moved verbatim out of the legacy modules in wave 2, step 2.1). An entry only shrinks:
# the test fails when the file or function grows past its size here (lower the number when it shrinks) and when it
# fits the limit again (remove the entry). New code never gets an entry.
_MOVED = "moved verbatim from the legacy modules (wave 2, step 2.1)"
FILE_SIZE_ALLOWLIST: dict[str, Allowance] = {
    "app/contracts/orchestration_v2/provider.py": Allowance(1_240, f"provider drafts, wire, binders; {_MOVED}",
                                                            "split by concern"),
    "app/services/staged_unit/provider_schemas.py": Allowance(1_102, f"staged provider schemas; {_MOVED}",
                                                              "split by concern"),
    "app/services/staged_unit/validation.py": Allowance(1_190, f"staged unit validation; {_MOVED}", "split by concern"),
    "app/content/source_readiness.py": Allowance(846, f"PDF readiness analysis; {_MOVED}", "split by concern"),
}
FUNCTION_SIZE_ALLOWLIST: dict[str, Allowance] = {
    "app/idm/module_design.py::run_idm_module_design": Allowance(
        210, "W3/W4 call, repair loop, salvage and assembly", "wave 2 (2.5)"),
    "app/idm/runtime.py::idm_generate": Allowance(
        131, "admission, transport, usage ledger, trace and parse of one call", "wave 2 (2.5)"),
    "app/idm/course_design.py::run_idm_course_design": Allowance(
        127, "W1 map/reduce, W2, W4 and assembly", "wave 2 (2.5)"),
    # Refactor wave 3b: the strict paths the rule did not cover before.
    "app/contracts/orchestration_v2/provider.py::salvage_chapter_shard_draft_v2": Allowance(
        122, f"chapter shard salvage; {_MOVED}", "split into steps"),
    "app/contracts/orchestration_v2/provider.py::_compile_unit_components_v2": Allowance(
        195, f"unit component compiler; {_MOVED}", "split into steps"),
    "app/services/retrieval/search.py::retrieve_chunks": Allowance(
        183, "embedding, vector and keyword search, merge and limits of one retrieval", "split into steps"),
    "app/services/retrieval/source_coverage.py::build_source_coverage_manifest": Allowance(
        146, "source coverage manifest of a lesson-author request", "split into steps"),
    "app/services/ingestion/index.py::_index_document": Allowance(
        185, "load, extract, structure, chunk, embed and persist of one index run", "split into steps"),
    "app/services/orchestration_v2/chapter_shard.py::_lesson_author_orchestration_v2_chapter_shard": Allowance(
        177, "chapter shard route service (IDM and v2-legacy)", "split into steps"),
    "app/services/orchestration_v2/source_snapshot.py::lesson_author_orchestration_v2_source_snapshot": Allowance(
        312, "source snapshot route service", "split into steps"),
    "app/services/orchestration_v2/unit.py::_lesson_author_orchestration_v2_unit_legacy": Allowance(
        481, "v2-legacy unit route service", "PRD-3 (removal of the v2-legacy pipeline)"),
    "app/services/staged_unit/provider_schemas.py::build_lesson_author_proposal_response_schema": Allowance(
        141, f"proposal response schema; {_MOVED}", "split into steps"),
    "app/services/staged_unit/provider_schemas.py::build_lesson_author_unit_response_schema": Allowance(
        172, f"unit response schema; {_MOVED}", "split into steps"),
    "app/services/staged_unit/provider_schemas.py::build_staged_lesson_content_response_model": Allowance(
        144, f"staged content response model; {_MOVED}", "split into steps"),
    "app/services/staged_unit/validation.py::staged_instructional_finding": Allowance(
        257, f"staged instructional checks; {_MOVED}", "split into steps"),
    "app/services/staged_unit/validation.py::validate_staged_unit_content": Allowance(
        270, f"staged unit validation; {_MOVED}", "split into steps"),
    "app/content/instructional_quality.py::render_source_locked_html": Allowance(
        128, f"source-locked HTML renderer; {_MOVED}", "split into steps"),
    "app/content/source_evidence_bundle.py::_validate_payload": Allowance(
        126, f"evidence bundle validation; {_MOVED}", "split into steps"),
    "app/content/source_readiness.py::_page_readiness": Allowance(
        131, f"page readiness analysis; {_MOVED}", "split into steps"),
    "app/content/source_structure.py::analyze_source_structure": Allowance(
        137, f"source structure analysis; {_MOVED}", "split into steps"),
}


def strict_paths() -> list[str]:
    """The ``strict_paths`` list of scripts/check.sh (files and directories, relative to the repository)."""

    match = re.search(r'^strict_paths="([^"]*)"', CHECK_SH.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if match is None:
        raise AssertionError("strict_paths not found in scripts/check.sh")
    return match.group(1).split()


def strict_files() -> dict[str, Path]:
    """Every Python file of the strict paths, by its path relative to the repository."""

    files: dict[str, Path] = {}
    for item in strict_paths():
        path = REPO_ROOT / item
        for found in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            files[found.relative_to(REPO_ROOT).as_posix()] = found
    return dict(sorted(files.items()))


def function_lengths(name: str, tree: ast.AST) -> dict[str, int]:
    """``file::Qualified.name`` -> lines from the ``def`` line to the last line, nested functions included."""

    lengths: dict[str, int] = {}
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.end_lineno is not None:
                lengths[f"{name}::{prefix}{child.name}"] = child.end_lineno - child.lineno + 1
                stack.append((child, f"{prefix}{child.name}."))
            elif isinstance(child, ast.ClassDef):
                stack.append((child, f"{prefix}{child.name}."))
            else:
                stack.append((child, prefix))
    return lengths


def size_violations(sizes: dict[str, int], limit: int, allowlist: dict[str, Allowance]) -> list[str]:
    found = [f"{name}: {lines} lines > {limit}" for name, lines in sorted(sizes.items())
             if lines > limit and name not in allowlist]
    for name, allowance in sorted(allowlist.items()):
        lines = sizes.get(name)
        if lines is None or lines <= limit:
            found.append(f"{name}: {lines} lines fits the limit (or is gone): remove its allowlist entry")
        elif lines > allowance.lines:
            found.append(f"{name}: grew to {lines} lines (allowlisted at {allowance.lines}; {allowance.target})")
    return found


# STD-7 / F-08: accent folding lives in app/core/text.py. A decomposing normalisation (NFD or NFKD, or a form
# not known statically) anywhere else is a private fold copy. These are the sites left after refactor wave 2
# (step 2.2) replaced the exact duplicates; each one differs in semantics on purpose or mirrors a Node rule,
# so it keeps its own code (changing which variant it uses changes matching: ask the owner first). An entry
# only shrinks: lower the count when a site goes, remove the entry when the file has none.
FOLD_HOME = "core/text.py"
DECOMPOSITION_SITES: dict[str, tuple[int, str]] = {
    "content/instructional_quality.py": (1, "crossword answer key: đ/Đ to d/D keeping case, then ASCII and upper"),
    "idm/artifacts.py": (1, "detects Vietnamese text by its combining marks; not a fold"),
    "idm/faq_checks.py": (1, "detects Vietnamese text by its combining marks; not a fold"),
    "idm/node_acceptance_text.py": (1, "Node mirror: normalizedInstructionalText"),
    "idm/node_acceptance_widgets.py": (1, "Node mirror: the crossword answer key"),
    "idm/text.py": (1, "idm_fold: NFC, lower() (not casefold), đ to d, whitespace collapsed"),
    "instructional_opportunities.py": (1, "NFKD, lower(), đ to d, unicodedata.combining"),
    "lesson_quality.py": (1, "drops the marks before casefold, then keeps words only"),
    "services/staged_unit/provider_schemas.py": (1, "crossword answer key of the staged validator: upper, A-Z, digits"),
    "services/staged_unit/validation.py": (1, "FAQ heading signal: drops the marks before casefold, spaces collapsed"),
}


def decomposition_count(tree: ast.AST) -> int:
    """``unicodedata.normalize`` (or an imported ``normalize``) calls whose form is NFD, NFKD or not a literal."""

    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        named = (isinstance(func, ast.Attribute) and func.attr == "normalize" and isinstance(func.value, ast.Name)
                 and func.value.id == "unicodedata") or (isinstance(func, ast.Name) and func.id == "normalize")
        form = node.args[0]
        if named and (not isinstance(form, ast.Constant) or form.value in {"NFD", "NFKD"}):
            count += 1
    return count


def decomposition_violations(found: dict[str, int], allowed: dict[str, tuple[int, str]]) -> list[str]:
    problems = [f"{name}: {count} decomposing normalisation(s), a private fold copy: use app.core.text"
                for name, count in sorted(found.items()) if name not in allowed]
    for name, (count, _reason) in sorted(allowed.items()):
        actual = found.get(name, 0)
        if actual > count:
            problems.append(f"{name}: {actual} decomposing normalisations (allowlisted {count}): use app.core.text")
        elif actual < count:
            problems.append(f"{name}: {actual} decomposing normalisations (allowlisted {count}): lower the entry")
    return problems


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
        # §19.2 dependency table. The service layer raises app.core.errors.ServiceError / ProviderError
        # (refactor wave 2b): it never imports the web framework.
        layers: dict[str, tuple[list[Path], tuple[str, ...]]] = {
            "services": (sorted((APP_ROOT / "services").rglob("*.py")),
                         ("fastapi", "starlette", "app.api", "app.main")),
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
                for name, _imported in imported_modules(path):
                    if any(name == item or name.startswith(item + ".") for item in forbidden):
                        offenders.append(f"{layer}:{path.relative_to(APP_ROOT).as_posix()}:{name}")
        self.assertEqual(offenders, [])

    def test_code_imports_moved_modules_by_their_new_path(self) -> None:
        # Refactor wave 3b removed the self-replacing alias modules of wave 2 (step 2.1). The one left serves the
        # backend's IDM fixture only; nothing in this repository imports it.
        self.assertIs(importlib.import_module("app.lesson_author_orchestration_v2"), orchestration_v2_blueprint)
        offenders = [
            f"{path.relative_to(REPO_ROOT).as_posix()}:{name}"
            for root in ("app", "tests", "tools", "scripts")
            for path in sorted((REPO_ROOT / root).rglob("*.py"))
            if "lesson_author_orchestration_v2" in path.read_text(encoding="utf-8")
            for name, imported in imported_modules(path)
            if name == "app.lesson_author_orchestration_v2"
            or (name == "app" and "lesson_author_orchestration_v2" in imported)
        ]
        self.assertEqual(offenders, [])

    def test_core_and_infra_import_only_the_layers_below_them(self) -> None:
        # §19.2: app/core (configuration, logging, errors, security, limits) is the bottom layer and imports no
        # other app package; app/infra (database, storage, provider adapters) builds on app/core alone and never on
        # the schemas, services, repositories, IDM, the routes or the legacy modules, nor on the web framework.
        allowed = {"core": ("app.core",), "infra": ("app.core", "app.infra")}
        offenders = []
        for layer, prefixes in allowed.items():
            paths = sorted((APP_ROOT / layer).rglob("*.py"))
            self.assertTrue(paths, layer)
            for path in paths:
                for name, imported in imported_modules(path):
                    for module in ([f"app.{item}" for item in imported] if name == "app" else [name]):
                        outside = (module.startswith("app.")
                                   and not any(module == item or module.startswith(item + ".") for item in prefixes))
                        web = layer == "infra" and module.split(".", 1)[0] in {"fastapi", "starlette"}
                        if outside or web:
                            offenders.append(f"{layer}:{path.relative_to(APP_ROOT).as_posix()}:{module}")
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
        self.assertEqual({name: lines for name, lines in sizes.items() if lines > MAX_MOVED_FILE_LINES}, {})
        # STD-5 on the strict code: <= 800 lines per file; the allowlist only shrinks.
        strict = {name: len(path.read_text(encoding="utf-8").splitlines()) for name, path in strict_files().items()}
        self.assertGreater(len(strict), 190)
        self.assertEqual(size_violations(strict, MAX_STRICT_FILE_LINES, FILE_SIZE_ALLOWLIST), [])

    def test_size_rules_catch_growth_new_offenders_and_stale_entries(self) -> None:
        allowlist = {"idm/big.py": Allowance(900, "reason", "wave 2"), "idm/fixed.py": Allowance(850, "r", "wave 2")}
        sizes = {"idm/big.py": 901, "idm/fixed.py": 700, "idm/new.py": 801, "idm/ok.py": 800}
        self.assertEqual(size_violations(sizes, 800, allowlist), [
            "idm/new.py: 801 lines > 800",
            "idm/big.py: grew to 901 lines (allowlisted at 900; wave 2)",
            "idm/fixed.py: 700 lines fits the limit (or is gone): remove its allowlist entry",
        ])
        self.assertEqual(size_violations({"idm/big.py": 880}, 800, {"idm/big.py": allowlist["idm/big.py"]}), [])
        tree = ast.parse("def outer():\n    def inner():\n        pass\n    return inner\n\n"
                         "class Box:\n    async def method(self):\n        return 1\n")
        self.assertEqual(function_lengths("m.py", tree), {"m.py::outer": 4, "m.py::outer.inner": 2,
                                                          "m.py::Box.method": 2})
        # The rule covers exactly the strict paths of scripts/check.sh, outside app/ too, and no legacy module.
        names = set(strict_files())
        self.assertLessEqual({"app/idm/qa.py", "app/hashing.py", "app/services/provider.py", "app/contract_bridge.py",
                              "app/services/staged_unit/validation.py", "scripts/coverage_by_package.py"}, names)
        self.assertEqual({name for name in names if name.startswith(("app/services/lesson_author/", "app/workflows/"))
                          or name in {"app/lesson_author_blueprint.py", "app/semantic_review.py"}}, set())

    def test_function_sizes_follow_std5(self) -> None:
        # STD-5 on the strict code: <= 120 lines per function (def line to last line); the allowlist only shrinks.
        lengths: dict[str, int] = {}
        for name, path in strict_files().items():
            lengths.update(function_lengths(name, ast.parse(path.read_text(encoding="utf-8"))))
        self.assertGreater(len(lengths), 1_500)
        self.assertEqual(size_violations(lengths, MAX_STRICT_FUNCTION_LINES, FUNCTION_SIZE_ALLOWLIST), [])

    def test_logs_are_structured_events(self) -> None:
        # One log style: ``log_event(logger, level, "event_name", **fields)`` (app.core.logging) or
        # ``logger.<level>("event_name", extra={"event": "event_name", ...})``. No JSON or printf
        # arguments in the message. The frozen V5 pipeline keeps its JSON lines until PRD-3 deletes it.
        legacy = ("services/lesson_author/",)
        levels = {"debug", "info", "warning", "error", "exception", "critical"}
        offenders = []
        for path in sorted(APP_ROOT.rglob("*.py")):
            relative = path.relative_to(APP_ROOT).as_posix()
            if relative.startswith(legacy):
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in levels and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "logger"):
                    continue
                message = node.args[0] if node.args else None
                if (len(node.args) != 1 or not isinstance(message, ast.Constant) or not isinstance(message.value, str)
                        or not EVENT_NAME_PATTERN.fullmatch(message.value)):
                    offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(offenders, [])

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

    def test_accent_folding_has_one_home(self) -> None:
        found: dict[str, int] = {}
        for path in sorted(APP_ROOT.rglob("*.py")):
            name = path.relative_to(APP_ROOT).as_posix()
            count = decomposition_count(ast.parse(path.read_text(encoding="utf-8")))
            if count and name != FOLD_HOME:
                found[name] = count
        self.assertEqual(decomposition_violations(found, DECOMPOSITION_SITES), [])
        # Guard against the check silently finding nothing after a move.
        self.assertGreaterEqual(decomposition_count(ast.parse((APP_ROOT / FOLD_HOME).read_text(encoding="utf-8"))), 2)

    def test_fold_rule_catches_new_copies_growth_and_stale_entries(self) -> None:
        tree = ast.parse("\n".join([
            "import unicodedata", "from unicodedata import normalize", 'a = unicodedata.normalize("NFD", x)',
            'b = normalize("NFKD", x)', "c = unicodedata.normalize(form, x)", 'd = unicodedata.normalize("NFC", x)',
            'e = unicodedata.normalize("NFKC", x)',
        ]))
        self.assertEqual(decomposition_count(tree), 3)
        allowed = {"idm/kept.py": (2, "reason"), "idm/gone.py": (1, "reason")}
        self.assertEqual(decomposition_violations({"idm/kept.py": 3, "idm/new.py": 1}, allowed), [
            "idm/new.py: 1 decomposing normalisation(s), a private fold copy: use app.core.text",
            "idm/gone.py: 0 decomposing normalisations (allowlisted 1): lower the entry",
            "idm/kept.py: 3 decomposing normalisations (allowlisted 2): use app.core.text",
        ])
        self.assertEqual(decomposition_violations({"idm/kept.py": 2, "idm/gone.py": 1}, allowed), [])

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
        self.assertIs(orchestration_v2_blueprint.canonical_hash, hashing.canonical_hash)
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
