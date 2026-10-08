"""Route response snapshots: the PRD-2 safety net for splitting ``app/main.py``.

Every HTTP route is replayed through the real ASGI stack (middleware, auth, request
validation, error handlers and serialization) with recorded inputs, and the response
(status, content type, JSON body) must equal the recorded snapshot. The provider
boundary is the Google SDK client itself (``google.genai.Client``), so prompts,
provider configs and schemas are part of the snapshot too; the database is a
recording fake pool, so the exact SQL text, argument values and call order are
snapshotted as well. Nothing here performs network or database I/O.

A refactor must never edit ``tests/snapshots/routes/*.json``. Only the patch-target
table below may follow symbols to their new modules. Re-record deliberately with
``LANDA_UPDATE_ROUTE_SNAPSHOTS=1`` after an intended behaviour change.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import threading
import unittest
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import ExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import asyncpg
import httpx
from fastapi import FastAPI

from app.core.config import settings
from app.infra import gemini as gemini_infra
from app.infra.schema_check import SchemaCheckResult
from app.main import AiUsage, app

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots" / "routes"
UPDATE_ENV = "LANDA_UPDATE_ROUTE_SNAPSHOTS"
TOKEN = "route-snapshot-token-0123456789"
API_KEY = "route-snapshot-provider-key"

# Current homes of the symbols this harness touches. Only this table may change during PRD-2.
RUNTIME_STATE_MODULE = "app.main"  # runtime_state, db_pool, database, supabase_client, schema_guard
RETRIEVE_CHUNKS_TARGET = "app.main.retrieve_chunks"

# Settings read by the routes under test, pinned so ambient AI_RAG_* variables cannot leak in.
PINNED_SETTINGS: dict[str, Any] = {
    "auth_mode": "token",
    "service_token": TOKEN,
    "build_sha": "snapshot-sha",
    "top_k": 8,
    "max_context_chars": 18_000,
    "lesson_author_top_k": 24,
    "lesson_author_max_context_chars": 32_000,
    "lesson_author_max_chunks_per_document": 12,
    "lesson_author_scope_max_chunks": 48,
    "retrieval_candidate_multiplier": 4,
    "retrieval_min_score": 0.25,
    "retrieval_keyword_min_score": 0.50,
    "retrieval_max_chunks_per_document": 4,
    "embedding_batch_size": 32,
    "chunk_max_chars": 3_200,
    "chunk_overlap_chars": 350,
    "source_coverage_canonical_max_chars": 1_000_000,
    "semantic_review_mode": "off",
    "generation_temperature": 0.2,
}
# Values that legitimately differ between two runs of the same request (wall-clock timing).
VOLATILE_KEYS = frozenset({"duration_ms", "elapsed_ms", "timestamp_utc", "node_durations_ms"})
VOLATILE = "<volatile>"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    """Canonical JSON form of recorded call arguments (tuples become lists)."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: VOLATILE if key in VOLATILE_KEYS else normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def decode_sets(value: Any) -> Any:
    """Inputs store Python sets as ``{"__set__": [...]}``."""
    if isinstance(value, dict):
        if set(value) == {"__set__"}:
            return set(value["__set__"])
        return {key: decode_sets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_sets(item) for item in value]
    return value


# --------------------------------------------------------------------------- fake database
class FakeRecord:
    """asyncpg.Record stand-in: mapping access, ``dict(record)`` and value iteration."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def keys(self) -> Any:
        return self._values.keys()

    def values(self) -> Any:
        return self._values.values()

    def items(self) -> Any:
        return self._values.items()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values.values())

    def __len__(self) -> int:
        return len(self._values)


class FakePool:
    """Pool *and* connection stand-in. Rules match a substring of the whitespace-normalized SQL."""

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        self.rules = rules
        self.calls: list[dict[str, Any]] = []

    def _rule(self, method: str, sql: str, args: Any) -> dict[str, Any]:
        text = " ".join(sql.split())
        rule = next((item for item in self.rules
                     if item["match"] in text and item.get("method", method) == method), {})
        self.calls.append({"method": method, "sql_sha256": sha256_text(text), "rule": rule.get("match"),
                           "args": jsonable(args)})
        if rule.get("raise") == "UndefinedTableError":
            raise asyncpg.exceptions.UndefinedTableError("relation does not exist")
        if rule.get("raise") == "OSError":
            raise OSError("database unavailable")
        return rule

    async def fetch(self, sql: str, *args: Any) -> list[FakeRecord]:
        return [FakeRecord(row) for row in self._rule("fetch", sql, args).get("rows", [])]

    async def fetchrow(self, sql: str, *args: Any) -> FakeRecord | None:
        rows = self._rule("fetchrow", sql, args).get("rows", [])
        return FakeRecord(rows[0]) if rows else None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        return self._rule("fetchval", sql, args).get("value")

    async def execute(self, sql: str, *args: Any) -> str:
        return str(self._rule("execute", sql, args).get("status", "OK"))

    async def executemany(self, sql: str, args: Any) -> None:
        self._rule("executemany", sql, list(args))

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[FakePool]:
        yield self

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[FakePool]:
        yield self


# --------------------------------------------------------------------------- fake Google SDK
def describe_schema(schema: Any) -> Any:
    if schema is None:
        return None
    if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
        wire = json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
    elif hasattr(schema, "model_dump"):
        wire = json.dumps(schema.model_dump(mode="json", exclude_none=True), ensure_ascii=False, sort_keys=True)
    else:
        wire = json.dumps(jsonable(schema), ensure_ascii=False, sort_keys=True)
    return {"sha256": sha256_text(wire), "chars": len(wire)}


def describe_config(config: Any) -> Any:
    if hasattr(config, "model_dump"):
        return jsonable(config.model_dump(mode="json", exclude_none=True))
    if isinstance(config, dict):
        return {key: describe_schema(value) if key == "response_schema" else jsonable(value)
                for key, value in sorted(config.items())}
    return jsonable(config)


def embedding_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [round(byte / 255, 6) for byte in digest[:4]]


def schema_title(config: Any) -> str | None:
    schema = config.get("response_schema") if isinstance(config, dict) else None
    if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
        return str(schema.model_json_schema().get("title", schema.__name__))
    return None


def fake_genai_client(script: list[dict[str, Any]], calls: list[dict[str, Any]]) -> type:
    """A ``genai.Client`` replacement replaying ``script`` for generate calls.

    A script entry with a ``schema`` answers only a call whose response schema has that
    title (parallel IDM stages may call in any order); other entries answer in order.
    """

    lock = threading.Lock()

    class Models:
        def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
            title = schema_title(config)
            with lock:
                return self._answer(model, contents, config, title)

        def _answer(self, model: str, contents: str, config: Any, title: str | None) -> Any:
            calls.append({"op": "generate", "model": model, "schema": title, "prompt_sha256": sha256_text(contents),
                          "prompt_chars": len(contents), "config": describe_config(config)})
            position = next((index for index, entry in enumerate(script)
                             if entry.get("schema") in {None, title}), None)
            if position is None:
                raise AssertionError(f"Unexpected provider generate call for {title}")
            item = script.pop(position)
            usage = item.get("usage", [11, 7, 18])
            return SimpleNamespace(
                text=item["text"], parsed=None, prompt_feedback=None,
                candidates=[SimpleNamespace(finish_reason="STOP")],
                usage_metadata=SimpleNamespace(prompt_token_count=usage[0], candidates_token_count=usage[1],
                                               total_token_count=usage[2]),
            )

        def embed_content(self, *, model: str, contents: list[str], config: Any) -> Any:
            calls.append({"op": "embed", "model": model, "count": len(contents),
                          "contents_sha256": sha256_text(json.dumps(contents, ensure_ascii=False)),
                          "config": describe_config(config)})
            return SimpleNamespace(embeddings=[SimpleNamespace(values=embedding_vector(text)) for text in contents])

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            self.models = Models()

        def close(self) -> None:
            return None

    return Client


# --------------------------------------------------------------------------- harness
def database_dependencies(application: FastAPI) -> set[Callable[..., Any]]:
    """The route dependency that yields the pool, found by name so the harness survives moves."""
    found: set[Callable[..., Any]] = set()
    for route in application.routes:
        stack = list(getattr(getattr(route, "dependant", None), "dependencies", []))
        while stack:
            dependency = stack.pop()
            if getattr(dependency.call, "__name__", "") == "get_db":
                found.add(dependency.call)
            stack.extend(dependency.dependencies)
    return found


def resolve(target: str) -> tuple[Any, str]:
    module_name, attribute = target.rsplit(".", 1)
    return importlib.import_module(module_name), attribute


@contextmanager
def runtime_state(case: dict[str, Any]) -> Iterator[None]:
    runtime = importlib.import_module(RUNTIME_STATE_MODULE)
    state = case.get("runtime") or {}
    runtime.runtime_state.reset()
    runtime.schema_guard.reset()
    previous = (runtime.db_pool, runtime.database, runtime.supabase_client)
    runtime.db_pool = runtime.database = runtime.supabase_client = None
    try:
        if state.get("started"):
            runtime.runtime_state.started = True
        if state.get("db_pool"):
            runtime.db_pool = FakePool(state["db_pool"])
        if state.get("schema") == "ok":
            runtime.schema_guard.record(SchemaCheckResult(status="ok"))
        yield
    finally:
        runtime.runtime_state.reset()
        runtime.schema_guard.reset()
        runtime.db_pool, runtime.database, runtime.supabase_client = previous


def retrieval_stub(spec: dict[str, Any], calls: list[dict[str, Any]]) -> Callable[..., Any]:
    async def retrieve_chunks(*args: Any, **kwargs: Any) -> Any:
        calls.append({"op": "retrieve_chunks", "positional": len(args), "keywords": sorted(kwargs)})
        return (decode_sets(spec["rows"]), AiUsage(**spec.get("usage", {})), decode_sets(spec["structure"]))

    return retrieve_chunks


def replay(case: dict[str, Any]) -> dict[str, Any]:
    request = case["request"]
    pool = FakePool(case.get("db") or [])
    provider_calls: list[dict[str, Any]] = []
    retrieval_calls: list[dict[str, Any]] = []
    client_class = fake_genai_client(list(case.get("provider") or []), provider_calls)
    headers = {"X-Landa-AI-Service-Token": TOKEN} if request.get("auth", True) else {}

    async def send() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://snapshot") as client:
            return await client.request(request["method"], request["path"], json=request.get("json"),
                                        headers=headers)

    overrides = database_dependencies(app)
    with ExitStack() as stack:
        for name, value in {**PINNED_SETTINGS, **case.get("settings", {})}.items():
            stack.enter_context(patch.object(settings, name, value))
        stack.enter_context(patch("google.genai.Client", client_class))
        stack.enter_context(runtime_state(case))
        if case.get("retrieval") is not None:
            module, attribute = resolve(RETRIEVE_CHUNKS_TARGET)
            stack.enter_context(patch.object(module, attribute, retrieval_stub(case["retrieval"], retrieval_calls)))
        for dependency in overrides:
            app.dependency_overrides[dependency] = lambda: pool
        try:
            response = asyncio.run(send())
        finally:
            for dependency in overrides:
                app.dependency_overrides.pop(dependency, None)
            gemini_infra.client_pool.clear()
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        body: Any = normalize(response.json())
    else:
        # Metric families only: sample values and ``*_created`` series depend on earlier traffic.
        body = sorted({line.split()[2] for line in response.text.splitlines()
                       if line.startswith("# TYPE ") and not line.split()[2].endswith("_created")})
    return {
        "status": response.status_code,
        "content_type": content_type,
        "body": body,
        "db_calls": pool.calls,
        "provider_calls": provider_calls,
        "retrieval_calls": retrieval_calls,
    }


def snapshot_files() -> list[Path]:
    return sorted(SNAPSHOT_DIR.glob("*.json"))


class RouteSnapshotTests(unittest.TestCase):
    maxDiff = None

    def test_every_registered_route_has_a_snapshot(self) -> None:
        recorded = {(json.loads(path.read_text(encoding="utf-8"))["request"]["method"],
                     json.loads(path.read_text(encoding="utf-8"))["request"]["path"]) for path in snapshot_files()}
        registered = {(method, route.path) for route in app.routes
                      for method in getattr(route, "methods", set()) - {"HEAD"}
                      if route.path not in {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}}
        self.assertEqual(sorted(registered - recorded), [])

    def test_route_responses_match_recorded_snapshots(self) -> None:
        files = snapshot_files()
        self.assertGreaterEqual(len(files), 15)
        for path in files:
            case = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(case=path.stem):
                actual = replay(case)
                if os.environ.get(UPDATE_ENV) == "1":
                    case["expected"] = actual
                    path.write_text(json.dumps(case, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                    continue
                self.assertEqual(actual, case["expected"])


if __name__ == "__main__":
    unittest.main()
