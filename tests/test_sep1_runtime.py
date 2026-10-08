"""SEP-1 #1/#2/#5/#6: configurable Postgres pool, no crash-loop on an unreachable database,
read-only schema check behind /readyz, GET /v1/meta and the keep-alive default. Offline only."""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import certifi
import httpx
from pydantic import ValidationError

from app import main
from app.__main__ import uvicorn_options
from app.core.config import Settings, missing_required_setting_names
from app.infra import db as db_infra
from app.infra import gemini as gemini_infra
from app.infra import schema_check
from app.infra.schema_check import SchemaCheckResult, SchemaGuard
from app.api.routes import health as health_routes
from app.services import runtime as runtime_service

TOKEN = "sep1-test-token-0123456789"


def config(**overrides: Any) -> db_infra.DatabaseConfig:
    values: dict[str, Any] = {
        "dsn": "postgresql://landa_ai_rag:pw@db.internal:5432/postgres", "production": True, "pool_min": 1,
        "pool_max": 8, "ssl_mode": None, "ssl_root_cert": "", "statement_cache_size": 100,
        "connect_timeout_seconds": 10.0, "command_timeout_seconds": 600.0, "tcp_keepalives_idle_seconds": 60,
    }
    values.update(overrides)
    return db_infra.DatabaseConfig(**values)


class PoolConfigurationTests(unittest.TestCase):
    def test_ssl_mode_precedence(self) -> None:
        cases = (
            ({}, "require"),  # production, network host
            ({"ssl_mode": "verify-full"}, "verify-full"),
            ({"dsn": "postgresql://u@db.internal/postgres?sslmode=verify-ca"}, "verify-ca"),
            ({"dsn": "postgresql://u@db.internal/postgres?sslmode=verify-ca", "ssl_mode": "disable"}, "disable"),
            ({"dsn": "postgresql://u@db.internal/postgres?sslmode=allow"}, "prefer"),
            ({"dsn": "postgresql://u@127.0.0.1:54322/postgres"}, "prefer"),  # loopback: TLS protects nothing
            ({"dsn": "postgresql://u@[::1]:5432/postgres"}, "prefer"),
            ({"dsn": "postgresql:///postgres"}, "prefer"),  # local socket
            ({"production": False}, "prefer"),
            ({"dsn": "::not a dsn::"}, "prefer"),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                self.assertEqual(db_infra.effective_ssl_mode(config(**overrides)), expected)

    def test_ssl_argument_per_mode(self) -> None:
        self.assertIs(db_infra.build_ssl_argument("disable", ""), False)
        self.assertEqual(db_infra.build_ssl_argument("prefer", ""), "prefer")
        require = db_infra.build_ssl_argument("require", "")
        assert isinstance(require, ssl.SSLContext)
        self.assertEqual((require.verify_mode, require.check_hostname), (ssl.CERT_NONE, False))
        ca = certifi.where()
        for mode, check_hostname in (("require", False), ("verify-ca", False), ("verify-full", True)):
            with self.subTest(mode=mode):
                context = db_infra.build_ssl_argument(mode, ca)  # type: ignore[arg-type]
                assert isinstance(context, ssl.SSLContext)
                self.assertEqual((context.verify_mode, context.check_hostname), (ssl.CERT_REQUIRED, check_hostname))

    def test_pool_arguments_carry_every_setting(self) -> None:
        arguments = db_infra.pool_arguments(config(statement_cache_size=0, pool_min=0, pool_max=3))
        self.assertEqual({key: arguments[key] for key in ("min_size", "max_size", "command_timeout",
                                                         "statement_cache_size", "timeout")},
                         {"min_size": 0, "max_size": 3, "command_timeout": 600.0, "statement_cache_size": 0,
                          "timeout": 10.0})
        self.assertIsInstance(arguments["ssl"], ssl.SSLContext)
        self.assertEqual(arguments["server_settings"], {
            "application_name": "landa-ai-rag", "tcp_keepalives_idle": "60", "tcp_keepalives_interval": "10",
            "tcp_keepalives_count": "6",
        })
        self.assertEqual(db_infra.server_settings(config(tcp_keepalives_idle_seconds=0)),
                         {"application_name": "landa-ai-rag"})
        # The DSN root certificate is honoured when no setting overrides it.
        dsn = f"postgresql://u@db.internal/postgres?sslmode=verify-full&sslrootcert={certifi.where()}"
        self.assertEqual(db_infra.effective_ssl_root_cert(config(dsn=dsn)), certifi.where())
        self.assertEqual(db_infra.effective_ssl_root_cert(config(dsn=dsn, ssl_root_cert="/ca.pem")), "/ca.pem")

    def test_settings_validate_pool_bounds_and_modes(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            defaults = Settings(environment="test")
            self.assertEqual((defaults.db_pool_min, defaults.db_pool_max, defaults.db_statement_cache_size,
                              defaults.db_ssl_mode, defaults.keep_alive_timeout_seconds), (1, 8, 100, None, 75))
            with self.assertRaises(ValidationError):
                Settings(environment="test", db_pool_min=9, db_pool_max=2)
        invalid_mode = {"AI_RAG_DB_SSL_MODE": "sometimes"}
        with patch.dict(os.environ, invalid_mode, clear=True), self.assertRaises(ValidationError):
            Settings(environment="test")
        with patch.dict(os.environ, {"AI_RAG_DB_SSL_MODE": "verify-full", "AI_RAG_DB_STATEMENT_CACHE_SIZE": "0"},
                        clear=True):
            configured = Settings(environment="test")
        self.assertEqual((configured.db_ssl_mode, configured.db_statement_cache_size), ("verify-full", 0))

    def test_service_key_is_optional_and_the_allowlist_replaces_supabase_url(self) -> None:
        base = {"environment": "production", "database_url": "postgresql://x", "auth_mode": "hmac",
                "service_hmac_secrets": "kid:0123456789abcdef"}
        self.assertEqual(missing_required_setting_names(Settings(**base, supabase_url="https://s.internal")), [])
        self.assertEqual(missing_required_setting_names(
            Settings(**base, storage_allowed_origins="https://s.internal:8443")), [])
        self.assertEqual(missing_required_setting_names(Settings(**base)), ["SUPABASE_URL"])

    def test_uvicorn_keep_alive_outlives_client_idle_timeouts(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            options = uvicorn_options(Settings(environment="test"))
        # nginx upstream keepalive_timeout is 60 s and the Node agent idles sockets out after 30 s.
        self.assertEqual(options["timeout_keep_alive"], 75)

    def test_safe_error_fields_and_backoff(self) -> None:
        class Rejected(Exception):
            sqlstate = "28P01"

        self.assertEqual(db_infra.safe_error_fields(Rejected("password for user x")),
                         {"error_type": "Rejected", "sqlstate": "28P01"})
        self.assertEqual(db_infra.safe_error_fields(OSError("host db.internal")), {"error_type": "OSError"})
        no_jitter = lambda low, high: 1.0  # noqa: E731
        self.assertEqual([db_infra.retry_delay_seconds(n, maximum=30.0, jitter=no_jitter) for n in (1, 2, 3, 6, 40)],
                         [1.0, 2.0, 4.0, 30.0, 30.0])
        self.assertEqual(db_infra.retry_delay_seconds(1, maximum=0.1, jitter=no_jitter), 0.5)


class DatabaseRuntimeTests(unittest.TestCase):
    def test_first_attempt_success_runs_the_post_connect_hook(self) -> None:
        pool = MagicMock(close=AsyncMock())
        hook = AsyncMock()

        async def scenario() -> None:
            runtime = db_infra.DatabaseRuntime(create_pool=AsyncMock(return_value=pool), on_connected=hook,
                                               retry_max_seconds=1.0)
            await runtime.start()
            self.assertEqual((runtime.state, runtime.attempts, runtime.pool), ("connected", 1, pool))
            await runtime.close()
            self.assertEqual((runtime.state, runtime.pool), ("closed", None))

        asyncio.run(scenario())
        hook.assert_awaited_once_with(pool)
        pool.close.assert_awaited_once()

    def test_unreachable_database_retries_in_the_background_without_logging_details(self) -> None:
        pool = MagicMock(close=AsyncMock())
        create = AsyncMock(side_effect=[OSError("password=hunter2 host=db.internal"), ConnectionError("x"), pool])
        delays: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            delays.append(seconds)

        async def scenario() -> db_infra.DatabaseRuntime:
            runtime = db_infra.DatabaseRuntime(create_pool=create, on_connected=AsyncMock(), retry_max_seconds=5.0,
                                               sleep=fake_sleep, jitter=lambda low, high: 1.0)
            await runtime.start()  # never raises
            self.assertIsNone(runtime.pool)
            self.assertEqual(runtime.state, "retrying")
            await runtime.wait_retry()
            return runtime

        with self.assertLogs("app.infra.db", logging.INFO) as logs:
            runtime = asyncio.run(scenario())
        self.assertEqual((runtime.state, runtime.attempts, runtime.pool), ("connected", 3, pool))
        self.assertEqual(delays, [1.0, 2.0])
        output = "\n".join(logs.output)
        self.assertIn("database_connect_failed", output)
        self.assertIn("database_connect_retry_scheduled", output)
        self.assertNotIn("hunter2", output)
        self.assertNotIn("db.internal", output)

    def test_close_stops_a_pending_retry_and_a_failing_hook_keeps_the_pool(self) -> None:
        async def never_connect() -> Any:
            raise OSError("down")

        async def scenario() -> None:
            runtime = db_infra.DatabaseRuntime(create_pool=never_connect, on_connected=AsyncMock(),
                                               retry_max_seconds=60.0)
            await runtime.start()
            await runtime.close()
            self.assertEqual(runtime.state, "closed")
            await runtime.wait_retry()  # no task left

            pool = MagicMock(close=AsyncMock())
            failing = db_infra.DatabaseRuntime(create_pool=AsyncMock(return_value=pool),
                                               on_connected=AsyncMock(side_effect=RuntimeError("hook")),
                                               retry_max_seconds=1.0)
            with self.assertLogs("app.infra.db", logging.WARNING) as logs:
                await failing.start()
            self.assertIs(failing.pool, pool)
            self.assertIn("database_post_connect_failed", "\n".join(logs.output))

        asyncio.run(scenario())


class FakeCatalog:
    """Answers the schema-check catalog queries from in-memory state (superuser by default)."""

    def __init__(self) -> None:
        names = [requirement.table for requirement in schema_check.AI_SCHEMA_REQUIREMENTS]
        self.tables = {name: 1000 + index for index, name in enumerate(names)}
        self.columns: dict[int, set[str]] = {}
        for requirement in schema_check.AI_SCHEMA_REQUIREMENTS:
            wanted = set(requirement.columns)
            for _, grant_columns in requirement.grants:
                wanted.update(grant_columns or ())
            self.columns[self.tables[requirement.table]] = wanted
        self.denied: set[tuple[int, str, str | None]] = set()
        self.bypass = True
        self.rls: dict[int, tuple[bool, bool, bool]] = {}
        self.policies: list[tuple[int, str]] = []
        self.extensions = {"vector", "pg_trgm"}
        self.unresolved: set[str] = set()
        self.schema_usage = True
        self.fail = False

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if self.fail:
            raise ConnectionError("lost")
        if "to_regclass(t.name)" in query:
            return [{"name": name, "oid": self.tables.get(name)} for name in args[0]]
        if "FROM pg_attribute" in query:
            return [{"oid": oid, "name": column} for oid in args[0] for column in sorted(self.columns.get(oid, ()))]
        if "has_table_privilege(k.rel" in query:
            return [{"ord": number, "ok": (rel, priv, col) not in self.denied}
                    for number, (rel, priv, col) in enumerate(zip(*args, strict=True), start=1)]
        if "FROM pg_roles" in query:
            return [{"bypass": self.bypass}]
        if "relrowsecurity" in query:
            return [{"oid": oid, "rls": rls, "forced": forced, "owner": owner}
                    for oid, (rls, forced, owner) in self.rls.items() if oid in args[0]]
        if "FROM pg_policy" in query:
            return [{"oid": oid, "cmd": cmd} for oid, cmd in self.policies if oid in args[0]]
        if "FROM pg_extension" in query:
            return [{"extname": name} for name in args[0] if name in self.extensions]
        if "to_regprocedure" in query:
            names = [f"function:{v}" for v in args[0]] + [f"type:{v}" for v in args[1]]
            names += [f"operator:{v}" for v in args[2]]
            return [{"name": name, "ok": name not in self.unresolved} for name in names]
        if "has_schema_privilege" in query:
            return [{"name": "public", "ok": self.schema_usage}]
        raise AssertionError(f"unexpected query: {query[:60]}")

    @asynccontextmanager
    async def transaction(self, *, readonly: bool = False) -> AsyncIterator[None]:
        assert readonly, "the schema check must run read-only"
        yield

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[FakeCatalog]:
        yield self


class SchemaCheckTests(unittest.TestCase):
    def missing(self, catalog: FakeCatalog) -> list[str]:
        return asyncio.run(schema_check.collect_missing(catalog))

    def test_superuser_and_a_fully_granted_role_pass(self) -> None:
        catalog = FakeCatalog()
        self.assertEqual(self.missing(catalog), [])
        # landa_ai_rag: no bypass, RLS on every table, the SQL file's policies (FOR ALL / FOR SELECT).
        catalog.bypass = False
        for requirement in schema_check.AI_SCHEMA_REQUIREMENTS:
            oid = catalog.tables[requirement.table]
            catalog.rls[oid] = (True, False, False)
            catalog.policies.append((oid, "r" if requirement.commands() == {"SELECT"} else "*"))
        self.assertEqual(self.missing(catalog), [])

    def test_missing_objects_are_named(self) -> None:
        catalog = FakeCatalog()
        del catalog.tables["rag_chunks"]
        del catalog.tables["tenants"]  # optional: absence is fine
        indexes = catalog.tables["rag_document_indexes"]
        catalog.columns[indexes].discard("chunk_count")
        usage = catalog.tables["tenant_data_quota_usage"]
        catalog.columns[usage].discard("state")  # optional table: only existing columns are checked
        kb = catalog.tables["kb_documents"]
        catalog.denied |= {(kb, "SELECT", "content"), (indexes, "DELETE", None)}
        catalog.extensions.discard("pg_trgm")
        catalog.unresolved.add("function:similarity(text,text)")
        catalog.schema_usage = False
        self.assertEqual(self.missing(catalog), [
            "table:rag_chunks", "column:rag_document_indexes.chunk_count", "privilege:SELECT:kb_documents.content",
            "privilege:DELETE:rag_document_indexes", "extension:pg_trgm", "function:similarity(text,text)",
            "schema_usage:public",
        ])

    def test_row_level_security_needs_an_applicable_policy(self) -> None:
        catalog = FakeCatalog()
        catalog.bypass = False
        chunks, kb, nodes = (catalog.tables[name] for name in ("rag_chunks", "kb_documents",
                                                               "rag_document_structure_nodes"))
        catalog.rls = {chunks: (True, False, False), kb: (True, False, False), nodes: (True, False, True)}
        catalog.policies = [(chunks, "r")]  # SELECT only: INSERT is silently filtered
        self.assertEqual(self.missing(catalog), ["rls_policy:SELECT:kb_documents", "rls_policy:INSERT:rag_chunks"])
        catalog.rls[nodes] = (True, True, True)  # FORCE applies RLS to the owner as well
        self.assertIn("rls_policy:DELETE:rag_document_structure_nodes", self.missing(catalog))
        catalog.rls = {}
        self.assertEqual(self.missing(catalog), [])

    def test_no_required_table_present_short_circuits(self) -> None:
        catalog = FakeCatalog()
        catalog.tables = {}
        self.assertEqual(self.missing(catalog), ["table:kb_documents", "table:rag_document_indexes",
                                                 "table:rag_chunks"])

    def test_run_schema_check_outcomes(self) -> None:
        catalog = FakeCatalog()
        ok = asyncio.run(schema_check.run_schema_check(catalog, timeout_seconds=5))
        self.assertEqual((ok.status, ok.code, ok.ok), ("ok", None, True))
        catalog.extensions.clear()
        with self.assertLogs("app.infra.schema_check", logging.ERROR) as logs:
            failed = asyncio.run(schema_check.run_schema_check(catalog, timeout_seconds=5))
        self.assertEqual((failed.status, failed.code), ("failed", "SCHEMA_CHECK_FAILED"))
        self.assertEqual(logs.records[0].missing, ["extension:vector", "extension:pg_trgm"])  # type: ignore[attr-defined]
        self.assertEqual(failed.summary()["missing"], ["extension:vector", "extension:pg_trgm"])
        catalog.fail = True
        errored = asyncio.run(schema_check.run_schema_check(catalog, timeout_seconds=5))
        self.assertEqual((errored.status, errored.code), ("error", "SCHEMA_CHECK_UNAVAILABLE"))

    def test_guard_caches_success_and_rechecks_failures_after_the_interval(self) -> None:
        now = [100.0]
        guard = SchemaGuard(retry_seconds=30.0, clock=lambda: now[0])
        self.assertEqual(guard.summary()["status"], "pending")
        catalog = FakeCatalog()
        catalog.fail = True
        self.assertEqual(asyncio.run(guard.current(catalog, timeout_seconds=1)).status, "error")
        catalog.fail = False
        now[0] += 10
        self.assertEqual(asyncio.run(guard.current(catalog, timeout_seconds=1)).status, "error")  # cached
        now[0] += 30
        self.assertTrue(asyncio.run(guard.current(catalog, timeout_seconds=1)).ok)  # re-checked
        catalog.fail = True
        now[0] += 3_600
        self.assertTrue(asyncio.run(guard.current(catalog, timeout_seconds=1)).ok)  # success is cached
        self.assertEqual(guard.summary()["status"], "ok")


class ReadyPool:
    async def fetchval(self, query: str, *args: Any) -> int:
        return 1


class EndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.patches = [patch.object(main.settings, "auth_mode", "token"),
                        patch.object(main.settings, "service_token", TOKEN)]
        for item in self.patches:
            item.start()
        runtime_service.runtime_state.reset()
        runtime_service.schema_guard.reset()

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        runtime_service.runtime_state.reset()
        runtime_service.schema_guard.reset()
        runtime_service.db_pool = None

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def scenario() -> httpx.Response:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(scenario())

    def test_readyz_reports_schema_failures_with_a_safe_code(self) -> None:
        runtime_service.runtime_state.started = True
        runtime_service.db_pool = ReadyPool()  # type: ignore[assignment]
        runtime_service.schema_guard.record(SchemaCheckResult(status="failed", missing=("table:rag_chunks",)))
        response = self.request("GET", "/readyz")
        self.assertEqual((response.status_code, response.json()["detail"]["code"]), (503, "SCHEMA_CHECK_FAILED"))
        self.assertNotIn("rag_chunks", response.text)
        runtime_service.schema_guard.record(SchemaCheckResult(status="ok"))
        self.assertEqual(self.request("GET", "/readyz").status_code, 200)

    def test_readyz_runs_a_pending_schema_check(self) -> None:
        runtime_service.runtime_state.started = True
        runtime_service.db_pool = ReadyPool()  # type: ignore[assignment]
        with patch.object(schema_check, "run_schema_check",
                          AsyncMock(return_value=SchemaCheckResult(status="error"))) as check:
            response = self.request("GET", "/readyz")
        check.assert_awaited_once()
        self.assertEqual(response.json()["detail"]["code"], "SCHEMA_CHECK_UNAVAILABLE")

    def test_meta_requires_auth_and_reports_build_contracts_and_schema(self) -> None:
        self.assertEqual(self.request("GET", "/v1/meta").status_code, 401)
        headers = {"X-Landa-AI-Service-Token": TOKEN}
        with patch.object(main.settings, "build_sha", "892c62e"):
            body = self.request("GET", "/v1/meta", headers=headers).json()
        self.assertEqual(body["service"], "landa-ai-rag")
        self.assertEqual(body["build_sha"], "892c62e")
        self.assertEqual(body["contracts"], {
            "orchestration_v2_contract_version": 2,
            "orchestration_v2_provider_schema_projection": "orchestration-v2-google-wire-2",
            "idm_pipeline_version": "idm-1", "idm_contract_version": 1, "idm_prompt_policy_version": "idm-prompt-1",
            "rag_index_request_version": 2,
        })
        self.assertEqual(body["capabilities"]["index_source_download_url"], True)
        self.assertEqual((body["schema_check"]["status"], body["database"]["state"]), ("pending", "idle"))
        with patch.object(main.settings, "build_sha", "bad sha <script>"):
            self.assertEqual(self.request("GET", "/v1/meta", headers=headers).json()["build_sha"], "unknown")

    def test_routes_answer_database_not_ready_while_the_pool_reconnects(self) -> None:
        response = self.request("POST", "/v1/kb/delete", headers={"X-Landa-AI-Service-Token": TOKEN},
                                json={"tenant_id": "11111111-1111-4111-8111-111111111111",
                                      "kb_id": "22222222-2222-4222-8222-222222222222"})
        self.assertEqual((response.status_code, response.json()["detail"]["code"]), (503, "DATABASE_NOT_READY"))


class StartupTests(unittest.TestCase):
    def tearDown(self) -> None:
        runtime_service.db_pool = None
        runtime_service.database = None
        runtime_service.schema_guard.reset()
        runtime_service.runtime_state.reset()

    def test_startup_survives_an_unreachable_database_and_connects_later(self) -> None:
        pool = MagicMock(close=AsyncMock(), fetchval=AsyncMock(return_value=1))
        create_pool = AsyncMock(side_effect=[OSError("refused"), pool])
        refresh = AsyncMock(side_effect=lambda *args, **kwargs: runtime_service.schema_guard.record(SchemaCheckResult("ok")))

        async def scenario() -> None:
            async with main.app.router.lifespan_context(main.app):
                self.assertTrue(runtime_service.runtime_state.started)
                self.assertIsNone(runtime_service.db_pool)
                self.assertEqual((await health_routes.readyz()).status_code, 503)
                assert runtime_service.database is not None
                await runtime_service.database.wait_retry()
                self.assertIs(runtime_service.db_pool, pool)
                self.assertEqual((await health_routes.readyz()).status_code, 200)
            self.assertIsNone(runtime_service.db_pool)

        with (
            patch.object(main.asyncpg, "create_pool", create_pool),
            patch.object(runtime_service, "require_settings", MagicMock()),
            patch.object(main.settings, "supabase_service_key", ""),
            patch.object(runtime_service.schema_guard, "refresh", refresh),
            patch.object(db_infra, "retry_delay_seconds", lambda *args, **kwargs: 0.01),
            patch.object(gemini_infra.client_pool, "clear"),
            patch.object(runtime_service.concurrency, "shutdown"),
        ):
            asyncio.run(scenario())
        self.assertEqual(create_pool.await_count, 2)
        kwargs = create_pool.await_args.kwargs
        self.assertEqual(kwargs["server_settings"]["application_name"], "landa-ai-rag")
        self.assertEqual((kwargs["min_size"], kwargs["max_size"]),
                         (main.settings.db_pool_min, main.settings.db_pool_max))
        refresh.assert_awaited_once()
        pool.close.assert_awaited_once()
        self.assertIsNone(runtime_service.supabase_client)  # no service key: signed URLs only

    def test_invalid_storage_allowlist_fails_fast(self) -> None:
        with (
            patch.object(runtime_service, "require_settings", MagicMock()),
            patch.object(main.settings, "storage_allowed_origins", "http://10.1.2.3:8000"),
            self.assertRaisesRegex(RuntimeError, "AI_RAG_STORAGE_ALLOWED_ORIGINS"),
        ):
            asyncio.run(runtime_service.startup())


if __name__ == "__main__":
    unittest.main()
