from __future__ import annotations

import ast
import asyncio
import logging
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import HTTPException

from app import main
from app.core import metrics
from app.core.errors import AppError
from app.infra import gemini as gemini_infra
from app.infra.schema_check import SchemaCheckResult
from app.schemas.chat import RagChatRequest
from app.schemas.kb import RagIndexRequest
from app.services import deadlines as deadlines_service
from app.services import provider as provider_service
from app.services import runtime as runtime_service

APP_ROOT = Path(__file__).resolve().parents[1] / "app"
TOKEN = "prd1-test-token-0123456789"
TENANT = "11111111-1111-4111-8111-111111111111"
KB = "22222222-2222-4222-8222-222222222222"
DOC = "33333333-3333-4333-8333-333333333333"


def metric(name: str, labels: dict[str, str]) -> float:
    value = metrics.REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


class ReadyPool:
    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay

    async def fetchval(self, query: str, *args: Any) -> int:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise OSError("db unreachable")
        return 1


class AsgiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.patches = [
            patch.object(main.settings, "auth_mode", "token"),
            patch.object(main.settings, "service_token", TOKEN),
        ]
        for item in self.patches:
            item.start()
        runtime_service.runtime_state.reset()

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        runtime_service.runtime_state.reset()
        runtime_service.db_pool = None
        runtime_service.schema_guard.reset()

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def scenario() -> httpx.Response:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(scenario())


class HealthAndReadinessTests(AsgiTestCase):
    def test_healthz_is_liveness_only_and_unauthenticated(self) -> None:
        response = self.request("GET", "/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertTrue(response.headers["x-request-id"])

    def test_readyz_requires_started_runtime_and_database(self) -> None:
        self.assertEqual(self.request("GET", "/readyz").status_code, 503)
        runtime_service.runtime_state.started = True
        runtime_service.db_pool = ReadyPool()  # type: ignore[assignment]
        # SEP-1: readiness also requires a passed schema check (covered in test_sep1_runtime).
        runtime_service.schema_guard.record(SchemaCheckResult(status="ok"))
        response = self.request("GET", "/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})

    def test_readyz_reports_draining_and_database_failures_without_details(self) -> None:
        runtime_service.runtime_state.started = True
        runtime_service.db_pool = ReadyPool(fail=True)  # type: ignore[assignment]
        response = self.request("GET", "/readyz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "NOT_READY")
        self.assertNotIn("unreachable", response.text)
        runtime_service.db_pool = ReadyPool()  # type: ignore[assignment]
        runtime_service.runtime_state.draining = True
        self.assertEqual(self.request("GET", "/readyz").status_code, 503)

    def test_readyz_times_out_slow_database(self) -> None:
        runtime_service.runtime_state.started = True
        runtime_service.db_pool = ReadyPool(delay=1.0)  # type: ignore[assignment]
        with patch.object(main.settings, "readiness_db_timeout_ms", 100):
            self.assertEqual(self.request("GET", "/readyz").status_code, 503)


class MetricsEndpointTests(AsgiTestCase):
    def test_metrics_require_internal_auth(self) -> None:
        self.assertEqual(self.request("GET", "/metrics").status_code, 401)
        response = self.request("GET", "/metrics", headers={"X-Landa-AI-Service-Token": TOKEN})
        self.assertEqual(response.status_code, 200)
        self.assertIn("ai_rag_http_requests_total", response.text)

    def test_request_ids_are_echoed_and_requests_are_measured(self) -> None:
        labels = {"route": "/healthz", "method": "GET", "status": "200"}
        before = metric("ai_rag_http_requests_total", labels)
        response = self.request("GET", "/healthz", headers={"X-Request-Id": "req-abc", "X-Correlation-Id": "corr-1"})
        self.assertEqual(response.headers["x-request-id"], "req-abc")
        self.assertEqual(response.headers["x-correlation-id"], "corr-1")
        self.assertEqual(metric("ai_rag_http_requests_total", labels), before + 1)


class LifespanTests(unittest.TestCase):
    def test_startup_and_shutdown_manage_pool_clients_and_executors(self) -> None:
        pool = MagicMock()
        pool.close = AsyncMock()
        with (
            patch.object(main.asyncpg, "create_pool", AsyncMock(return_value=pool)) as create_pool,
            patch.object(runtime_service, "create_client", MagicMock(return_value="supabase")),
            patch.object(runtime_service, "require_settings", MagicMock()),
            # SEP-1: the legacy storage client exists only while a service key is configured.
            patch.object(main.settings, "supabase_url", "http://127.0.0.1:54321"),
            patch.object(main.settings, "supabase_service_key", "legacy-service-key"),
            patch.object(runtime_service.schema_guard, "refresh", AsyncMock()),
            patch.object(gemini_infra.client_pool, "clear") as clear_clients,
            patch.object(runtime_service.concurrency, "shutdown") as shutdown_executors,
        ):
            async def scenario() -> None:
                async with main.app.router.lifespan_context(main.app):
                    self.assertIs(runtime_service.db_pool, pool)
                    self.assertEqual(runtime_service.supabase_client, "supabase")
                    self.assertTrue(runtime_service.runtime_state.started)

            asyncio.run(scenario())
        create_pool.assert_awaited_once()
        pool.close.assert_awaited_once()
        clear_clients.assert_called_once()
        shutdown_executors.assert_called_once()
        self.assertIsNone(runtime_service.db_pool)
        self.assertTrue(runtime_service.runtime_state.draining)
        runtime_service.runtime_state.reset()


class DeadlineTests(unittest.TestCase):
    def test_run_with_deadline_maps_only_its_own_expiry(self) -> None:
        before = metric("ai_rag_deadline_exceeded_total", {"route": "/x"})

        async def slow() -> None:
            await asyncio.sleep(1)

        with self.assertRaises(AppError) as raised:
            asyncio.run(deadlines_service.run_with_deadline("/x", 10, slow()))
        self.assertEqual((raised.exception.code, raised.exception.http_status), ("REQUEST_DEADLINE_EXCEEDED", 504))
        self.assertEqual(metric("ai_rag_deadline_exceeded_total", {"route": "/x"}), before + 1)

        async def inner_timeout() -> None:
            raise TimeoutError

        with self.assertRaises(TimeoutError):
            asyncio.run(deadlines_service.run_with_deadline("/x", 10_000, inner_timeout()))

    def test_chat_route_is_bounded_by_its_deadline(self) -> None:
        async def slow_retrieve(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(1)

        request = RagChatRequest(
            tenant_id=TENANT, kb_id=KB, conversation_id="44444444-4444-4444-8444-444444444444",
            target="admin", model="gemini-3.8-flash", embedding_model="gemini-embedding-001",
            system_prompt="persona", user_message="hello", api_key="k",
        )
        with (
            patch.object(main, "retrieve_chunks", slow_retrieve),
            patch.object(main.settings, "chat_deadline_ms", 20),
            self.assertRaises(AppError) as raised,
        ):
            asyncio.run(main.chat(request, pool=object()))  # type: ignore[arg-type]
        self.assertEqual(raised.exception.code, "REQUEST_DEADLINE_EXCEEDED")


class IndexRuntimeTests(unittest.TestCase):
    def index_request(self) -> Any:
        return RagIndexRequest(
            tenant_id=TENANT, kb_id=KB, document_id=DOC, embedding_model="gemini-embedding-001", api_key="k",
        )

    def test_saturated_indexer_answers_service_busy(self) -> None:
        async def scenario() -> str:
            limiter = runtime_service.concurrency.index
            held = [limiter.slot() for _ in range(limiter.limit)]
            for slot in held:
                await slot.__aenter__()
            try:
                with (
                    patch.object(limiter, "acquire_timeout_seconds", 0.01),
                    self.assertRaises(AppError) as raised,
                ):
                    await main.index_document(self.index_request(), pool=object())  # type: ignore[arg-type]
                return raised.exception.code
            finally:
                for slot in held:
                    await slot.__aexit__(None, None, None)

        self.assertEqual(asyncio.run(scenario()), "SERVICE_BUSY")

    def test_index_deadline_marks_the_run_with_a_safe_code(self) -> None:
        row = {"id": DOC, "tenant_id": TENANT, "kb_id": KB, "type": "article", "name": "Doc", "status": "learning",
               "source_info": None, "file_path": None, "content": "Nội dung kiểm thử " * 20}
        marked: list[str] = []

        async def mark(pool: Any, index_id: Any, reason: str) -> None:
            marked.append(reason)

        def slow_structure(sections: Any) -> dict[str, Any]:
            time.sleep(0.2)
            return {"nodes": []}

        with (
            patch.object(main, "load_document", AsyncMock(return_value=row)),
            patch.object(main, "start_index_row", AsyncMock(return_value="55555555-5555-4555-8555-555555555555")),
            patch.object(main, "mark_index_error", mark),
            patch.object(main, "analyze_source_structure", slow_structure),
            patch.object(main.settings, "index_deadline_ms", 50),
        ):
            result = asyncio.run(main.index_document(self.index_request(), pool=object()))  # type: ignore[arg-type]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_reason"], "INDEX_DEADLINE_EXCEEDED")
        self.assertEqual(marked, ["INDEX_DEADLINE_EXCEEDED"])

    def test_cpu_heavy_index_steps_do_not_block_the_event_loop(self) -> None:
        row = {"id": DOC, "tenant_id": TENANT, "kb_id": KB, "type": "article", "name": "Doc", "status": "learning",
               "source_info": None, "file_path": None, "content": "Nội dung kiểm thử " * 20}

        def busy_structure(sections: Any) -> dict[str, Any]:
            time.sleep(0.25)  # stands in for CPU-bound parsing; must run off the loop
            return {"nodes": []}

        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        asyncio_logger = logging.getLogger("asyncio")
        handler = Capture()
        asyncio_logger.addHandler(handler)

        async def scenario() -> None:
            loop = asyncio.get_running_loop()
            loop.slow_callback_duration = 0.1
            await main.index_document(self.index_request(), pool=object())  # type: ignore[arg-type]

        try:
            with (
                patch.object(main, "load_document", AsyncMock(return_value=row)),
                patch.object(main, "start_index_row", AsyncMock(return_value="55555555-5555-4555-8555-555555555555")),
                patch.object(main, "mark_index_error", AsyncMock()),
                patch.object(main, "analyze_source_structure", busy_structure),
                patch.object(provider_service, "embed_texts",
                             AsyncMock(side_effect=RuntimeError("stop after offloaded steps"))),
            ):
                asyncio.run(scenario(), debug=True)
        finally:
            asyncio_logger.removeHandler(handler)
        slow = [record.getMessage() for record in records if "took" in record.getMessage()]
        self.assertEqual(slow, [])


class ProviderRetryPolicyTests(unittest.TestCase):
    def test_backoff_grows_with_jitter_and_respects_cap(self) -> None:
        with patch.object(provider_service, "PROVIDER_TRANSIENT_RETRY_DELAY_SECONDS", 1.0), \
                patch.object(main.settings, "provider_retry_max_ms", 3_000):
            first = provider_service.provider_retry_delay_seconds(0)
            second = provider_service.provider_retry_delay_seconds(1)
            capped = provider_service.provider_retry_delay_seconds(5)
        self.assertTrue(0.8 <= first <= 1.2)
        self.assertTrue(1.6 <= second <= 2.4)
        self.assertEqual(capped, 3.0)

    def test_retry_hint_is_read_from_retry_info_or_retry_after(self) -> None:
        error = Exception("rate limited")
        error.details = {  # type: ignore[attr-defined]
            "error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"}]},
        }
        self.assertEqual(provider_service.provider_retry_hint_seconds(error), 2.0)
        header_error = Exception("rate limited")
        header_error.response = MagicMock(headers={"retry-after": "3"})
        self.assertEqual(provider_service.provider_retry_hint_seconds(header_error), 3.0)
        self.assertIsNone(provider_service.provider_retry_hint_seconds(Exception("none")))

    def test_short_rate_limit_hint_retries_then_succeeds(self) -> None:
        calls = {"count": 0}

        retry_info = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0.01s"}

        class RateLimited(Exception):
            status_code = 429

            def __init__(self, message: str) -> None:
                super().__init__(message)
                self.details = {"error": {"details": [retry_info]}}

        def run() -> str:
            calls["count"] += 1
            if calls["count"] == 1:
                raise RateLimited("RESOURCE_EXHAUSTED")
            return "ok"

        events: list[dict[str, Any]] = []
        result = asyncio.run(
            provider_service.call_provider_with_timeout(run, "m", on_provider_diagnostic=events.append))
        self.assertEqual(result, "ok")
        self.assertIn("provider_rate_limited_retry", [event["event"] for event in events])

    def test_rate_limit_without_hint_is_surfaced_immediately(self) -> None:
        class QuotaExhausted(Exception):
            status_code = 429

        def run() -> str:
            raise QuotaExhausted("RESOURCE_EXHAUSTED")

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                provider_service.call_provider_with_timeout(run, "m", on_provider_diagnostic=lambda event: None))
        self.assertEqual(raised.exception.detail["code"], "AI_PROVIDER_QUOTA_EXHAUSTED")

    def test_provider_calls_are_counted_by_operation_and_outcome(self) -> None:
        labels = {"operation": "embed", "outcome": "success"}
        before = metric("ai_rag_provider_calls_total", labels)
        asyncio.run(provider_service.call_provider_with_timeout(lambda: "ok", "m", operation="embed"))
        self.assertEqual(metric("ai_rag_provider_calls_total", labels), before + 1)

    def test_saturated_provider_limiter_answers_service_busy(self) -> None:
        async def scenario() -> str:
            limiter = runtime_service.concurrency.provider
            held = [limiter.slot() for _ in range(limiter.limit)]
            for slot in held:
                await slot.__aenter__()
            try:
                with (
                    patch.object(limiter, "acquire_timeout_seconds", 0.01),
                    self.assertRaises(AppError) as raised,
                ):
                    await provider_service.call_provider_with_timeout(lambda: "ok", "m")
                return raised.exception.code
            finally:
                for slot in held:
                    await slot.__aexit__(None, None, None)

        self.assertEqual(asyncio.run(scenario()), "SERVICE_BUSY")


class EmbeddingTransportTests(unittest.TestCase):
    def test_embed_batch_uses_pooled_client_and_counts_tokens(self) -> None:
        embedding = MagicMock(values=[0.1, 0.2])
        client = MagicMock()
        client.models.embed_content.return_value = MagicMock(embeddings=[embedding, embedding])
        factory = MagicMock(return_value=client)
        labels = {"operation": "embed", "kind": "embedding"}
        before = metric("ai_rag_provider_tokens_total", labels)
        with patch("app.infra.gemini.genai.Client", factory):
            vectors, usage = asyncio.run(provider_service.embed_text_batch(
                "embed-key", "gemini-embedding-001", ["xin chào", "an toàn"], task_type="RETRIEVAL_DOCUMENT",
            ))
            asyncio.run(provider_service.embed_text_batch("embed-key", "gemini-embedding-001", ["lần hai"]))
        self.assertEqual(vectors, [[0.1, 0.2], [0.1, 0.2]])
        self.assertGreater(usage.embeddingTokens, 0)
        self.assertEqual(factory.call_count, 1)  # second batch reuses the pooled client
        config = client.models.embed_content.call_args_list[0].kwargs["config"]
        self.assertEqual(config.output_dimensionality, 768)
        self.assertGreater(metric("ai_rag_provider_tokens_total", labels), before)


class FallbackMetricTests(unittest.TestCase):
    def test_structured_fallback_results_are_counted_per_stage(self) -> None:
        before = metric("ai_rag_fallbacks_total", {"stage": "unit"})
        deadlines_service.record_fallback("unit", {"content_origin": "structured_fallback"})
        deadlines_service.record_fallback("unit", {"content_origin": "provider_validated"})
        deadlines_service.record_fallback("unit", None)
        self.assertEqual(metric("ai_rag_fallbacks_total", {"stage": "unit"}), before + 1)


class SourceCodeSecurityTests(unittest.TestCase):
    def test_no_direct_outbound_http_clients_in_application_code(self) -> None:
        """SEC-13: only the Gemini SDK, the (legacy) Supabase SDK and the allowlisted storage
        adapter ``app/infra/storage.py`` (SEP-1 signed-URL downloads) talk to the network."""
        forbidden_modules = {"requests", "urllib.request", "urllib3", "aiohttp", "http.client"}
        allowed_httpx = {"app/infra/storage.py"}
        offenders: list[str] = []
        for path in APP_ROOT.rglob("*.py"):
            relative = path.relative_to(APP_ROOT.parent).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(name in forbidden_modules or (name == "httpx" and relative not in allowed_httpx)
                       for name in names):
                    offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_new_runtime_layers_respect_import_direction(self) -> None:
        rules = {
            APP_ROOT / "infra": {"app.main", "app.api", "app.services", "app.idm", "fastapi", "starlette"},
            APP_ROOT / "core": {"app.main", "app.api", "app.services", "app.idm", "app.infra"},
            APP_ROOT / "prompt_safety.py": {"app", "fastapi", "starlette"},
        }
        offenders: list[str] = []
        for target, forbidden in rules.items():
            paths = [target] if target.is_file() else list(target.rglob("*.py"))
            for path in paths:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    module = node.module if isinstance(node, ast.ImportFrom) else None
                    names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                    for name in [module, *names]:
                        if name and any(name == item or name.startswith(item + ".") for item in forbidden):
                            offenders.append(f"{path.name}:{node.lineno}:{name}")
        self.assertEqual(offenders, [])

    def test_index_temp_file_names_never_come_from_user_input(self) -> None:
        """SEC-7: the temp path is server-owned even for hostile display names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            for name in ("../../evil.py", "C:/evil/x.docx", "/abs/evil.pdf", "báo cáo.pdf"):
                path = main.index_document_temp_path(temp_dir, DOC, name)
                self.assertEqual(path.parent, Path(temp_dir).resolve())
                self.assertTrue(path.name.startswith(DOC))


if __name__ == "__main__":
    unittest.main()
