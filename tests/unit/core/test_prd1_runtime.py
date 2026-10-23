from __future__ import annotations

import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
from google import genai
from google.genai import types as genai_types

from app.__main__ import uvicorn_options
from app.core import metrics
from app.core.concurrency import ConcurrencyRuntime, WorkloadLimiter
from app.core.config import Settings
from app.core.errors import AppError
from app.core.lifespan import RuntimeState, build_lifespan
from app.core.logging import correlation_id_var, request_id_var
from app.core.request_context import DisconnectCancellationMiddleware, RequestContextMiddleware, route_template
from app.infra.gemini.pool import GeminiClientPool
from app.infra.gemini.transport import call_provider_with_timeout
from app.services import provider as provider_service


def metric_value(name: str, labels: dict[str, str]) -> float:
    value = metrics.REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


def http_scope(
    path: str = "/v1/chat",
    method: str = "POST",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> dict[str, Any]:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
        "query_string": b"",
    }


class WorkloadLimiterTests(unittest.TestCase):
    def test_limits_concurrency_and_rejects_when_saturated(self) -> None:
        limiter = WorkloadLimiter("index", 1, acquire_timeout_seconds=0.05)
        before = metric_value("ai_rag_limiter_rejections_total", {"workload": "index"})

        async def scenario() -> str:
            async with limiter.slot():
                with self.assertRaises(AppError) as raised:
                    async with limiter.slot():
                        pass
                return raised.exception.code

        self.assertEqual(asyncio.run(scenario()), "SERVICE_BUSY")
        self.assertEqual(metric_value("ai_rag_limiter_rejections_total", {"workload": "index"}), before + 1)

    def test_each_event_loop_gets_its_own_semaphore(self) -> None:
        limiter = WorkloadLimiter("provider", 1, acquire_timeout_seconds=0.05)

        async def hold_and_release() -> None:
            async with limiter.slot():
                await asyncio.sleep(0)

        asyncio.run(hold_and_release())
        asyncio.run(hold_and_release())

    def test_slot_wait_is_capped_by_caller_budget(self) -> None:
        limiter = WorkloadLimiter("provider", 1, acquire_timeout_seconds=30)

        async def scenario() -> float:
            async with limiter.slot():
                loop = asyncio.get_running_loop()
                started = loop.time()
                with self.assertRaises(AppError):
                    async with limiter.slot(max_wait_seconds=0.01):
                        pass
                return loop.time() - started

        self.assertLess(asyncio.run(scenario()), 1.0)

    def test_rejects_invalid_limit(self) -> None:
        with self.assertRaises(ValueError):
            WorkloadLimiter("cpu", 0, acquire_timeout_seconds=1)


class ConcurrencyRuntimeTests(unittest.TestCase):
    def runtime(self) -> ConcurrencyRuntime:
        return ConcurrencyRuntime(
            provider_limit=2,
            index_limit=1,
            cpu_workers=2,
            acquire_timeout_seconds=1,
            extraction_executor="thread",
        )

    def test_run_cpu_executes_off_loop_and_keeps_request_context(self) -> None:
        runtime = self.runtime()

        def work(value: int) -> tuple[int, str, str | None]:
            return value * 2, threading.current_thread().name, request_id_var.get()

        async def scenario() -> tuple[int, str, str | None]:
            token = request_id_var.set("req-123")
            try:
                return await runtime.run_cpu(work, 21)
            finally:
                request_id_var.reset(token)

        try:
            result, thread_name, request_id = asyncio.run(scenario())
        finally:
            runtime.shutdown()
        self.assertEqual(result, 42)
        self.assertTrue(thread_name.startswith("ai-rag-cpu"))
        self.assertEqual(request_id, "req-123")

    def test_run_extraction_uses_thread_executor_by_default(self) -> None:
        runtime = self.runtime()
        try:
            name = asyncio.run(runtime.run_extraction(lambda: threading.current_thread().name))
        finally:
            runtime.shutdown()
        self.assertTrue(name.startswith("ai-rag-cpu"))

    def test_shutdown_is_idempotent(self) -> None:
        runtime = self.runtime()
        asyncio.run(runtime.run_cpu(int, "1"))
        runtime.shutdown()
        runtime.shutdown()

    def test_provider_calls_run_on_their_own_threads_one_per_provider_slot(self) -> None:
        # Audit 2026-10-24, P5: the SDK calls used the default executor that every asyncio.to_thread shares.
        runtime = self.runtime()  # provider limit 2
        lock = threading.Lock()
        running = {"now": 0, "peak": 0}
        names: set[str] = set()

        def call() -> str | None:
            with lock:
                running["now"] += 1
                running["peak"] = max(running["peak"], running["now"])
                names.add(threading.current_thread().name)
            time.sleep(0.05)
            with lock:
                running["now"] -= 1
            return request_id_var.get()

        async def scenario() -> list[str | None]:
            token = request_id_var.set("req-provider")
            try:
                return await asyncio.gather(*(runtime.run_provider_call(call) for _ in range(6)))
            finally:
                request_id_var.reset(token)

        try:
            results = asyncio.run(scenario())
            self.assertIsNotNone(runtime._provider_executor)
        finally:
            runtime.shutdown()
        self.assertEqual(results, ["req-provider"] * 6)
        self.assertEqual(running["peak"], 2)
        self.assertTrue(names and all(name.startswith("ai-rag-provider") for name in names))
        self.assertIsNone(runtime._provider_executor)  # shut down with the service (services.runtime.shutdown)

    def test_provider_calls_do_not_wait_for_the_default_executor(self) -> None:
        runtime = self.runtime()
        release = threading.Event()

        async def scenario() -> str:
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
            # Holds the default executor's only thread, like a long storage download or structure analysis.
            blocked = asyncio.ensure_future(asyncio.to_thread(release.wait, 5))
            await asyncio.sleep(0.01)
            try:
                return await asyncio.wait_for(runtime.run_provider_call(lambda: threading.current_thread().name), 2)
            finally:
                release.set()
                await blocked

        try:
            self.assertTrue(asyncio.run(scenario()).startswith("ai-rag-provider"))
        finally:
            runtime.shutdown()

    def test_the_transport_runs_sdk_calls_on_the_provider_threads(self) -> None:
        name = asyncio.run(call_provider_with_timeout(lambda: threading.current_thread().name, "gemini-test",
                                                      request_timeout_ms=5_000))
        self.assertTrue(name.startswith("ai-rag-provider"))


class LifespanTests(unittest.TestCase):
    def test_drain_returns_immediately_when_idle(self) -> None:
        state = RuntimeState()
        self.assertTrue(asyncio.run(state.drain(0.01)))
        self.assertTrue(state.draining)

    def test_drain_waits_for_in_flight_request(self) -> None:
        state = RuntimeState()

        async def scenario() -> bool:
            state.begin_request()

            async def finish() -> None:
                await asyncio.sleep(0.01)
                state.end_request()

            task = asyncio.create_task(finish())
            drained = await state.drain(1)
            await task
            return drained

        self.assertTrue(asyncio.run(scenario()))

    def test_drain_times_out_with_stuck_request(self) -> None:
        state = RuntimeState()
        state.begin_request()
        self.assertFalse(asyncio.run(state.drain(0.01)))

    def test_lifespan_runs_startup_then_drain_then_shutdown(self) -> None:
        events: list[str] = []
        state = RuntimeState()

        async def startup() -> None:
            events.append("startup")

        async def shutdown() -> None:
            events.append(f"shutdown draining={state.draining}")

        lifespan = build_lifespan(state=state, startup=startup, shutdown=shutdown, grace_seconds=0.01)

        async def scenario() -> None:
            async with lifespan(object()):
                events.append(f"serving started={state.started}")

        asyncio.run(scenario())
        self.assertEqual(events, ["startup", "serving started=True", "shutdown draining=True"])


class RequestContextMiddlewareTests(unittest.TestCase):
    def run_middleware(
        self,
        scope: dict[str, Any],
        *,
        state: RuntimeState | None = None,
        status: int = 200,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        seen: dict[str, Any] = {}
        sent: list[dict[str, Any]] = []

        async def app(inner_scope: dict[str, Any], receive: Any, send: Any) -> None:
            seen["request_id"] = request_id_var.get()
            seen["correlation_id"] = correlation_id_var.get()
            seen["in_flight"] = (state or RuntimeState()).in_flight
            inner_scope["route"] = type("Route", (), {"path": "/v1/chat"})()
            await send({"type": "http.response.start", "status": status, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        middleware = RequestContextMiddleware(app, state=state or RuntimeState())
        asyncio.run(middleware(scope, receive, send))
        return sent, seen

    def test_propagates_valid_ids_and_echoes_headers(self) -> None:
        scope = http_scope(headers=[(b"x-request-id", b"req-1"), (b"x-correlation-id", b"corr-1")])
        sent, seen = self.run_middleware(scope)
        self.assertEqual(seen["request_id"], "req-1")
        self.assertEqual(seen["correlation_id"], "corr-1")
        headers = dict(sent[0]["headers"])
        self.assertEqual(headers[b"x-request-id"], b"req-1")
        self.assertEqual(headers[b"x-correlation-id"], b"corr-1")
        self.assertIsNone(request_id_var.get())

    def test_replaces_unsafe_request_id_and_falls_back_to_landa_header(self) -> None:
        _, seen = self.run_middleware(http_scope(headers=[(b"x-request-id", b"bad id\nnext")]))
        self.assertNotEqual(seen["request_id"], "bad id\nnext")
        self.assertEqual(len(seen["request_id"]), 36)
        _, seen = self.run_middleware(http_scope(headers=[(b"x-landa-request-id", b"landa-1")]))
        self.assertEqual(seen["request_id"], "landa-1")

    def test_counts_in_flight_work_but_not_operational_probes(self) -> None:
        state = RuntimeState()
        _, seen = self.run_middleware(http_scope(), state=state)
        self.assertEqual(seen["in_flight"], 1)
        self.assertEqual(state.in_flight, 0)
        _, seen = self.run_middleware(http_scope(path="/healthz", method="GET"), state=state)
        self.assertEqual(seen["in_flight"], 0)

    def test_counts_in_flight_work_per_route_class_but_not_the_meta_poll(self) -> None:
        # Audit 2026-10-24, O1: /v1/meta reports these counts, so it is not counted itself.
        state = RuntimeState()
        seen: list[dict[str, Any]] = []

        async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            seen.append(state.in_flight_summary())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            return None

        paths = ["/v1/lesson-author/orchestration-v2/unit", "/v1/lesson-author/org-context/distill",
                 "/v1/lesson-author/proposal", "/v1/chat", "/v1/kb/documents/index", "/v1/unknown", "/v1/meta",
                 "/healthz", "/metrics"]
        for path in paths:
            asyncio.run(RequestContextMiddleware(app, state=state)(http_scope(path=path), receive, send))
        busy = [{name: count for name, count in summary["by_route_class"].items() if count} for summary in seen]
        self.assertEqual(busy, [{"ai_id": 1}, {"ai_id": 1}, {"lesson_author_legacy": 1}, {"chat": 1},
                                {"knowledge_base": 1}, {"other": 1}, {}, {}, {}])
        self.assertEqual([summary["total"] for summary in seen], [1, 1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(state.in_flight_summary()["total"], 0)
        self.assertEqual(set(state.in_flight_summary()["by_route_class"].values()), {0})
        state.begin_request("not-a-class")
        self.assertEqual(state.in_flight_summary()["by_route_class"]["other"], 1)

    def test_records_metrics_with_route_template(self) -> None:
        labels = {"route": "/v1/chat", "method": "POST", "status": "201"}
        before = metric_value("ai_rag_http_requests_total", labels)
        self.run_middleware(http_scope(), status=201)
        self.assertEqual(metric_value("ai_rag_http_requests_total", labels), before + 1)

    def test_unknown_methods_share_one_metric_label(self) -> None:
        # Any client can send any method token; each one used to add new series (audit 2026-10-24, L3).
        other = {"route": "/v1/chat", "method": "OTHER", "status": "200"}
        before = metric_value("ai_rag_http_requests_total", other)
        for method in ("X" + "Q" * 30, "get", "PROPFIND", "post"):
            self.run_middleware(http_scope(method=method))
        self.assertEqual(metric_value("ai_rag_http_requests_total", other), before + 4)
        methods = {sample.labels["method"] for family in metrics.REGISTRY.collect()
                   if family.name.startswith("ai_rag_http_request") for sample in family.samples
                   if "method" in sample.labels}
        self.assertLessEqual(methods, {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "OTHER"})
        for method in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
            labels = {"route": "/v1/chat", "method": method, "status": "200"}
            before = metric_value("ai_rag_http_requests_total", labels)
            self.run_middleware(http_scope(method=method))
            self.assertEqual(metric_value("ai_rag_http_requests_total", labels), before + 1)

    def test_route_template_never_uses_unmatched_raw_paths(self) -> None:
        self.assertEqual(route_template({"path": "/v1/kb/123"}), "unmatched")
        self.assertEqual(route_template({"path": "/healthz"}), "/healthz")

    def test_non_http_scope_passes_through(self) -> None:
        called: list[str] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            called.append(scope["type"])

        asyncio.run(RequestContextMiddleware(app, state=RuntimeState())({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
        self.assertEqual(called, ["lifespan"])


class DisconnectCancellationMiddlewareTests(unittest.TestCase):
    def test_cancels_route_work_when_client_disconnects(self) -> None:
        state: dict[str, Any] = {"cancelled": False, "body": None}
        before = metric_value("ai_rag_http_client_disconnects_total", {"route": "unmatched"})

        async def app(scope: Any, receive: Any, send: Any) -> None:
            message = await receive()
            state["body"] = message["body"]
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise

        async def scenario() -> None:
            messages = [{"type": "http.request", "body": b'{"a":1}', "more_body": False}]

            async def receive() -> dict[str, Any]:
                if messages:
                    return messages.pop(0)
                await asyncio.sleep(0.02)
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                raise AssertionError("no response may be sent after disconnect")

            await DisconnectCancellationMiddleware(app)(http_scope(), receive, send)

        asyncio.run(scenario())
        self.assertTrue(state["cancelled"])
        self.assertEqual(state["body"], b'{"a":1}')
        self.assertEqual(metric_value("ai_rag_http_client_disconnects_total", {"route": "unmatched"}), before + 1)

    def test_completed_response_is_not_cancelled(self) -> None:
        sent: list[dict[str, Any]] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            first = await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": first["body"]})

        async def scenario() -> None:
            complete = asyncio.Event()
            chunks = [
                {"type": "http.request", "body": b"ab", "more_body": True},
                {"type": "http.request", "body": b"cd", "more_body": False},
            ]

            async def receive() -> dict[str, Any]:
                if chunks:
                    return chunks.pop(0)
                await complete.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                sent.append(message)
                if message["type"] == "http.response.body":
                    complete.set()

            await DisconnectCancellationMiddleware(app)(http_scope(), receive, send)

        asyncio.run(scenario())
        self.assertEqual(sent[-1]["body"], b"ab")

    def test_route_errors_propagate(self) -> None:
        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            raise RuntimeError("boom")

        async def scenario() -> None:
            sent = False

            async def receive() -> dict[str, Any]:
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": b"", "more_body": False}
                await asyncio.sleep(1)
                return {"type": "http.disconnect"}

            await DisconnectCancellationMiddleware(app)(http_scope(), receive, lambda message: asyncio.sleep(0))

        with self.assertRaises(RuntimeError):
            asyncio.run(scenario())

    def test_disconnect_before_body_skips_route(self) -> None:
        called: list[bool] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            called.append(True)

        async def receive() -> dict[str, Any]:
            return {"type": "http.disconnect"}

        asyncio.run(DisconnectCancellationMiddleware(app)(http_scope(), receive, lambda message: asyncio.sleep(0)))
        self.assertEqual(called, [])

    def test_get_and_non_v1_requests_pass_through(self) -> None:
        called: list[str] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            called.append(scope["path"])

        middleware = DisconnectCancellationMiddleware(app)
        asyncio.run(middleware(http_scope(path="/healthz", method="GET"), None, None))  # type: ignore[arg-type]
        asyncio.run(middleware(http_scope(path="/other", method="POST"), None, None))  # type: ignore[arg-type]
        self.assertEqual(called, ["/healthz", "/other"])


class FakeSdkClient:
    """``genai.Client`` stand-in: records each call's config and fails a call made after close()."""

    def __init__(self, calls: list[dict[str, Any]], **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = 0
        self.calls = calls
        self.models = self

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        if self.closed:
            raise RuntimeError("client used after close")
        self.calls.append({"client": self, "model": model, "contents": contents, "config": config})
        return SimpleNamespace(text="{}", parsed=None, candidates=[], prompt_feedback=None, usage_metadata=None)

    def close(self) -> None:
        self.closed += 1


def fake_factory(calls: list[dict[str, Any]], created: list[FakeSdkClient]) -> Any:
    def make(**kwargs: Any) -> FakeSdkClient:
        client = FakeSdkClient(calls, **kwargs)
        created.append(client)
        return client

    return make


def hold_lease(pool: GeminiClientPool, key: str, inside: threading.Event, release: threading.Event) -> threading.Thread:
    """A worker thread inside an SDK call on ``key``'s client until ``release`` is set."""

    def run() -> None:
        with pool.lease(key):
            inside.set()
            release.wait(5)

    thread = threading.Thread(target=run)
    thread.start()
    if not inside.wait(5):
        raise AssertionError("lease holder did not start")
    return thread


class GeminiClientPoolTests(unittest.TestCase):
    def test_reuses_one_client_per_key_without_storing_raw_key(self) -> None:
        created: list[FakeSdkClient] = []
        pool = GeminiClientPool(max_clients=4)
        with patch("app.infra.gemini.pool.genai.Client", fake_factory([], created)):
            with pool.lease("secret-key") as first:
                pass
            with pool.lease("secret-key") as second:
                pass
            with pool.lease("other-key") as other:
                pass
        self.assertIs(first, second)
        self.assertIsNot(first, other)
        self.assertEqual(len(created), 2)
        # The client carries no timeout: every call passes its own deadline.
        self.assertEqual(created[0].kwargs, {"api_key": "secret-key"})
        self.assertNotIn("secret-key", repr(list(pool._clients)))

    def test_same_key_reuses_one_client_across_call_timeouts(self) -> None:
        # IDM passes min(180000, remaining_ms): the last calls of a task each get another timeout.
        calls: list[dict[str, Any]] = []
        created: list[FakeSdkClient] = []
        timeouts = [180_000, 9_731, 4_002, 1_000]
        with patch("app.infra.gemini.pool.client_pool", GeminiClientPool()), \
                patch("app.infra.gemini.pool.genai.Client", fake_factory(calls, created)):
            for timeout_ms in timeouts:
                asyncio.run(provider_service.generate_content("same-key", "gemini-3.5-flash", "prompt",
                                                              max_output_tokens=64, request_timeout_ms=timeout_ms))
        self.assertEqual(len(created), 1)
        self.assertEqual([call["client"] for call in calls], [created[0]] * len(timeouts))
        self.assertEqual([call["config"]["http_options"].timeout for call in calls], timeouts)

    def test_per_call_deadline_reaches_the_http_request(self) -> None:
        # The real SDK on an in-memory transport: each request carries its own call's deadline.
        seen: list[tuple[Any, str | None]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.extensions["timeout"]["read"], request.headers.get("x-server-timeout")))
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"},
                                                             "finishReason": "STOP"}]})

        real_client = genai.Client
        transport = httpx.Client(transport=httpx.MockTransport(handler))

        def make(**kwargs: Any) -> Any:
            return real_client(**kwargs, http_options=genai_types.HttpOptions(httpx_client=transport))

        with patch("app.infra.gemini.pool.client_pool", GeminiClientPool()), \
                patch("app.infra.gemini.pool.genai.Client", make):
            for timeout_ms in (180_000, 1_234):
                text, _usage = asyncio.run(provider_service.generate_content(
                    "offline-key", "gemini-3.5-flash", "prompt", max_output_tokens=64, request_timeout_ms=timeout_ms))
                self.assertEqual(text, "ok")
        transport.close()
        self.assertEqual(seen, [(180.0, "180"), (1.234, "2")])

    def test_embeddings_carry_the_configured_request_timeout(self) -> None:
        client = MagicMock()
        client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5])])
        with patch("app.infra.gemini.pool.client_pool", GeminiClientPool()), \
                patch("app.infra.gemini.pool.genai.Client", MagicMock(return_value=client)), \
                patch("app.core.config.settings.provider_request_timeout_ms", 45_000):
            asyncio.run(provider_service.embed_text_batch("embed-key", "gemini-embedding-001", ["xin chào"]))
        config = client.models.embed_content.call_args.kwargs["config"]
        self.assertEqual(config.http_options.timeout, 45_000)

    def test_eviction_never_closes_a_client_in_use(self) -> None:
        created: list[FakeSdkClient] = []
        pool = GeminiClientPool(max_clients=1)
        inside, release = threading.Event(), threading.Event()
        with patch("app.infra.gemini.pool.genai.Client", fake_factory([], created)):
            holder = hold_lease(pool, "a", inside, release)
            with pool.lease("b"):
                pass  # evicts "a" from the cache while a call still holds it
            self.assertEqual(len(pool), 1)
            self.assertEqual(created[0].closed, 0)
            release.set()
            holder.join(5)
        self.assertEqual(created[0].closed, 1)  # closed once its last call returned
        self.assertEqual(created[1].closed, 0)  # "b" is still cached
        pool.clear()
        self.assertEqual(created[1].closed, 1)

    def test_clear_closes_a_leased_client_only_after_its_call(self) -> None:
        created: list[FakeSdkClient] = []
        pool = GeminiClientPool()
        inside, release = threading.Event(), threading.Event()
        with patch("app.infra.gemini.pool.genai.Client", fake_factory([], created)):
            holder = hold_lease(pool, "a", inside, release)
            pool.clear()
            self.assertEqual((len(pool), created[0].closed), (0, 0))
            release.set()
            holder.join(5)
        self.assertEqual(created[0].closed, 1)

    def test_concurrent_calls_with_eviction_never_use_a_closed_client(self) -> None:
        calls: list[dict[str, Any]] = []
        created: list[FakeSdkClient] = []
        pool = GeminiClientPool(max_clients=2)
        failures: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                for step in range(40):
                    with pool.lease(f"key-{(index + step) % 5}") as client:
                        client.models.generate_content(model="m", contents="p", config={})
            except BaseException as error:  # collected and asserted below
                failures.append(error)

        with patch("app.infra.gemini.pool.genai.Client", fake_factory(calls, created)):
            threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            pool.clear()
        self.assertEqual(failures, [])
        self.assertEqual(len(calls), 8 * 40)
        self.assertEqual({client.closed for client in created}, {1})  # every client closed exactly once

    def test_client_built_by_a_racing_thread_is_closed_unused(self) -> None:
        created: list[FakeSdkClient] = []
        barrier = threading.Barrier(2)
        make = fake_factory([], created)

        def slow_make(**kwargs: Any) -> FakeSdkClient:
            barrier.wait(5)  # both threads found the cache empty before either stores its client
            return make(**kwargs)

        pool = GeminiClientPool()
        leased: list[Any] = []

        def call() -> None:
            with pool.lease("same") as client:
                leased.append(client)

        with patch("app.infra.gemini.pool.genai.Client", slow_make):
            threads = [threading.Thread(target=call) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        self.assertEqual(len(created), 2)
        self.assertIs(leased[0], leased[1])
        self.assertEqual(sorted(client.closed for client in created), [0, 1])
        self.assertEqual(len(pool), 1)

    def test_patched_factory_never_receives_stale_cached_client(self) -> None:
        pool = GeminiClientPool()
        with patch("app.infra.gemini.pool.genai.Client", MagicMock(return_value="first")), pool.lease("k") as first:
            self.assertEqual(first, "first")
        with patch("app.infra.gemini.pool.genai.Client", MagicMock(return_value="second")), pool.lease("k") as second:
            self.assertEqual(second, "second")

    def test_close_failures_are_swallowed(self) -> None:
        client = MagicMock()
        client.close.side_effect = RuntimeError("close failed")
        pool = GeminiClientPool(max_clients=1)
        with patch("app.infra.gemini.pool.genai.Client", MagicMock(return_value=client)), pool.lease("a"):
            pass
        pool.clear()
        client.close.assert_called_once()

    def test_rejects_invalid_size(self) -> None:
        with self.assertRaises(ValueError):
            GeminiClientPool(max_clients=0)


class EntrypointTests(unittest.TestCase):
    def test_uvicorn_options_come_from_settings(self) -> None:
        value = Settings(
            AI_RAG_HOST="0.0.0.0",  # noqa: S104 - settings value under test, not a bind
            AI_RAG_PORT=9000,
            AI_RAG_WORKERS=3,
            AI_RAG_SHUTDOWN_GRACE_SECONDS=30,
            AI_RAG_KEEP_ALIVE_TIMEOUT_SECONDS=20,
            AI_RAG_FORWARDED_ALLOW_IPS="10.0.0.1",
        )
        options = uvicorn_options(value)
        self.assertEqual(options["host"], "0.0.0.0")  # noqa: S104 - assertion only
        self.assertEqual(options["port"], 9000)
        self.assertEqual(options["workers"], 3)
        self.assertEqual(options["timeout_graceful_shutdown"], 30)
        self.assertEqual(options["timeout_keep_alive"], 20)
        self.assertEqual(options["forwarded_allow_ips"], "10.0.0.1")
        self.assertTrue(options["proxy_headers"])

    def test_uvicorn_caps_open_connections_and_running_requests(self) -> None:
        # Audit 2026-10-24, L4: a generous ceiling, far above the provider (8), index (2) and CPU limiters.
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(uvicorn_options(Settings(environment="test"))["limit_concurrency"], 128)
        self.assertEqual(uvicorn_options(Settings(AI_RAG_LIMIT_CONCURRENCY=300))["limit_concurrency"], 300)
        for invalid in (0, 15, 100_001):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                Settings(AI_RAG_LIMIT_CONCURRENCY=invalid)

    def test_metrics_exposition_lists_service_metrics(self) -> None:
        body, content_type = metrics.render_metrics()
        self.assertIn(b"ai_rag_http_requests_total", body)
        self.assertIn(b"ai_rag_provider_calls_total", body)
        self.assertTrue(content_type.startswith("text/plain"))


if __name__ == "__main__":
    unittest.main()
