from __future__ import annotations

import asyncio
import threading
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from app.__main__ import uvicorn_options
from app.core import metrics
from app.core.concurrency import ConcurrencyRuntime, WorkloadLimiter
from app.core.config import Settings
from app.core.errors import AppError
from app.core.lifespan import RuntimeState, build_lifespan
from app.core.logging import correlation_id_var, request_id_var
from app.core.request_context import DisconnectCancellationMiddleware, RequestContextMiddleware, route_template
from app.infra.gemini import GeminiClientPool


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

    def test_records_metrics_with_route_template(self) -> None:
        labels = {"route": "/v1/chat", "method": "POST", "status": "201"}
        before = metric_value("ai_rag_http_requests_total", labels)
        self.run_middleware(http_scope(), status=201)
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


class GeminiClientPoolTests(unittest.TestCase):
    def test_reuses_client_per_key_and_timeout_without_storing_raw_key(self) -> None:
        factory = MagicMock(side_effect=lambda **kwargs: MagicMock(name="client"))
        pool = GeminiClientPool(max_clients=4)
        with patch("app.infra.gemini.genai.Client", factory):
            first = pool.get("secret-key", 1000)
            second = pool.get("secret-key", 1000)
            other_timeout = pool.get("secret-key", 2000)
        self.assertIs(first, second)
        self.assertIsNot(first, other_timeout)
        self.assertEqual(factory.call_count, 2)
        self.assertNotIn("secret-key", repr(list(pool._clients)))

    def test_evicts_and_closes_least_recently_used_client(self) -> None:
        clients: list[MagicMock] = []

        def make(**kwargs: Any) -> MagicMock:
            client = MagicMock(name=f"client{len(clients)}")
            clients.append(client)
            return client

        pool = GeminiClientPool(max_clients=1)
        with patch("app.infra.gemini.genai.Client", MagicMock(side_effect=make)):
            pool.get("a", 1)
            pool.get("b", 1)
        self.assertEqual(len(pool), 1)
        clients[0].close.assert_called_once()
        pool.clear()
        clients[1].close.assert_called_once()
        self.assertEqual(len(pool), 0)

    def test_patched_factory_never_receives_stale_cached_client(self) -> None:
        pool = GeminiClientPool()
        with patch("app.infra.gemini.genai.Client", MagicMock(return_value="first")):
            self.assertEqual(pool.get("k", 1), "first")
        with patch("app.infra.gemini.genai.Client", MagicMock(return_value="second")):
            self.assertEqual(pool.get("k", 1), "second")

    def test_close_failures_are_swallowed(self) -> None:
        client = MagicMock()
        client.close.side_effect = RuntimeError("close failed")
        pool = GeminiClientPool(max_clients=1)
        with patch("app.infra.gemini.genai.Client", MagicMock(return_value=client)):
            pool.get("a", 1)
        pool.clear()

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

    def test_metrics_exposition_lists_service_metrics(self) -> None:
        body, content_type = metrics.render_metrics()
        self.assertIn(b"ai_rag_http_requests_total", body)
        self.assertIn(b"ai_rag_provider_calls_total", body)
        self.assertTrue(content_type.startswith("text/plain"))


if __name__ == "__main__":
    unittest.main()
