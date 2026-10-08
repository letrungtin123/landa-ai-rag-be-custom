import asyncio
import io
import json
import logging
import unittest
from typing import Any

from app.core.lifespan import RuntimeState
from app.core.logging import JsonFormatter, SecretRedactionFilter, configure_application_logging
from app.core.request_context import RequestContextMiddleware


class ApplicationLoggingTests(unittest.TestCase):
    def test_application_logger_keeps_info_diagnostics_enabled(self) -> None:
        logger = configure_application_logging("tests.application")
        self.assertTrue(logger.isEnabledFor(logging.INFO))
        self.assertFalse(logger.propagate)
        self.assertTrue(logger.handlers)
        self.assertTrue(any(handler.level <= logging.INFO for handler in logger.handlers))

    def test_json_formatter_has_observability_fields_and_redacts_secrets(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(SecretRedactionFilter())
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("tests.redaction")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

        logger.info(
            "request token=plain-secret",
            extra={"event": "request_finished", "api_key": "AIza012345678901234567890123"},
        )

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["api_key"], "<redacted>")
        self.assertNotIn("plain-secret", payload["message"])
        required = {
            "ts", "level", "logger", "event", "request_id", "correlation_id",
            "route", "duration_ms", "status",
        }
        self.assertEqual(required - payload.keys(), set())


class RequestCompletedLogTests(unittest.TestCase):
    def capture(self, logger_name: str) -> io.StringIO:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(SecretRedactionFilter())
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger(logger_name)
        previous = (list(logger.handlers), logger.propagate, logger.level)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

        def restore() -> None:
            logger.handlers, logger.propagate = previous[0], previous[1]
            logger.setLevel(previous[2])

        self.addCleanup(restore)
        return stream

    def test_completed_request_line_carries_route_status_and_duration(self) -> None:
        stream = self.capture("app.core.request_context")

        async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            scope["route"] = type("Route", (), {"path": "/v1/chat"})()
            await send({"type": "http.response.start", "status": 201, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            return None

        scope = {"type": "http", "method": "POST", "path": "/v1/chat", "headers": [], "query_string": b""}
        asyncio.run(RequestContextMiddleware(app, state=RuntimeState())(scope, receive, send))

        lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
        completed = [line for line in lines if line["event"] == "http_request_completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["route"], "/v1/chat")
        self.assertEqual(completed[0]["status"], 201)
        self.assertIsInstance(completed[0]["duration_ms"], int)
        self.assertGreaterEqual(completed[0]["duration_ms"], 0)
        self.assertEqual(completed[0]["method"], "POST")

    def test_outcome_fields_default_to_null_and_stay_redacted(self) -> None:
        stream = self.capture("tests.request_outcome")
        logger = logging.getLogger("tests.request_outcome")
        logger.info("no_outcome", extra={"event": "no_outcome"})
        logger.info(
            "with_outcome",
            extra={
                "event": "with_outcome",
                "route": "/v1/x?token=plain-secret-value",
                "status": 200,
                "duration_ms": 12,
            },
        )
        first, second = (json.loads(line) for line in stream.getvalue().splitlines())
        self.assertIsNone(first["route"])
        self.assertIsNone(first["status"])
        self.assertIsNone(first["duration_ms"])
        self.assertEqual(second["status"], 200)
        self.assertEqual(second["duration_ms"], 12)
        self.assertNotIn("plain-secret-value", second["route"])
        self.assertIn("<redacted>", second["route"])
