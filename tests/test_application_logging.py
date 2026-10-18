import asyncio
import io
import json
import logging
import unittest
from typing import Any

from app.core.lifespan import RuntimeState
from app.core.logging import (
    JsonFormatter,
    SecretRedactionFilter,
    configure_application_logging,
    correlation_id_var,
    exception_fields,
    exception_type_label,
    log_event,
    log_payload,
)
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


class StructuredEventTests(unittest.TestCase):
    def capture(self, name: str) -> tuple[logging.Logger, io.StringIO]:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(SecretRedactionFilter())
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        return logger, stream

    def test_log_event_puts_fields_on_the_line_and_keeps_the_old_prefix_in_the_message(self) -> None:
        logger, stream = self.capture("tests.structured")
        log_event(logger, logging.WARNING, "unit_ready", family="lesson_author_orchestration_v2",
                  correlation_id="run-1", component_count=3, module="mod_01", level="high")
        line = json.loads(stream.getvalue())
        self.assertEqual((line["event"], line["message"]), ("unit_ready", "lesson_author_orchestration_v2 unit_ready"))
        self.assertEqual((line["component_count"], line["correlation_id"]), (3, "run-1"))
        # Names the LogRecord or the formatter own are kept under a trailing underscore.
        self.assertEqual((line["module_"], line["level_"], line["level"]), ("mod_01", "high", "WARNING"))

    def test_payload_events_and_token_counts_survive_redaction_but_credentials_do_not(self) -> None:
        logger, stream = self.capture("tests.structured_payload")
        log_payload(logger, logging.INFO, "lesson_author_idm", {
            "event": "idm_call_rejected", "spent_output_tokens": 8107, "max_output_tokens": None,
            "observed_usage": {"provider_input_tokens": 12, "provider_output_tokens": 7},
            "token_allowance": {"input_tokens": 1_800_000, "output_tokens": 131_072},
            "service_token": "plain-secret-token", "tokens": "plain-secret-string", "api_key": 123,
        })
        line = json.loads(stream.getvalue())
        self.assertEqual(line["event"], "idm_call_rejected")
        self.assertEqual((line["spent_output_tokens"], line["max_output_tokens"]), (8107, None))
        self.assertEqual(line["observed_usage"], {"provider_input_tokens": 12, "provider_output_tokens": 7})
        self.assertEqual(line["token_allowance"], {"input_tokens": 1_800_000, "output_tokens": 131_072})
        self.assertEqual((line["service_token"], line["tokens"], line["api_key"]), ("<redacted>",) * 3)
        self.assertNotIn("plain-secret", stream.getvalue())

    def test_exception_fields_are_a_sanitised_traceback_without_the_message(self) -> None:
        def inner() -> None:
            raise KeyError("PRIVATE_PROVIDER_TEXT")

        try:
            try:
                inner()
            except KeyError as error:
                raise ValueError("PRIVATE_WRAPPER") from error
        except ValueError as error:
            fields = exception_fields(error)
        self.assertEqual((fields["exception_type"], fields["exception_cause_type"]), ("ValueError", "KeyError"))
        self.assertTrue(fields["exception_cause_frames"][-1].endswith(":inner:" + str(inner.__code__.co_firstlineno + 1)))
        self.assertTrue(all(frame.startswith("tests.test_application_logging:") for frame in fields["exception_frames"]))
        self.assertNotIn("PRIVATE", json.dumps(fields))
        dynamic = type("ProviderError", (Exception,), {})()
        self.assertEqual(exception_type_label(dynamic), "OtherException")

    def test_logged_exceptions_carry_frames_and_no_message(self) -> None:
        logger, stream = self.capture("tests.structured_exception")
        try:
            {}["PRIVATE_KEY_NAME"]  # noqa: B018 - the KeyError under test
        except KeyError:
            logger.exception("unhandled_request_error", extra={"event": "unhandled_request_error"})
        line = json.loads(stream.getvalue())
        self.assertEqual(line["exception_type"], "KeyError")
        self.assertRegex(line["exception_frames"][-1],
                         r"^tests\.test_application_logging:test_logged_exceptions_carry_frames_and_no_message:\d+$")
        self.assertNotIn("PRIVATE_KEY_NAME", stream.getvalue())

    def test_event_correlation_id_fills_in_outside_a_request(self) -> None:
        logger, stream = self.capture("tests.structured_correlation")
        log_event(logger, logging.INFO, "idm_stage_started", correlation_id="from-the-event")
        token = correlation_id_var.set("from-the-header")
        try:
            log_event(logger, logging.INFO, "idm_stage_started", correlation_id="from-the-event")
        finally:
            correlation_id_var.reset(token)
        first, second = (json.loads(line) for line in stream.getvalue().splitlines())
        self.assertEqual((first["correlation_id"], second["correlation_id"]), ("from-the-event", "from-the-header"))
