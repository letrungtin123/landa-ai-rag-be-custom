import io
import json
import logging
import unittest

from app.core.logging import JsonFormatter, SecretRedactionFilter, configure_application_logging


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
