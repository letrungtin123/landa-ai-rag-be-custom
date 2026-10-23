"""uvicorn's loggers write the service's JSON lines through the secret filter (security audit 2026-10-24, L1).

After the app answers 500, Starlette re-raises the exception and uvicorn logs it on ``uvicorn.error`` with
``exc_info``; its default formatter printed the traceback with the exception's message to the PM2 log.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import logging.config
import os
import unittest
from typing import Any
from unittest.mock import patch

import uvicorn

from app.__main__ import uvicorn_log_config, uvicorn_options
from app.core.config import Settings
from app.main import create_app

FAKE_KEY = "AIza" + "Sy0FAKE0FAKE0FAKE0FAKE0FAKE0FAKE000"
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class UvicornLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        loggers = [logging.getLogger(name) for name in UVICORN_LOGGERS]
        saved = [(list(logger.handlers), logger.level, logger.propagate, list(logger.filters)) for logger in loggers]

        def restore() -> None:
            for logger, (handlers, level, propagate, filters) in zip(loggers, saved, strict=True):
                logger.handlers, logger.propagate, logger.filters = handlers, propagate, filters
                logger.setLevel(level)

        self.addCleanup(restore)

    def configure(self) -> io.StringIO:
        stream = io.StringIO()
        config = uvicorn_log_config()
        config["handlers"]["default"]["stream"] = stream
        logging.config.dictConfig(config)
        return stream

    def escaped_exception(self) -> BaseException:
        app = create_app()

        async def boom() -> dict[str, str]:
            raise RuntimeError(f"unexpected failure while handling {FAKE_KEY} and tenant source text")

        app.add_api_route("/test/boom", boom, methods=["GET"])  # in this test's app only
        sent: list[dict[str, Any]] = []
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
                 "path": "/test/boom", "raw_path": b"/test/boom", "query_string": b"", "headers": [],
                 "scheme": "http", "server": ("127.0.0.1", 8010), "client": ("127.0.0.1", 5000), "root_path": "",
                 "app": app}

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        try:
            asyncio.run(app(scope, receive, send))
        except RuntimeError as error:
            self.assertEqual(next(m["status"] for m in sent if m["type"] == "http.response.start"), 500)
            return error
        self.fail("the exception did not escape the app")

    def test_an_escaped_exception_is_logged_as_json_without_its_message(self) -> None:
        stream = self.configure()
        with self.assertLogs("app", level=logging.ERROR):  # the app's own unhandled_request_error line
            error = self.escaped_exception()
        # What uvicorn's protocol does with it (uvicorn/protocols/http/*_impl.py, run_asgi).
        logging.getLogger("uvicorn.error").error("Exception in ASGI application\n", exc_info=error)
        output = stream.getvalue()
        line = json.loads(output.splitlines()[-1])
        self.assertEqual((line["logger"], line["level"], line["exception_type"]), ("uvicorn.error", "ERROR",
                                                                                   "RuntimeError"))
        self.assertTrue(line["exception_frames"])
        self.assertNotIn(FAKE_KEY, output)
        self.assertNotIn("tenant source text", output)

    def test_messages_are_redacted(self) -> None:
        stream = self.configure()
        logging.getLogger("uvicorn.error").warning("Invalid HTTP request received: %s", f"/v1/chat?key={FAKE_KEY}")
        logging.getLogger("uvicorn").info("Started server process [%d]", 1234)
        first, second = (json.loads(line) for line in stream.getvalue().splitlines())
        self.assertNotIn(FAKE_KEY, first["message"])
        self.assertIn("<redacted>", first["message"])
        self.assertEqual(second["message"], "Started server process [1234]")

    def test_lines_are_named_uvicorn_log_without_the_colour_copy(self) -> None:
        stream = self.configure()
        # uvicorn passes an ANSI-coloured copy of its message as the extra field color_message.
        logging.getLogger("uvicorn.error").info("Uvicorn running on %s", "http://127.0.0.1:8010",
                                                extra={"color_message": "Uvicorn running on \x1b[1m%s\x1b[0m"})
        line = json.loads(stream.getvalue().splitlines()[-1])
        self.assertEqual((line["event"], line["message"]),
                         ("uvicorn_log", "Uvicorn running on http://127.0.0.1:8010"))
        self.assertNotIn("color_message", line)

    def test_uvicorn_accepts_the_options(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            options = uvicorn_options(Settings(environment="test"))
        self.assertEqual(options["log_config"], uvicorn_log_config())
        config = uvicorn.Config("app.main:app", **options)  # applies the logging configuration
        self.assertEqual(config.log_level, "info")
        handler = logging.getLogger("uvicorn").handlers[0]
        self.assertEqual(type(handler.formatter).__name__, "JsonFormatter")
        self.assertEqual([type(item).__name__ for item in handler.filters],
                         ["UvicornEventFilter", "SecretRedactionFilter"])
        self.assertFalse(logging.getLogger("uvicorn.error").handlers)  # it propagates to "uvicorn"


if __name__ == "__main__":
    unittest.main()
