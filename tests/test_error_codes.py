from __future__ import annotations

import ast
import unittest
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.core.error_codes import APP_ERROR_CODES, ERROR_CODES, PROVIDER_ERROR_CODES
from app.core.errors import ProviderError, ServiceError
from app.infra.gemini.errors import NON_RETRYABLE_PROVIDER_ERROR_CODES, safe_provider_error_diagnostics
from app.infra.provider_limits import QUOTA_EXHAUSTED_CODE, RATE_LIMITED_CODE

APP_ROOT = Path(__file__).resolve().parents[1] / "app"
# Calls that build an AppError from a literal code and status (``super().__init__`` of its subclasses).
APP_ERROR_BUILDERS = {"AppError", "storage_error", "__init__"}


def _name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    return func.attr if isinstance(func, ast.Attribute) else None


def _literal(node: ast.Call, position: int, keyword: str) -> object:
    value = node.args[position] if len(node.args) > position else next(
        (item.value for item in node.keywords if item.arg == keyword), None)
    return value.value if isinstance(value, ast.Constant) else None


def literal_calls(names: set[str]) -> list[tuple[str, str, object, object]]:
    """(file, called name, literal code, literal status) of every call of ``names`` in the application."""

    found = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Call) and _name(node.func) in names:
                code = _literal(node, 0, "code")
                if isinstance(code, str):
                    found.append((path.relative_to(APP_ROOT).as_posix(), str(_name(node.func)), code,
                                  _literal(node, 1, "http_status")))
    return found


class ErrorCodeRegistryTests(unittest.TestCase):
    def test_every_literal_app_error_code_is_registered_with_its_status(self) -> None:
        # A ``super().__init__`` without a status belongs to an exception that is not an AppError.
        calls = [call for call in literal_calls(APP_ERROR_BUILDERS)
                 if call[2].isupper() and not (call[1] == "__init__" and call[3] is None)]
        self.assertGreaterEqual(len(calls), 8)
        problems = [f"{path}:{code}:{status}" for path, _name_, code, status in calls
                    if code not in APP_ERROR_CODES or APP_ERROR_CODES[code].http_status != status]
        self.assertEqual(problems, [])

    def test_every_provider_error_code_is_registered(self) -> None:
        raised = {code for _path, _name_, code, _status in literal_calls({"ProviderError"})}
        self.assertGreaterEqual(len(raised), 4)
        self.assertLessEqual(raised | {RATE_LIMITED_CODE, QUOTA_EXHAUSTED_CODE}, set(PROVIDER_ERROR_CODES))
        self.assertLessEqual(NON_RETRYABLE_PROVIDER_ERROR_CODES, set(PROVIDER_ERROR_CODES))

    def test_registry_keys_match_their_codes(self) -> None:
        self.assertEqual([key for key, item in ERROR_CODES.items() if key != item.code], [])
        self.assertTrue(all(400 <= item.http_status <= 599 for item in ERROR_CODES.values()))

    def test_unknown_provider_code_is_refused(self) -> None:
        with self.assertRaises(KeyError):
            ProviderError("AI_PROVIDER_SOMETHING_NEW", "message")


class ServiceErrorWireTests(unittest.TestCase):
    """The service errors answer exactly as the ``fastapi.HTTPException`` they replaced."""

    def test_provider_and_service_errors_keep_the_http_exception_envelope(self) -> None:
        cases = {
            "/provider": (ProviderError("AI_PROVIDER_TIMEOUT", "slow"),
                          HTTPException(504, {"code": "AI_PROVIDER_TIMEOUT", "message": "slow"})),
            "/limit": (ProviderError(RATE_LIMITED_CODE, "wait"),
                       HTTPException(503, {"code": RATE_LIMITED_CODE, "message": "wait"})),
            "/plain": (ServiceError(status_code=502, detail="AI không trả JSON hợp lệ."),
                       HTTPException(502, "AI không trả JSON hợp lệ.")),
        }
        app = FastAPI()
        for path, (error, legacy) in cases.items():
            app.add_api_route(path, _raiser(error))
            app.add_api_route(path + "/legacy", _raiser(legacy))
        client = TestClient(app)
        for path in cases:
            with self.subTest(path=path):
                new, old = client.get(path), client.get(path + "/legacy")
                self.assertEqual((new.status_code, new.json(), new.headers.get("content-type")),
                                 (old.status_code, old.json(), old.headers.get("content-type")))
                self.assertNotIn("correlation_id", str(new.json()))

    def test_provider_error_carries_its_registry_entry(self) -> None:
        error = ProviderError("AI_PROVIDER_QUOTA_EXHAUSTED", "top up")
        self.assertIsInstance(error, HTTPException)
        self.assertEqual((error.status_code, error.detail), (503, {"code": "AI_PROVIDER_QUOTA_EXHAUSTED",
                                                                    "message": "top up"}))
        self.assertFalse(error.retryable)
        self.assertTrue(ProviderError("AI_PROVIDER_UNAVAILABLE", "down").retryable)

    def test_diagnostics_keep_the_http_exception_type_label(self) -> None:
        for error in (ProviderError("AI_PROVIDER_UNAVAILABLE", "down"), ServiceError(502, "bad"),
                      HTTPException(503, "legacy")):
            with self.subTest(error=type(error).__name__):
                diagnostics = safe_provider_error_diagnostics(error)
                self.assertEqual(diagnostics["provider_error_type"], "HTTPException")
                if isinstance(error, ServiceError):
                    self.assertIsNone(diagnostics["provider_http_status"])


def _raiser(error: Exception) -> Callable[[], Awaitable[None]]:
    async def endpoint() -> None:
        raise error

    return endpoint


if __name__ == "__main__":
    unittest.main()
