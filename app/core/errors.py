from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.error_codes import PROVIDER_ERROR_CODES, ErrorCode
from app.core.logging import correlation_id_var

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppError(Exception):
    code: str
    http_status: int
    safe_message: str

    def __post_init__(self) -> None:
        Exception.__init__(self, self.safe_message)


class DocumentLimitError(AppError):
    def __init__(self, safe_message: str = "Document exceeds a configured processing limit.") -> None:
        super().__init__(
            code="DOCUMENT_LIMIT_EXCEEDED",
            http_status=422,
            safe_message=safe_message,
        )


class ServiceError(HTTPException):
    """A failure of the service layer, answered as ``{"detail": detail}`` with its HTTP status.

    It is the ``fastapi.HTTPException`` the service code raised before refactor wave 2b, under the one
    name the service layer may know: FastAPI's own handler renders it unchanged (same status, same body,
    no correlation id, which only ``AppError`` carries), so no wire envelope changed. Raised with a
    ``{"code", "message"}`` detail (a code of ``app.core.error_codes``) or, in the legacy lesson-author
    validation paths, a plain message.
    """

    # ``provider_error_type`` of the provider diagnostics (app.infra.gemini.errors), which reach Node:
    # these errors were reported there as HTTPException and keep that value.
    diagnostic_type: ClassVar[str] = "HTTPException"


class ProviderError(ServiceError):
    """An AI provider failure with a registered code (``app.core.error_codes.PROVIDER_ERROR_CODES``)."""

    def __init__(self, code: str, message: str) -> None:
        self.error_code: ErrorCode = PROVIDER_ERROR_CODES[code]
        super().__init__(status_code=self.error_code.http_status, detail={"code": code, "message": message})

    @property
    def retryable(self) -> bool:
        return self.error_code.retryable


def error_payload(code: str, message: str, *, include_correlation_id: bool = False) -> dict[str, Any]:
    detail: dict[str, Any] = {"code": code, "message": message}
    correlation_id = correlation_id_var.get()
    if include_correlation_id and correlation_id:
        detail["correlation_id"] = correlation_id
    return {"detail": detail}


async def app_error_handler(_request: Request, error: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=error.http_status,
        content=error_payload(error.code, error.safe_message, include_correlation_id=True),
    )


async def request_validation_error_handler(request: Request, _error: RequestValidationError) -> JSONResponse:
    if request.url.path == "/v1/lesson-author/chapter-checkpoint":
        code = "CHAPTER_CHECKPOINT_CONTRACT_INVALID"
        message = "Invalid chapter checkpoint request."
    elif request.url.path.startswith("/v1/lesson-author/orchestration-v2/"):
        code = "ORCHESTRATION_V2_CONTRACT_INVALID"
        message = "Invalid orchestration request."
    else:
        code = "REQUEST_VALIDATION_FAILED"
        message = "Invalid request."
    return JSONResponse(status_code=422, content=error_payload(code, message))


async def unhandled_error_handler(request: Request, error: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_request_error",
        extra={"event": "unhandled_request_error", "route": request.url.path},
    )
    return JSONResponse(
        status_code=500,
        content=error_payload(
            "INTERNAL_ERROR",
            "The service could not complete the request.",
            include_correlation_id=True,
        ),
    )
