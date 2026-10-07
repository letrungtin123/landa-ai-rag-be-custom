from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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
