"""IDM runtime wiring of orchestration-v2: provider transport and error mapping."""

from __future__ import annotations

import logging
from time import monotonic
from typing import Any, Literal, Protocol

from pydantic import SecretStr

from app.core import metrics
from app.core.config import settings
from app.core.errors import AppError, ServiceError
from app.core.handled_exceptions import report_handled_exception
from app.core.logging import SERVICE_LOGGER_NAME, log_event, log_payload
from app.idm.policy import IDM_PROMPT_POLICY_VERSION
from app.idm.prompt_manifest import prompt_templates_hash
from app.idm.runtime import (
    IDM_LOG_FAMILY,
    RUN_STOPPING_PROVIDER_CODES,
    TRANSIENT_PROVIDER_CODES,
    IdmError,
    IdmProviderError,
    IdmRuntime,
    IdmTokenAllowance,
    current_provider_stage,
)
from app.infra.gemini.calls import require_provider_api_key
from app.infra.gemini.errors import http_error_detail, provider_http_error_status, safe_provider_error_diagnostics
from app.schemas.common import AiUsage
from app.services import provider
from app.services import runtime as service_runtime
from app.services.orchestration_v2.common import _orchestration_v2_http_error

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def _idm_generate(api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, AiUsage]:
    """Provider transport for ``app.idm``: map service errors onto ``IdmProviderError``.

    An exhausted key or a rate limit that outlived the bounded wait is terminal for the
    stage (``RUN_STOPPING_PROVIDER_CODES``): it reaches Node as a 503 with that code
    instead of turning into deterministic course content.
    """

    try:
        with metrics.provider_stage(current_provider_stage() or "idm"):
            return await provider.generate_content(api_key, model, prompt, **options)
    except ServiceError as error:
        code = str(http_error_detail(error).get("code") or "AI_PROVIDER_UNAVAILABLE")
        if code in RUN_STOPPING_PROVIDER_CODES:
            raise IdmProviderError(code, terminal=True, http_status=503) from error
        raise IdmProviderError(code, terminal=code not in TRANSIENT_PROVIDER_CODES,
                               http_status=error.status_code) from error
    except AppError as error:
        raise IdmProviderError(error.code, terminal=False, http_status=error.http_status) from error
    except Exception as error:
        status = provider_http_error_status(error)
        log_payload(logger, logging.WARNING, IDM_LOG_FAMILY, {
            "event": "provider_request_failed", "model": model, **safe_provider_error_diagnostics(error),
        })
        report_handled_exception(
            logger, error, stage=current_provider_stage() or "idm", family=IDM_LOG_FAMILY, model=model,
            outcome="AI_PROVIDER_REQUEST_REJECTED" if status == 400  # noqa: PLR2004 - HTTP status
            else "AI_PROVIDER_AUTH_REJECTED" if status in {401, 403} else "AI_PROVIDER_UNAVAILABLE")
        if status == 400:  # noqa: PLR2004 - HTTP status
            raise IdmProviderError("AI_PROVIDER_REQUEST_REJECTED", terminal=True) from error
        if status in {401, 403}:
            raise IdmProviderError("AI_PROVIDER_AUTH_REJECTED", terminal=True) from error
        raise IdmProviderError("AI_PROVIDER_UNAVAILABLE", terminal=False) from error


class IdmCallerRequest(Protocol):
    """What the runtime reads from a request: every RAG envelope (``RagRequestBase``) and the organization context
    distillation request have it."""

    @property
    def api_key(self) -> SecretStr: ...

    @property
    def model(self) -> str: ...

    @property
    def locale(self) -> Literal["vi", "en"]: ...

    @property
    def correlation_id(self) -> str | None: ...


def _idm_runtime(request: IdmCallerRequest, *, budget_ms: int, allowance: Any | None) -> IdmRuntime:
    # Ties the content of every IDM request (course design, module, unit) to the prompt templates
    # that wrote it, also across a deploy in the middle of a run.
    log_event(logger, logging.INFO, "idm_prompt_templates", family=IDM_LOG_FAMILY,
              correlation_id=request.correlation_id, prompt_policy_version=IDM_PROMPT_POLICY_VERSION,
              prompt_templates_hash=prompt_templates_hash())
    return IdmRuntime(
        generate=_idm_generate, api_key=require_provider_api_key(request.api_key), model=request.model,
        locale=request.locale, deadline=monotonic() + budget_ms / 1000, correlation_id=request.correlation_id,
        token_allowance=(IdmTokenAllowance(allowance.input_tokens, allowance.output_tokens)
                         if allowance is not None else None),
        provider_call_timeout_ms=settings.idm_provider_call_timeout_ms,
        rate_limit_max_wait_ms=settings.provider_rate_limit_max_wait_ms,
        run_cpu=service_runtime.concurrency.run_cpu,
    )


def _idm_http_error(error: IdmError) -> ServiceError:
    if isinstance(error, IdmProviderError):
        return _orchestration_v2_http_error(error.code, "AI provider failed for the IDM stage.",
                                            status_code=error.http_status)
    return _orchestration_v2_http_error(error.code, "The IDM stage cannot be completed for this source.")
