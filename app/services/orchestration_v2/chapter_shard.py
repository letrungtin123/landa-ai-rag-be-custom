"""Orchestration-v2 chapter shard: IDM module design or the legacy shard."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from app.contracts.orchestration_v2.blueprint import OrchestrationContractError
from app.contracts.orchestration_v2.provider import (
    CHAPTER_SHARD_PROVIDER_SCHEMA_METADATA,
    ChapterShardProviderWireV2,
    bind_chapter_shard_v2,
    chapter_shard_prompt_v2,
    fallback_chapter_shard_draft_v2,
    parse_chapter_shard_draft_v2,
    salvage_chapter_shard_draft_v2,
)
from app.core.errors import ServiceError
from app.core.handled_exceptions import report_handled_exception
from app.core.logging import SERVICE_LOGGER_NAME, log_payload
from app.idm.module_design import run_idm_module_design
from app.idm.preserve_design import is_preserve_context, run_preserve_module_design
from app.idm.runtime import IdmError, IdmStageError
from app.infra.gemini.errors import http_error_detail
from app.infra.gemini.usage import combine_usage
from app.schemas.common import AiUsage
from app.schemas.orchestration_v2 import RagLessonAuthorChapterShardV2Request
from app.services import runtime as service_runtime
from app.services.deadlines import record_fallback
from app.services.orchestration_v2.common import (
    ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES,
    V2_LOG_FAMILY,
    _orchestration_v2_generate_content,
    _orchestration_v2_http_error,
)
from app.services.orchestration_v2.idm import _idm_http_error, _idm_runtime

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def lesson_author_orchestration_v2_chapter_shard(
    request: RagLessonAuthorChapterShardV2Request,
) -> dict[str, Any]:
    context = request.idm_module_context
    if context is not None and is_preserve_context(context.project_context):
        # QLT-6: a kept document's lessons are laid out without a provider call (its intended output, no fallback).
        try:
            return await service_runtime.concurrency.run_cpu(
                run_preserve_module_design, context=context, skeleton=request.skeleton, plan=request.shard_plan,
                facts=request.source_facts)
        except IdmError as error:
            raise _idm_http_error(error) from error
        except ValidationError as error:
            raise _idm_http_error(IdmStageError("IDM_STAGE_OUTPUT_INVALID")) from error
    if context is not None:
        try:
            result = await run_idm_module_design(
                context=context, skeleton=request.skeleton, plan=request.shard_plan, facts=request.source_facts,
                runtime=_idm_runtime(request, budget_ms=context.remaining_budget_ms,
                                     allowance=context.token_allowance),
            )
        except IdmError as error:
            raise _idm_http_error(error) from error
        except ValidationError as error:
            raise _idm_http_error(IdmStageError("IDM_STAGE_OUTPUT_INVALID")) from error
    else:
        result = await _lesson_author_orchestration_v2_chapter_shard(request=request)
    record_fallback("chapter_shard", result)
    return result


async def _lesson_author_orchestration_v2_chapter_shard(
    request: RagLessonAuthorChapterShardV2Request,
) -> dict[str, Any]:
    """Generate one independently retryable, source-bounded chapter shard."""

    try:
        prompt = chapter_shard_prompt_v2(
            request.locale, request.skeleton, request.shard_plan, request.source_facts,
        )
    except (OrchestrationContractError, StopIteration) as error:
        context_code = (error.code if isinstance(error, OrchestrationContractError)
                        else "ARCHITECTURE_SHARD_IDENTITY_MISMATCH")
        raise _orchestration_v2_http_error(context_code, "The chapter shard context is invalid.") from error
    log_payload(logger, logging.INFO, V2_LOG_FAMILY, {
        "event": "provider_schema_projection_ready", "correlation_id": request.correlation_id,
        "generation_stage": "chapter_shard", "chapter_key": request.shard_plan.chapter_key,
        "shard_index": request.shard_plan.shard_index, **CHAPTER_SHARD_PROVIDER_SCHEMA_METADATA,
    })
    usage = AiUsage()
    last_code = "ARCHITECTURE_SHARD_INVALID"
    provider_failure_code: str | None = None
    repair_hint = ""
    last_provider_text: str | None = None
    best_salvage: tuple[Any, dict[str, Any]] | None = None
    for attempt in range(1, request.max_attempts + 1):
        try:
            text, attempt_usage = await _orchestration_v2_generate_content(
                request.api_key, request.model, prompt + repair_hint,
                max_output_tokens=request.max_output_tokens,
                response_schema=ChapterShardProviderWireV2,
                correlation_id=request.correlation_id,
                generation_stage="chapter_shard",
            )
        except ServiceError as error:
            detail = http_error_detail(error)
            code = detail.get("code")
            if code not in ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES:
                raise
            provider_failure_code = str(code)
            last_code = provider_failure_code
            break
        except Exception as error:
            provider_failure_code = "AI_PROVIDER_UNAVAILABLE"
            last_code = provider_failure_code
            report_handled_exception(logger, error, stage="chapter_shard", outcome=provider_failure_code,
                                     family=V2_LOG_FAMILY, correlation_id=request.correlation_id,
                                     chapter_key=request.shard_plan.chapter_key,
                                     shard_index=request.shard_plan.shard_index, provider_attempt=attempt)
            log_payload(logger, logging.WARNING, V2_LOG_FAMILY, {
                "event": "chapter_shard_provider_fallback",
                "correlation_id": request.correlation_id,
                "chapter_key": request.shard_plan.chapter_key,
                "shard_index": request.shard_plan.shard_index,
                "failure_code": provider_failure_code,
                "exception_type": type(error).__name__,
            })
            break
        usage = combine_usage(usage, attempt_usage)
        last_provider_text = text
        try:
            draft = parse_chapter_shard_draft_v2(text)
            compiler_diagnostics: list[dict[str, Any]] = []
            shard = bind_chapter_shard_v2(
                draft,
                skeleton=request.skeleton,
                plan=request.shard_plan,
                source_facts=request.source_facts,
                compiler_diagnostics=compiler_diagnostics,
            )
            log_payload(logger, logging.INFO, V2_LOG_FAMILY, {
                "event": "chapter_shard_ready", "correlation_id": request.correlation_id,
                "source_snapshot_hash": request.skeleton.source_snapshot_hash,
                "chapter_key": request.shard_plan.chapter_key,
                "shard_index": request.shard_plan.shard_index,
                "shard_count": request.shard_plan.shard_count,
                "source_fact_count": len(request.source_facts), "provider_attempt": attempt,
                "component_compiler": compiler_diagnostics[:24],
                "omitted_component_compiler_unit_count": max(0, len(compiler_diagnostics) - 24),
            })
            return {"contract_version": 2, "shard": shard.model_dump(mode="json"),
                    "usage": usage.model_dump(), "usage_complete": True,
                    "usage_source": "provider", "content_origin": "provider_validated",
                    "quality_state": "validated"}
        except OrchestrationContractError as error:
            last_code = error.code
            safe_errors: list[dict[str, Any]] = [{"type": "contract_error", "loc": [], "code": error.code}]
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            last_code = "ARCHITECTURE_SHARD_INVALID"
            safe_errors = ([{"type": item.get("type"), "loc": list(item.get("loc") or ())}
                            for item in error.errors(include_url=False, include_input=False)[:12]]
                           if isinstance(error, ValidationError)
                           else [{"type": type(error).__name__, "loc": []}])
        log_payload(logger, logging.WARNING, V2_LOG_FAMILY, {
            "event": "chapter_shard_validation_failed", "correlation_id": request.correlation_id,
            "chapter_key": request.shard_plan.chapter_key, "shard_index": request.shard_plan.shard_index,
            "provider_attempt": attempt, "failure_code": last_code, "validation_errors": safe_errors,
        })
        try:
            candidate_salvage = salvage_chapter_shard_draft_v2(
                text,
                skeleton=request.skeleton,
                plan=request.shard_plan,
                source_facts=request.source_facts,
            )
            if (candidate_salvage is not None
                    and (best_salvage is None
                         or candidate_salvage[1]["accepted_provider_unit_count"]
                         > best_salvage[1]["accepted_provider_unit_count"])):
                best_salvage = candidate_salvage
        except (OrchestrationContractError, ValidationError, ValueError, json.JSONDecodeError):
            pass
        repair_hint = (" REPAIR_REQUIREMENTS: Return the complete schema again. Correct these validation locations: "
                       + json.dumps(safe_errors, separators=(",", ":")) + ".")

    if best_salvage is None and last_provider_text is not None:
        try:
            best_salvage = salvage_chapter_shard_draft_v2(
                last_provider_text,
                skeleton=request.skeleton,
                plan=request.shard_plan,
                source_facts=request.source_facts,
            )
        except (OrchestrationContractError, ValidationError, ValueError, json.JSONDecodeError) as error:
            log_payload(logger, logging.WARNING, V2_LOG_FAMILY, {
                "event": "chapter_shard_partial_salvage_rejected",
                "correlation_id": request.correlation_id,
                "chapter_key": request.shard_plan.chapter_key,
                "shard_index": request.shard_plan.shard_index,
                "failure_code": error.code if isinstance(error, OrchestrationContractError) else type(error).__name__,
            })
            best_salvage = None
    if best_salvage is not None:
        shard, diagnostics = best_salvage
        component_decisions = diagnostics.pop("component_decisions", [])
        log_payload(logger, logging.WARNING, V2_LOG_FAMILY, {
            "event": "chapter_shard_partial_salvage_ready",
            "correlation_id": request.correlation_id,
            "chapter_key": request.shard_plan.chapter_key,
            "shard_index": request.shard_plan.shard_index,
            "source_fact_count": len(request.source_facts),
            "provider_attempts": request.max_attempts,
            "last_failure_code": last_code,
            **diagnostics,
            "component_compiler": component_decisions[:24],
            "omitted_component_compiler_unit_count": max(0, len(component_decisions) - 24),
        })
        return {"contract_version": 2, "shard": shard.model_dump(mode="json"),
                "usage": usage.model_dump(), "usage_complete": provider_failure_code is None,
                "usage_source": "reserved_upper_bound" if provider_failure_code else "provider",
                "content_origin": "structured_fallback", "quality_state": "review_required"}

    fallback = fallback_chapter_shard_draft_v2(
        request.skeleton, request.shard_plan, request.source_facts,
    )
    compiler_diagnostics = []
    shard = bind_chapter_shard_v2(
        fallback,
        skeleton=request.skeleton,
        plan=request.shard_plan,
        source_facts=request.source_facts,
        compiler_diagnostics=compiler_diagnostics,
    )
    log_payload(logger, logging.WARNING, V2_LOG_FAMILY, {
        "event": "chapter_shard_deterministic_fallback_ready",
        "correlation_id": request.correlation_id,
        "chapter_key": request.shard_plan.chapter_key,
        "shard_index": request.shard_plan.shard_index,
        "source_fact_count": len(request.source_facts),
        "provider_attempts": request.max_attempts,
        "last_failure_code": last_code,
        "component_compiler": compiler_diagnostics[:24],
        "omitted_component_compiler_unit_count": max(0, len(compiler_diagnostics) - 24),
    })
    return {"contract_version": 2, "shard": shard.model_dump(mode="json"),
            "usage": usage.model_dump(), "usage_complete": provider_failure_code is None,
            "usage_source": "reserved_upper_bound" if provider_failure_code else "provider",
            "content_origin": "structured_fallback", "quality_state": "review_required"}
