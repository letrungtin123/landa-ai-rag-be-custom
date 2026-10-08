"""Orchestration-v2 unit: IDM storyboard unit or the legacy staged unit."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from time import perf_counter
from typing import Any, Literal

from google.genai import types
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.logging import SERVICE_LOGGER_NAME
from app.idm.diagram import idm_source_step_diagram
from app.idm.runtime import IdmError, IdmStageError
from app.idm.source_locked import idm_source_faq, idm_source_grounded_single_choice, render_idm_source_locked_html
from app.idm.storyboard import IdmUnitDeps, run_idm_unit
from app.instructional_quality import source_relationship_pairs
from app.learner_content_purity import build_learner_content_purity_context
from app.lesson_author_orchestration_v2_provider import (
    UnitGenerationContractV2,
    unit_contract_manifest_v2,
    unit_contract_v5_architecture_v2,
)
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorRequest
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.semantic_review import (
    SemanticReviewResponse,
    SemanticReviewRunOutcome,
    build_semantic_repair_prompt,
    build_semantic_review_prompt,
    run_bounded_semantic_review,
    safe_semantic_review_summary,
    semantic_review_config_hash,
    validate_semantic_review_response,
)
from app.services import provider
from app.services.deadlines import record_fallback
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.staged import writer as staged_writer
from app.services.lesson_author.staged.provider_schemas import (
    bind_staged_instance_payload,
    build_staged_instance_response_model,
    build_staged_multi_repair_model,
    decode_staged_multi_repair,
)
from app.services.lesson_author.staged.validation import (
    StagedUnitFinding,
    merge_staged_component_payload_delta,
    validate_staged_unit_content,
)
from app.services.orchestration_v2.common import _orchestration_v2_http_error
from app.services.orchestration_v2.idm import _idm_http_error, _idm_runtime
from app.services.orchestration_v2.source_locked import (
    build_orchestration_v2_source_locked_components,
    build_orchestration_v2_source_locked_unit,
)
from app.services.provider import PROVIDER_TRANSIENT_MAX_ATTEMPTS, combine_usage
from app.services.retrieval.source_coverage import format_source_coverage_manifest
from app.workflows.contracts import WorkflowFailure

logger = logging.getLogger(SERVICE_LOGGER_NAME)


# Keep the outer Node -> Python request alive long enough to serialize and
# return a deterministic source-locked fallback after an inner provider
# deadline. Without this gap, a provider timeout and the HTTP client timeout
# race at the same millisecond and turn a valid fallback into outcome_unknown.
ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS = 5_000


async def lesson_author_orchestration_v2_unit(
    request: RagLessonAuthorUnitV2Request,
) -> dict[str, Any]:
    if request.unit_contract.idm_unit_brief is not None:
        budget_ms = max(1_000, request.remaining_workflow_budget_ms - ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS)
        try:
            result = await run_idm_unit(
                contract=request.unit_contract, runtime=_idm_runtime(request, budget_ms=budget_ms, allowance=None),
                deps=_idm_unit_deps(request), fallback_only=request.fallback_only,
            )
        except IdmError as error:
            raise _idm_http_error(error) from error
        except ValidationError as error:
            raise _idm_http_error(IdmStageError("IDM_STAGE_OUTPUT_INVALID")) from error
    else:
        result = await _lesson_author_orchestration_v2_unit(request=request)
    record_fallback("unit", result)
    return result


def _idm_unit_deps(request: RagLessonAuthorUnitV2Request) -> IdmUnitDeps:
    contract = request.unit_contract
    manifest = unit_contract_manifest_v2(contract, locale=request.locale)
    bundle = manifest.get("source_evidence_bundle")
    supporting = list(dict.fromkeys(item for plan in contract.component_plan
                                    for item in plan.supporting_evidence_fact_ids))
    source_locked_components = build_orchestration_v2_source_locked_components(
        contract, request.locale, html_renderer=render_idm_source_locked_html,
        single_choice_builder=idm_source_grounded_single_choice, diagram_builder=idm_source_step_diagram,
        faq_builder=idm_source_faq)
    return IdmUnitDeps(
        build_instance_model=build_staged_instance_response_model,
        bind_instance_payload=bind_staged_instance_payload,
        validate_unit=lambda unit, expected: validate_staged_unit_content(unit, expected, strict_payload=True),
        build_repair_model=build_staged_multi_repair_model,
        decode_repair=decode_staged_multi_repair,
        merge_repair=lambda unit, delta, targets, coverage: merge_staged_component_payload_delta(
            unit, delta, targets, coverage_targets=coverage),
        source_locked_unit=build_orchestration_v2_source_locked_unit(
            contract, request.locale, components=source_locked_components),
        source_locked_components=source_locked_components,
        purity_context=build_learner_content_purity_context(
            {"source_fact_ids": list(contract.unit_source_fact_ids), "supporting_evidence_fact_ids": supporting},
            manifest, _orchestration_v2_unit_source_rows(request)),
        evidence_review_required=isinstance(bundle, dict) and bundle.get("status") == "review_required",
        judge_mode=settings.idm_judge_mode,
        recoverable_errors=(LessonAuthorProposalValidationError, WorkflowFailure, ValueError, TypeError, KeyError),
    )


async def _lesson_author_orchestration_v2_unit(
    request: RagLessonAuthorUnitV2Request,
) -> dict[str, Any]:
    """Generate exactly one immutable inventory unit with the proven Stage-2 writer."""

    unit_started = perf_counter()
    contract = request.unit_contract
    manifest = unit_contract_manifest_v2(contract, locale=request.locale)
    evidence_bundle = manifest.get("source_evidence_bundle")
    evidence_review_required = (
        isinstance(evidence_bundle, dict)
        and evidence_bundle.get("status") == "review_required"
    )
    evidence_quality_state = "review_required" if evidence_review_required else "validated"
    source_rows = _orchestration_v2_unit_source_rows(request)
    context = "\n".join(f"[{fact.fact_key}] {fact.fact_text}" for fact in contract.source_facts)
    source_outline = f"{contract.chapter_title} > {contract.lesson_title} > {contract.unit_title}"
    source_coverage = format_source_coverage_manifest(manifest)
    adapted_architecture = unit_contract_v5_architecture_v2(contract)
    return await _lesson_author_orchestration_v2_unit_legacy(
        request, contract, manifest, evidence_quality_state, source_rows, context, source_outline,
        source_coverage, adapted_architecture, unit_started,
    )


def _orchestration_v2_unit_source_rows(request: RagLessonAuthorUnitV2Request) -> list[dict[str, Any]]:
    contract = request.unit_contract
    document_names = {document.document_id: document.name for document in request.source_documents}
    # Adapt the immutable V2 source ledger to the complete legacy formatter
    # row contract. Omitting any of these display/retrieval fields used to turn
    # a valid source fact into an ASGI 500 before the provider was called.
    return [{
        "document_id": fact.document_id,
        "document_name": document_names[fact.document_id],
        "source_page": fact.source_page,
        "source_section": fact.locator.get("source_section") or fact.source_ref,
        "chunk_no": fact.source_chunk,
        "content": fact.fact_text,
        "score": 1.0,
        "vector_score": 1.0,
        "keyword_score": 0.0,
        "method": "orchestration_v2_source_ledger",
        "methods": ["orchestration_v2_source_ledger"],
        "metadata": {
            "source_ref": fact.source_ref,
            "fact_key": fact.fact_key,
            "heading_path": fact.locator.get("heading_path"),
        },
    } for fact in contract.source_facts]


async def _lesson_author_orchestration_v2_unit_legacy(
    request: RagLessonAuthorUnitV2Request,
    contract: UnitGenerationContractV2,
    manifest: dict[str, Any],
    evidence_quality_state: str,
    source_rows: list[dict[str, Any]],
    context: str,
    source_outline: str,
    source_coverage: str,
    adapted_architecture: dict[str, Any],
    unit_started: float,
) -> dict[str, Any]:
    adapted = RagLessonAuthorRequest.model_validate({
        **request.model_dump(exclude={"contract_version", "unit_contract", "remaining_workflow_budget_ms"}),
        "outline_context": contract.unit_path,
        "target_scope_instruction": "Generate only the approved immutable orchestration V2 unit.",
        "output_schema_hint": "Return the server-supplied staged unit schema.",
        "operation": "create", "target_type": "chapter", "generation_mode": "staged",
        "blueprint_architecture": adapted_architecture,
    })
    source_locked_fallback_unit = build_orchestration_v2_source_locked_unit(contract, request.locale)
    attempt_trace: list[dict[str, Any]] = []
    semantic_review_mode = settings.semantic_review_mode
    semantic_review_model = settings.semantic_review_model or request.model
    semantic_config_hash = semantic_review_config_hash(
        mode=semantic_review_mode,
        model=semantic_review_model,
        timeout_ms=settings.semantic_review_timeout_ms,
        max_output_tokens=settings.semantic_review_max_output_tokens,
        provider_attempt_cap=settings.semantic_review_provider_attempt_cap,
    )
    unit_supporting_evidence_fact_ids = list(dict.fromkeys(
        fact_id
        for plan in contract.component_plan
        for fact_id in plan.supporting_evidence_fact_ids
    ))
    fact_text_by_id = {fact.fact_key: fact.fact_text for fact in contract.source_facts}
    diagram_relationships_by_plan_id = {
        plan.component_plan_id: [
            [left, right, *([relation] if relation else [])]
            for left, right, relation in source_relationship_pairs(
                fact_text_by_id[fact_id]
                for fact_id in dict.fromkeys([
                    *plan.source_fact_ids,
                    *plan.supporting_evidence_fact_ids,
                ])
                if fact_id in fact_text_by_id
            )
        ]
        for plan in contract.component_plan
        if plan.type == "la_diagram"
    }
    expected = {
        "unit_title": contract.unit_title,
        "unit_purpose": contract.unit_purpose,
        "source_fact_ids": list(contract.unit_source_fact_ids),
        "supporting_evidence_fact_ids": unit_supporting_evidence_fact_ids,
        "component_types": [plan.type for plan in contract.component_plan],
        "component_plan": [plan.model_dump(mode="json") for plan in contract.component_plan],
        "diagram_relationships_by_plan_id": diagram_relationships_by_plan_id,
        "learning_objectives": list(contract.lesson_learning_objectives),
        "learning_objective_refs": list(contract.unit_learning_objective_refs),
        "locale": request.locale,
        "learner_content_purity": build_learner_content_purity_context(
            {
                "source_fact_ids": list(contract.unit_source_fact_ids),
                "supporting_evidence_fact_ids": unit_supporting_evidence_fact_ids,
            },
            manifest,
            source_rows,
        ),
        "instructional_output_budget": {
            "policy_version": adapted_architecture["lessons"][0]["units"][0]["instructional_density_policy_version"],
            "source_content_chars": adapted_architecture["lessons"][0]["units"][0]["source_content_chars"],
            "source_estimated_words": adapted_architecture["lessons"][0]["units"][0]["source_estimated_words"],
            "max_visible_chars": adapted_architecture["lessons"][0]["units"][0]["max_generated_visible_chars"],
            "max_words": adapted_architecture["lessons"][0]["units"][0]["max_generated_words"],
        } if adapted_architecture["lessons"][0]["units"][0]["instructional_density_policy_version"] else None,
    }

    def semantic_review_unavailable(unit: dict[str, Any], code: str) -> dict[str, Any]:
        return safe_semantic_review_summary(
            SemanticReviewRunOutcome(
                unit=deepcopy(unit),
                quality_state="review_required",
                review_status="unavailable",
                first_review=None,
                scoped_review=None,
                repair_attempted=False,
                repair_applied=False,
                repair_component_indices=(),
                failure_code=code,
            ),
            config_hash=semantic_config_hash,
        )

    def deterministic_fallback(reason_code: str, *, provider_dispatched: bool) -> dict[str, Any]:
        unit = deepcopy(source_locked_fallback_unit)
        finding = (validate_staged_unit_content(unit, expected, strict_payload=True)
                   if isinstance(unit, dict) else
                   StagedUnitFinding("Deterministic unit fallback is unavailable.",
                                     "UNIT_FALLBACK_UNAVAILABLE", "unit", False))
        if finding is not None:
            logger.error("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "unit_deterministic_fallback_rejected",
                "correlation_id": request.correlation_id,
                "chapter_key": contract.chapter_key,
                "unit_path": contract.unit_path,
                "reason_code": reason_code,
                "validation_finding": finding.diagnostic() if isinstance(finding, StagedUnitFinding) else {
                    "code": "UNIT_FALLBACK_INVALID", "path": "unit", "repairable": False,
                },
            }, sort_keys=True))
            raise _orchestration_v2_http_error(
                "ORCHESTRATION_V2_UNIT_FALLBACK_INVALID",
                "The source-locked unit fallback did not pass server validation.",
                status_code=422,
            )
        # This is a whole-unit deterministic draft, not a provider-validated
        # candidate. It must remain reviewable even when every source fact is
        # text-only and otherwise validated. The V2 persistence fence relies
        # on this exact provenance/quality pair for provider-free replay.
        fallback_quality_state = "review_required"
        semantic_summary = (
            semantic_review_unavailable(unit, "SEMANTIC_REVIEW_NOT_RUN")
            if semantic_review_mode != "off" else None
        )
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "unit_deterministic_fallback_ready",
            "correlation_id": request.correlation_id,
            "source_snapshot_hash": contract.source_snapshot_hash,
            "chapter_key": contract.chapter_key,
            "unit_path": contract.unit_path,
            "source_fact_count": len(contract.source_facts),
            "component_count": len(contract.component_plan),
            "reason_code": reason_code,
            "provider_dispatched": provider_dispatched,
            "content_origin": "structured_fallback",
            "quality_state": fallback_quality_state,
            **({"semantic_review": semantic_summary} if semantic_summary is not None else {}),
        }, sort_keys=True))
        usage = AiUsage().model_dump()
        # The Node dispatch fence is persisted before this HTTP request. A
        # first-attempt fallback therefore settles the reserved upper bound
        # even when Python rejected the request before reaching Gemini. The
        # durable fallback-only replay is provider-free and fully complete.
        first_attempt = not request.fallback_only
        if len(attempt_trace) < 64:
            attempt_trace.append({
                "sequence": len(attempt_trace) + 1,
                "invocation_kind": "deterministic",
                "invocation_index": 1,
                "provider_attempt": None,
                "phase": "fallback",
                "outcome": "fallback",
                "event_code": "deterministic_fallback",
                "failure_stage": ("provider_usage" if reason_code == "PROVIDER_USAGE_INCOMPLETE"
                                  else "durable_replay" if reason_code == "DURABLE_FINAL_ATTEMPT"
                                  else "unit_generation"),
                "failure_code": reason_code[:100] if re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", reason_code) else "FALLBACK_REQUIRED",
                "failure_path": None,
                "provider_dispatched": provider_dispatched,
                "usage_source": "unknown",
                "observed_usage": {},
                "duration_ms": 0,
                "diagnostics": {},
            })
        return {"contract_version": 2, "source_snapshot_hash": contract.source_snapshot_hash,
                "unit_path": contract.unit_path, "unit": unit, "usage": usage,
                "usage_complete": not first_attempt,
                "usage_source": "reserved_upper_bound" if first_attempt else "deterministic_fallback",
                "content_origin": "structured_fallback", "quality_state": fallback_quality_state,
                "attempt_trace": attempt_trace,
                **({"semantic_review": semantic_summary} if semantic_summary is not None else {})}

    if request.fallback_only:
        return deterministic_fallback("DURABLE_FINAL_ATTEMPT", provider_dispatched=False)
    provider_dispatched = False

    def mark_provider_dispatched() -> None:
        nonlocal provider_dispatched
        provider_dispatched = True

    semantic_usage = AiUsage()
    semantic_usage_complete = True
    semantic_provider_dispatched = False

    def semantic_provider_attempts_used() -> int:
        return sum(
            event.get("event_code") == "provider_http_attempt_started"
            for event in attempt_trace
            if isinstance(event, dict)
        )

    async def call_semantic_provider(
        *,
        prompt: str,
        response_schema: type[BaseModel],
        invocation_kind: Literal["evaluator", "repair"],
        invocation_index: int,
        max_output_tokens: int,
    ) -> str:
        """Dispatch one frozen semantic invocation inside the shared request budget."""

        nonlocal semantic_usage, semantic_usage_complete, semantic_provider_dispatched
        # `generate_content` may retry one transient 5xx response. Reserve both
        # transport attempts before dispatch so nested retries cannot exceed the
        # writer/evaluator/repair request ceiling.
        if (
            semantic_provider_attempts_used() + PROVIDER_TRANSIENT_MAX_ATTEMPTS
            > settings.semantic_review_provider_attempt_cap
        ):
            raise RuntimeError("SEMANTIC_PROVIDER_ATTEMPT_BUDGET_EXHAUSTED")
        elapsed_ms = max(0, int((perf_counter() - unit_started) * 1000))
        remaining_ms = request.remaining_workflow_budget_ms - elapsed_ms
        timeout_ms = min(settings.semantic_review_timeout_ms, remaining_ms)
        if timeout_ms < 1_000:
            raise RuntimeError("SEMANTIC_WORKFLOW_BUDGET_EXHAUSTED")
        invocation_dispatched = False
        invocation_usage_complete = False

        def capture(metadata: dict[str, Any]) -> None:
            nonlocal invocation_dispatched, invocation_usage_complete, semantic_provider_dispatched
            event = str(metadata.get("event") or "")
            provider_attempt = metadata.get("provider_attempt")
            if event == "provider_http_attempt_started":
                invocation_dispatched = True
                semantic_provider_dispatched = True
            counts = [
                metadata.get(key)
                for key in ("provider_input_tokens", "provider_output_tokens", "provider_total_tokens")
            ]
            if (
                metadata.get("provider_http_status") == 200
                and metadata.get("usage_source") == "provider"
                and all(type(value) is int and value >= 0 for value in counts)
                and counts[2] >= counts[0] + counts[1]
            ):
                invocation_usage_complete = True
            outcome = {
                "provider_http_attempt_started": "started",
                "provider_http_attempt_succeeded": "succeeded",
                "provider_response_received": "succeeded",
                "provider_transient_retry": "retrying",
                "provider_timeout": "failed",
                "provider_quota_exhausted": "failed",
                "provider_unavailable": "failed",
                "provider_request_failed": "failed",
            }.get(event)
            if (
                outcome is None
                or type(provider_attempt) is not int
                or not 1 <= provider_attempt <= 8
                or len(attempt_trace) >= 64
            ):
                return
            raw_code = str(metadata.get("internal_failure_code") or event).upper()
            failure_code = re.sub(r"[^A-Z0-9_]", "_", raw_code)[:100] or "SEMANTIC_PROVIDER_FAILED"
            observed_usage = {
                key: metadata[key]
                for key in ("provider_input_tokens", "provider_output_tokens", "provider_total_tokens")
                if type(metadata.get(key)) is int and metadata[key] >= 0
            }
            diagnostics = {
                key: metadata[key]
                for key in ("provider_http_status", "provider_status", "provider_error_category",
                            "provider_schema_constraint", "provider_finish_reason")
                if metadata.get(key) is not None
            }
            attempt_trace.append({
                "sequence": len(attempt_trace) + 1,
                "invocation_kind": invocation_kind,
                "invocation_index": invocation_index,
                "provider_attempt": provider_attempt,
                "phase": "semantic_review" if invocation_kind == "evaluator" else "semantic_repair",
                "outcome": outcome,
                "event_code": event[:96],
                "failure_stage": "provider_transport" if outcome == "failed" else None,
                "failure_code": failure_code if outcome == "failed" else None,
                "failure_path": None,
                "provider_dispatched": True,
                "usage_source": (
                    "provider_reported" if metadata.get("usage_source") == "provider"
                    else "estimated" if metadata.get("usage_source") == "local_estimate"
                    else "unknown"
                ),
                "observed_usage": observed_usage,
                "duration_ms": max(0, int(metadata.get("duration_ms") or 0)),
                "diagnostics": diagnostics,
            })

        try:
            text, invocation_usage = await provider.generate_content(
                request.api_key,
                semantic_review_model,
                prompt,
                max_output_tokens=max_output_tokens,
                json_mode=True,
                response_schema=response_schema,
                thinking_config=types.ThinkingConfig(include_thoughts=False),
                request_timeout_ms=timeout_ms,
                on_provider_telemetry=capture,
            )
            semantic_usage = combine_usage(semantic_usage, invocation_usage)
            return text
        finally:
            if invocation_dispatched and not invocation_usage_complete:
                semantic_usage_complete = False

    # The durable caller has its own request deadline. End generation before
    # that boundary so timeout handling can still build and return the
    # deterministic fallback over the same HTTP request.
    generation_budget_ms = max(
        1,
        request.remaining_workflow_budget_ms
        - ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS,
    )
    try:
        value, usage = await asyncio.wait_for(staged_writer.generate_staged_lesson_author_proposal(
            adapted, context, source_outline, source_coverage, source_rows=source_rows,
            source_coverage_manifest=manifest, checkpoint_unit_index=0,
            remaining_workflow_budget_ms=generation_budget_ms,
            checkpoint_component_fallback_unit=source_locked_fallback_unit,
            on_provider_dispatch=mark_provider_dispatched,
            attempt_trace_sink=attempt_trace,
        ), generation_budget_ms / 1000)
        unit = value.get("unit") if isinstance(value, dict) else None
        if not isinstance(unit, dict) or unit.get("title") != contract.unit_title:
            raise LessonAuthorProposalValidationError("ORCHESTRATION_V2_UNIT_IDENTITY_CHANGED")
        provider_usage_complete = value.pop("provider_usage_complete", False) is True
        if not provider_usage_complete:
            return deterministic_fallback("PROVIDER_USAGE_INCOMPLETE", provider_dispatched=provider_dispatched)
        fallback_component_indices = value.pop("fallback_component_indices", [])
        partial_fallback = bool(fallback_component_indices)
        content_origin = "structured_fallback" if partial_fallback else "provider_validated"
        semantic_summary: dict[str, Any] | None = None
        semantic_outcome: SemanticReviewRunOutcome | None = None
        if semantic_review_mode != "off":
            reviewer_invocation_index = 0
            allowed_review_fact_ids = {
                fact.fact_key for fact in contract.source_facts
            }

            async def semantic_reviewer(
                candidate: dict[str, Any],
                component_indices: tuple[int, ...],
            ) -> SemanticReviewResponse:
                nonlocal reviewer_invocation_index
                reviewer_invocation_index += 1
                prompt = build_semantic_review_prompt(
                    unit=candidate,
                    expected=expected,
                    manifest=manifest,
                    component_indices=component_indices,
                    locale=request.locale,
                )
                text = await call_semantic_provider(
                    prompt=prompt,
                    response_schema=SemanticReviewResponse,
                    invocation_kind="evaluator",
                    invocation_index=reviewer_invocation_index,
                    max_output_tokens=settings.semantic_review_max_output_tokens,
                )
                parsed = SemanticReviewResponse.model_validate_json(text)
                return validate_semantic_review_response(
                    parsed,
                    component_count=len(candidate.get("components", [])),
                    allowed_source_fact_ids=allowed_review_fact_ids,
                    requested_component_indices=component_indices,
                )

            async def semantic_repairer(
                candidate: dict[str, Any],
                component_indices: tuple[int, ...],
                findings: tuple[Any, ...],
            ) -> dict[str, Any]:
                prompt = build_semantic_repair_prompt(
                    unit=candidate,
                    expected=expected,
                    manifest=manifest,
                    component_indices=component_indices,
                    findings=findings,
                    locale=request.locale,
                )
                response_model = build_staged_multi_repair_model(
                    candidate, list(component_indices), [],
                )
                text = await call_semantic_provider(
                    prompt=prompt,
                    response_schema=response_model,
                    invocation_kind="repair",
                    invocation_index=1,
                    max_output_tokens=min(
                        max(settings.semantic_review_max_output_tokens, 8_192),
                        request.max_output_tokens,
                        16_384,
                    ),
                )
                diagnostics: dict[str, Any] = {}
                delta = decode_staged_multi_repair(
                    text, candidate, list(component_indices), [], diagnostics,
                )
                return merge_staged_component_payload_delta(
                    candidate, delta, list(component_indices), coverage_targets=[],
                )

            def deterministic_semantic_recheck(candidate: dict[str, Any]) -> str | None:
                finding = validate_staged_unit_content(candidate, expected, strict_payload=True)
                return str(finding) if finding is not None else None

            semantic_outcome = await run_bounded_semantic_review(
                unit,
                reviewer=semantic_reviewer,
                repairer=semantic_repairer,
                deterministic_validate=deterministic_semantic_recheck,
                allow_repair=semantic_review_mode == "repair",
            )
            unit = semantic_outcome.unit
            semantic_summary = safe_semantic_review_summary(
                semantic_outcome,
                config_hash=semantic_config_hash,
            )
        # Component fallback passes the same instructional validator as the
        # provider-authored components, so provenance does not reduce its
        # publication readiness.
        quality_state = evidence_quality_state
        if (
            semantic_review_mode == "repair"
            and semantic_outcome is not None
            and semantic_outcome.quality_state == "review_required"
        ):
            quality_state = "review_required"
        combined_usage = combine_usage(usage, semantic_usage)
        combined_usage_complete = provider_usage_complete and semantic_usage_complete
        logger.info("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "unit_ready", "correlation_id": request.correlation_id,
            "source_snapshot_hash": contract.source_snapshot_hash, "chapter_key": contract.chapter_key,
            "unit_path": contract.unit_path, "source_fact_count": len(contract.source_facts),
            "component_count": len(unit.get("components", [])),
            "provider_usage_complete": provider_usage_complete,
            "fallback_component_indices": fallback_component_indices,
            "content_origin": content_origin,
            "quality_state": quality_state,
            "semantic_review_mode": semantic_review_mode,
            "semantic_provider_dispatched": semantic_provider_dispatched,
            **({"semantic_review": semantic_summary} if semantic_summary is not None else {}),
        }, sort_keys=True))
        return {"contract_version": 2, "source_snapshot_hash": contract.source_snapshot_hash,
                "unit_path": contract.unit_path, "unit": unit, "usage": combined_usage.model_dump(),
                "usage_complete": combined_usage_complete,
                "usage_source": "provider" if combined_usage_complete else "reserved_upper_bound",
                "content_origin": content_origin, "quality_state": quality_state,
                "attempt_trace": attempt_trace,
                **({"semantic_review": semantic_summary} if semantic_summary is not None else {})}
    except asyncio.TimeoutError:
        return deterministic_fallback("AI_STAGED_LESSON_WORKFLOW_TIMEOUT", provider_dispatched=provider_dispatched)
    except WorkflowFailure as error:
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "unit_generation_failed",
            "correlation_id": request.correlation_id,
            "chapter_key": contract.chapter_key,
            "unit_path": contract.unit_path,
            "failure_stage": error.failure_stage,
            "failure_code": error.internal_code or error.code,
            "provider_dispatched": provider_dispatched,
        }, sort_keys=True))
        return deterministic_fallback(error.internal_code or error.code, provider_dispatched=provider_dispatched)
    except (LessonAuthorProposalValidationError, ValidationError, ValueError) as error:
        validation_errors = ([{"loc": list(item.get("loc", ())), "type": item.get("type")}
                              for item in error.errors()[:16]] if isinstance(error, ValidationError) else [])
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "unit_generation_invalid",
            "correlation_id": request.correlation_id,
            "chapter_key": contract.chapter_key,
            "unit_path": contract.unit_path,
            "exception_type": type(error).__name__,
            "validation_errors": validation_errors,
            "provider_dispatched": provider_dispatched,
        }, sort_keys=True))
        return deterministic_fallback("ORCHESTRATION_V2_UNIT_INVALID", provider_dispatched=provider_dispatched)
