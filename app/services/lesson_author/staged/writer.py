"""Staged lesson writer: skeleton, unit batches, validation and bounded repairs."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from time import perf_counter
from typing import Any, Callable

from fastapi import HTTPException
from google.genai import types
from pydantic import BaseModel

from app.core.logging import SERVICE_LOGGER_NAME
from app.learner_content_purity import build_learner_content_purity_context
from app.lesson_author_checkpoint import ChapterCheckpointUnit, checkpoint_expected_units, select_checkpoint_unit
from app.lesson_author_provider_schema import staged_provider_response_model
from app.lesson_content_observation import observe_lesson_content
from app.lesson_prompt_policy import instructional_contract_review_signals, lesson_output_language_policy
from app.ordered_learning_content import ProviderSemanticVersionError, bind_provider_semantic_versions
from app.prompt_safety import untrusted_block
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorRequest
from app.services import provider
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.prompts import STAGED_UNIT_UNTRUSTED_RULE
from app.services.lesson_author.proposal_validation import (
    normalize_lesson_author_proposal_tree,
    validate_lesson_author_proposal_shape,
)
from app.services.lesson_author.staged import source_locked as staged_source_locked
from app.services.lesson_author.staged.plan import (
    STAGED_LESSON_AUTHOR_RECOVERY_UNITS,
    STAGED_LESSON_AUTHOR_SKELETON_TOKENS,
    STAGED_LESSON_AUTHOR_UNIT_CONTEXT_CHARS,
    StagedLessonWorkflowDeadline,
    consolidate_staged_thin_units,
    staged_lesson_content_output_tokens,
)
from app.services.lesson_author.staged.provider_schemas import (
    STAGED_COMPONENT_CONTRACT_VERSION,
    STAGED_COMPONENT_COVERAGE_REPAIR_CONTRACT_VERSION,
    STAGED_COMPONENT_REPAIR_CONTRACT_VERSION,
    STAGED_INSTANCE_OUTPUT_CONTRACT_VERSION,
    STAGED_MULTI_REPAIR_CONTRACT_VERSION,
    bind_staged_instance_payload,
    build_lesson_author_skeleton_response_schema,
    build_staged_instance_response_model,
    build_staged_lesson_content_response_model,
    build_staged_multi_repair_model,
    decode_staged_multi_repair,
    is_stage_two_provider_schema_error,
    staged_component_contract_prompt,
    staged_response_schema_diagnostics,
)
from app.services.lesson_author.staged.skeleton import (
    _apply_blueprint_architecture_to_skeleton,
    extract_lesson_author_unit_batches,
    match_staged_unit_by_title,
    parse_lesson_author_json_value,
    staged_unit_candidates,
    staged_unit_match_diagnostics,
    validate_staged_skeleton_source_facts,
)
from app.services.lesson_author.staged.source_locked import (
    build_source_locked_staged_skeleton,
    build_staged_instructional_contract,
    prepare_source_locked_expected,
    staged_unit_source_material,
)
from app.services.lesson_author.staged.validation import (
    StagedUnitFinding,
    lesson_author_proposal_quality_metrics,
    merge_checkpoint_component_fallback,
    merge_staged_component_payload_delta,
    staged_component_repair_guard,
    staged_component_repair_targets,
    staged_coverage_claim_diagnostics,
    staged_coverage_repair_diagnostics,
    staged_evidence_scope_diagnostics,
    staged_instructional_diagnostics,
    staged_instructional_finding,
    staged_payload_diagnostics,
    validate_staged_unit_content,
)
from app.services.provider import combine_usage, is_non_retryable_provider_error, safe_provider_error_diagnostics
from app.workflows.contracts import WorkflowFailure

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def generate_staged_lesson_author_proposal(
    request: RagLessonAuthorRequest,
    context: str,
    source_outline: str,
    source_coverage: str = "",
    source_rows: list[dict[str, Any]] | None = None,
    source_coverage_manifest: dict[str, Any] | None = None,
    *,
    checkpoint_unit_index: int | None = None,
    remaining_workflow_budget_ms: int | None = None,
    checkpoint_component_fallback_unit: dict[str, Any] | None = None,
    on_provider_dispatch: Callable[[], None] | None = None,
    attempt_trace_sink: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], AiUsage]:
    """Generate a large chapter as a validated skeleton plus bounded unit batches."""
    workflow_deadline = StagedLessonWorkflowDeadline(remaining_workflow_budget_ms)
    checkpoint_provider_usage_complete = True
    checkpoint_fallback_component_indices: list[int] = []
    checkpoint_attempt_events = attempt_trace_sink if attempt_trace_sink is not None else []
    # A V5 Blueprint is already an approved, server-validated topology with
    # canonical fact ownership. Calling the provider to recreate that tree
    # permits silent drift before Stage 2 begins, so derive the skeleton
    # directly and reserve the provider for bounded unit content only.
    use_direct_v5_skeleton = (
        request.blueprint_architecture is not None
        and request.blueprint_architecture.architecture_contract_version == 5
    )
    if checkpoint_unit_index is not None and not use_direct_v5_skeleton:
        raise ValueError("CHAPTER_CHECKPOINT_REQUIRES_V5")
    skeleton_prompt = "\n\n".join(
        [
            "SERVER STAGE 1: Build only the compact structure for exactly one chapter.",
            lesson_output_language_policy(request.locale),
            f"User request:\n{request.user_message}",
            f"Course context:\n{request.course_context}" if request.course_context else "",
            f"Target scope instruction:\n{request.target_scope_instruction}" if request.target_scope_instruction else "",
            (
                f"Approved Blueprint content architecture (mandatory):\n{request.blueprint_architecture.model_dump_json()}"
                if request.blueprint_architecture is not None else ""
            ),
            f"Selected outline scope:\n{request.outline_context}" if request.outline_context else "",
            f"Source outline:\n{source_outline}" if source_outline else "",
            f"{source_coverage}" if source_coverage else "",
            "SERVER STAGE 1 OVERRIDE: Return only a compact JSON skeleton for exactly one chapter.",
            "Include chapter, lesson and unit titles plus an explicit component_plan for every unit. Each plan item must have a supported type, a one-sentence rationale, and the source_fact_ids it serves. Do not generate html, quiz choices, FAQ items, sortable items, crossword words, diagram nodes, or edges yet.",
            "Do not default every unit to html. Choose only evidence-supported formats: html for explanation; problem for assessable concepts; la_diagram and la_sortable for an explicit ordered process/model; la_crossword only for explicit terminology suitable for clues. When the server supplies an approved Blueprint architecture, preserve its required final FAQ in each lesson exactly.",
            "Assign every mandatory source_fact_id from the checklist to exactly one or more relevant units. Do not invent IDs and do not omit checklist IDs.",
            "When an Approved Blueprint content architecture is provided, it is a hard contract: return exactly its chapter, lesson, unit titles, unit order, and component_plan types. Allocate source_fact_ids to that existing topology; never add, remove, rename, merge, reorder, or substitute nodes or component types.",
            "The skeleton must preserve the selected scope and include every unit needed for this chapter. Structural titles remain semantic and must not contain Chương/Bài/Mục numbering or source slide/page suffixes.",
            "Return only one JSON object and keep every title concise.",
        ]
    )
    skeleton_text = ""
    total_usage = AiUsage()
    if not use_direct_v5_skeleton:
        skeleton_text, skeleton_usage = await provider.generate_content(
            request.api_key,
            request.model,
            skeleton_prompt,
            max_output_tokens=min(STAGED_LESSON_AUTHOR_SKELETON_TOKENS, request.max_output_tokens),
            json_mode=True,
            response_schema=build_lesson_author_skeleton_response_schema(),
            thinking_config=types.ThinkingConfig(include_thoughts=False),
        )
        total_usage = combine_usage(skeleton_usage)

    async def generate_stage_two_content(
        *,
        generation_stage: str,
        batch_index: int,
        batch_count: int,
        unit_count: int,
        prompt: str,
        response_schema: types.Schema | type[BaseModel],
    ) -> tuple[str, AiUsage]:
        """Generate one detailed batch under the dedicated Stage-2 budget."""
        nonlocal checkpoint_provider_usage_complete
        # One boundary covers normal content and the existing recovery call.
        # It adds no provider attempt and changes none of the timing/token limits.
        prompt = lesson_output_language_policy(request.locale) + "\n\n" + prompt
        # Stage-2 generation and its existing payload repair share a wire-only
        # projection. Do not alter the authoritative models or other workflows.
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            response_schema, projection = staged_provider_response_model(response_schema)
            logger.info("lesson_author_staged_schema_projection %s", json.dumps({
                "correlation_id": request.correlation_id, "conversation_id": request.conversation_id,
                "generation_stage": generation_stage, "batch_index": batch_index, **projection,
            }, sort_keys=True))
        schema_diagnostics = staged_response_schema_diagnostics(response_schema)
        logger.info(
            "lesson_author_staged_provider_schema generation_stage=%s batch_index=%s batch_count=%s unit_count=%s schema_adapter=%s schema_valid=%s array_field_count=%s array_missing_items_count=%s nullable_array_count=%s provider_schema_fingerprint=%s correlation_id=%s",
            generation_stage,
            batch_index,
            batch_count,
            unit_count,
            schema_diagnostics["schema_adapter"],
            schema_diagnostics["schema_valid"],
            schema_diagnostics["array_field_count"],
            schema_diagnostics["array_missing_items_count"],
            schema_diagnostics["nullable_array_count"],
            schema_diagnostics["provider_schema_fingerprint"],
            request.correlation_id or "none",
        )
        if not schema_diagnostics["schema_valid"]:
            raise WorkflowFailure(
                "PROVIDER_ERROR",
                "The staged lesson response schema is not safe to send to the provider.",
                internal_code="LESSON_PROVIDER_REQUEST_SCHEMA_INVALID",
                failure_stage="staged_lesson_provider_schema_preflight",
                diagnostics=schema_diagnostics,
            )
        try:
            provider_timeout_ms, remaining_workflow_budget_ms = workflow_deadline.stage_two_provider_timeout_ms()
        except HTTPException:
            logger.info(
                "lesson_author_staged_content_batch generation_stage=%s batch_index=%s batch_count=%s unit_count=%s content_output_tokens=%s provider_timeout_ms=%s remaining_workflow_budget_ms=%s duration_ms=%s provider_finish_reason=%s status=workflow_deadline_exhausted correlation_id=%s",
                generation_stage,
                batch_index,
                batch_count,
                unit_count,
                content_output_tokens,
                0,
                0,
                0,
                "unavailable",
                request.correlation_id or "none",
            )
            raise
        provider_finish_reason: str | None = None
        provider_event = "completed"
        response_usage_complete = False
        uncertain_provider_attempt = False
        provider_diagnostics: dict[str, Any] = {}

        def capture_provider_telemetry(metadata: dict[str, Any]) -> None:
            nonlocal provider_event, provider_finish_reason, response_usage_complete, uncertain_provider_attempt
            event = metadata.get("event")
            if event == "provider_request_started" and on_provider_dispatch is not None:
                on_provider_dispatch()
            if isinstance(event, str) and event.strip():
                provider_event = event.strip()[:96]
            finish_reason = metadata.get("provider_finish_reason")
            if isinstance(finish_reason, str) and finish_reason.strip():
                provider_finish_reason = finish_reason.strip()[:96]
            if metadata.get("event") in {"provider_transient_retry", "provider_retry", "provider_unavailable"}:
                uncertain_provider_attempt = True
            if metadata.get("provider_http_status") == 200 and "provider_total_tokens" in metadata:
                counts = [metadata.get(key) for key in ("provider_input_tokens", "provider_output_tokens", "provider_total_tokens")]
                response_usage_complete = metadata.get("usage_source") == "provider" and all(type(v) is int and v >= 0 for v in counts)
                response_usage_complete = response_usage_complete and counts[2] >= counts[0] + counts[1]
            safe_keys = {"provider_http_status", "provider_error_type", "provider_status", "provider_error_category",
                         "provider_message_class", "provider_error_detail_count",
                         "provider_schema_constraint", "provider_error_markers", "usage_source", "provider_input_tokens", "provider_output_tokens",
                         "provider_total_tokens", "provider_finish_reason"}
            provider_diagnostics.update({key: value for key, value in metadata.items() if key in safe_keys})
            trace_outcome = {
                "provider_http_attempt_started": "started",
                "provider_http_attempt_succeeded": "succeeded",
                "provider_response_received": "succeeded",
                "provider_transient_retry": "retrying",
                "provider_timeout": "failed",
                "provider_quota_exhausted": "failed",
                "provider_unavailable": "failed",
                "provider_request_failed": "failed",
            }.get(str(event or ""))
            provider_attempt = metadata.get("provider_attempt")
            if (trace_outcome and type(provider_attempt) is int and 1 <= provider_attempt <= 8
                    and len(checkpoint_attempt_events) < 64):
                raw_usage_source = metadata.get("usage_source")
                trace_usage_source = ("provider_reported" if raw_usage_source == "provider"
                                      else "estimated" if raw_usage_source == "local_estimate" else "unknown")
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
                checkpoint_attempt_events.append({
                    "sequence": len(checkpoint_attempt_events) + 1,
                    "invocation_kind": "writer",
                    "invocation_index": batch_index + 1,
                    "provider_attempt": provider_attempt,
                    "phase": "provider_transport",
                    "outcome": trace_outcome,
                    "event_code": str(event)[:96],
                    "failure_stage": ("provider_transport" if trace_outcome == "failed" else None),
                    "failure_code": (str(metadata.get("internal_failure_code") or event).upper()[:100]
                                     if trace_outcome == "failed" else None),
                    "failure_path": None,
                    "provider_dispatched": True,
                    "usage_source": trace_usage_source,
                    "observed_usage": observed_usage,
                    "duration_ms": max(0, int(metadata.get("duration_ms") or 0)),
                    "diagnostics": diagnostics,
                })
            logger.info("lesson_author_staged_provider_diagnostic %s", json.dumps({
                "correlation_id": request.correlation_id, "conversation_id": request.conversation_id,
                "generation_stage": generation_stage, "batch_index": batch_index, "batch_count": batch_count,
                "event": event or "provider_response_completed", "duration_ms": round((perf_counter() - started_at) * 1000),
                "provider_schema_fingerprint": schema_diagnostics["provider_schema_fingerprint"],
                **provider_diagnostics,
            }, sort_keys=True))

        started_at = perf_counter()
        try:
            content_text, content_usage = await provider.generate_content(
                request.api_key,
                request.model,
                prompt,
                max_output_tokens=content_output_tokens,
                json_mode=True,
                response_schema=response_schema,
                thinking_config=types.ThinkingConfig(include_thoughts=False),
                request_timeout_ms=provider_timeout_ms,
                on_provider_telemetry=capture_provider_telemetry,
            )
            checkpoint_provider_usage_complete = checkpoint_provider_usage_complete and response_usage_complete and not uncertain_provider_attempt
        except Exception as error:
            # `types.Schema` in the installed SDK can throw before a response
            # object exists. Never classify it as a successful empty result
            # and never leak the raw SDK exception through the HTTP boundary.
            error_diagnostics = safe_provider_error_diagnostics(error)
            # Retain actual response usage/status if the SDK returned a response
            # before later parsing failed. Local exceptions aren't HTTP statuses.
            error_diagnostics.update(provider_diagnostics)
            if is_stage_two_provider_schema_error(error):
                provider_event = "provider_request_schema_rejected"
                failure: Exception = WorkflowFailure(
                    "PROVIDER_ERROR",
                    "The provider rejected the staged lesson response schema.",
                    internal_code="LESSON_PROVIDER_REQUEST_SCHEMA_INVALID",
                    failure_stage="staged_lesson_provider_schema_request",
                    diagnostics={
                        **schema_diagnostics,
                        **error_diagnostics,
                    },
                )
            elif isinstance(error, TypeError):
                provider_event = "provider_response_deserialization_failed"
                logger.warning(
                    "lesson_author_staged_provider_deserialization_failed generation_stage=%s batch_index=%s error_type=%s correlation_id=%s",
                    generation_stage,
                    batch_index,
                    type(error).__name__,
                    request.correlation_id or "none",
                )
                failure: Exception = WorkflowFailure(
                    "PROVIDER_ERROR",
                    "The provider response could not be safely deserialized for staged lesson content.",
                    internal_code="LESSON_PROVIDER_RESPONSE_DESERIALIZATION_FAILED",
                    failure_stage="staged_lesson_provider_response",
                    diagnostics={
                        "provider_event": provider_event,
                        **error_diagnostics,
                    },
                )
            elif isinstance(error, (HTTPException, WorkflowFailure)):
                failure = error
            else:
                provider_event = "provider_request_failed"
                failure = WorkflowFailure(
                    "PROVIDER_ERROR", "The staged lesson provider call could not complete safely.",
                    internal_code="LESSON_PROVIDER_REQUEST_FAILED",
                    failure_stage="staged_lesson_provider_request",
                    diagnostics=error_diagnostics,
                )
            logger.info("lesson_author_staged_provider_failure %s", json.dumps({
                "correlation_id": request.correlation_id, "conversation_id": request.conversation_id,
                "generation_stage": generation_stage, "batch_index": batch_index, "batch_count": batch_count,
                "event": "final_failure", "provider_event": provider_event,
                "duration_ms": round((perf_counter() - started_at) * 1000),
                "internal_failure_code": getattr(failure, "internal_code", None),
                "failure_stage": getattr(failure, "failure_stage", "staged_lesson_provider_request"),
                "external_failure_code": "PROVIDER_ERROR",
                "provider_schema_fingerprint": schema_diagnostics["provider_schema_fingerprint"],
                **error_diagnostics,
            }, sort_keys=True))
            logger.info(
                "lesson_author_staged_content_batch generation_stage=%s batch_index=%s batch_count=%s unit_count=%s content_output_tokens=%s provider_timeout_ms=%s remaining_workflow_budget_ms=%s duration_ms=%s provider_event=%s provider_finish_reason=%s status=failed provider_schema_fingerprint=%s correlation_id=%s",
                generation_stage,
                batch_index,
                batch_count,
                unit_count,
                content_output_tokens,
                provider_timeout_ms,
                remaining_workflow_budget_ms,
                max(0, round((perf_counter() - started_at) * 1000)),
                provider_event,
                provider_finish_reason or "unavailable",
                schema_diagnostics["provider_schema_fingerprint"],
                request.correlation_id or "none",
            )
            raise failure
        logger.info(
            "lesson_author_staged_content_batch generation_stage=%s batch_index=%s batch_count=%s unit_count=%s content_output_tokens=%s provider_timeout_ms=%s remaining_workflow_budget_ms=%s duration_ms=%s provider_event=%s provider_finish_reason=%s status=completed provider_schema_fingerprint=%s correlation_id=%s",
            generation_stage,
            batch_index,
            batch_count,
            unit_count,
            content_output_tokens,
            provider_timeout_ms,
            remaining_workflow_budget_ms,
            max(0, round((perf_counter() - started_at) * 1000)),
            provider_event,
            provider_finish_reason or "unavailable",
            schema_diagnostics["provider_schema_fingerprint"],
            request.correlation_id or "none",
        )
        return content_text, content_usage

    def parse_and_validate_skeleton(value: str, label: str) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
        parsed = parse_lesson_author_json_value(value, label)
        normalized = normalize_lesson_author_proposal_tree(
            parsed if isinstance(parsed, dict) else {},
        )
        normalized = _apply_blueprint_architecture_to_skeleton(normalized, request)
        if request.blueprint_architecture is None:
            normalized = consolidate_staged_thin_units(
                normalized,
                source_coverage_manifest,
                request.locale,
            )
        normalized_batches = extract_lesson_author_unit_batches(
            normalized,
            source_coverage_manifest,
            request.locale,
        )
        if not normalized_batches:
            raise LessonAuthorProposalValidationError(
                "Staged skeleton is too small for staged generation.",
            )
        validate_staged_skeleton_source_facts(normalized_batches, source_coverage_manifest)
        return normalized, normalized_batches

    skeleton: dict[str, Any] | None = None
    batches: list[list[dict[str, Any]]] = []
    skeleton_failure: str | None = None
    if use_direct_v5_skeleton:
        skeleton = build_source_locked_staged_skeleton(request, source_coverage_manifest)
        batches = extract_lesson_author_unit_batches(
            skeleton,
            source_coverage_manifest,
            request.locale,
        )
        if not batches:
            raise LessonAuthorProposalValidationError(
                "Approved V5 Blueprint does not contain a unit for staged generation.",
            )
        validate_staged_skeleton_source_facts(batches, source_coverage_manifest)
        logger.info(
            "lesson_author_staged_skeleton_resolved mode=approved_v5_blueprint units=%s",
            sum(len(batch) for batch in batches),
        )
    else:
        try:
            skeleton, batches = parse_and_validate_skeleton(skeleton_text, "skeleton")
        except (HTTPException, LessonAuthorProposalValidationError) as error:
            skeleton_failure = str(getattr(error, "detail", None) or error)
            logger.warning(
                "lesson_author_staged_skeleton_recovery_start reason=%s response_chars=%s",
                skeleton_failure,
                len(skeleton_text),
            )

    if skeleton is None:
        recovery_prompt = "\n\n".join(
            part
            for part in [
                "SERVER STAGE 1 RECOVERY: Return a minimal source-coverage skeleton for exactly one chapter.",
                lesson_output_language_policy(request.locale),
                f"User request:\n{request.user_message}",
                f"Target scope instruction:\n{request.target_scope_instruction}" if request.target_scope_instruction else "",
                (
                    f"Approved Blueprint content architecture (mandatory):\n{request.blueprint_architecture.model_dump_json()}"
                    if request.blueprint_architecture is not None else ""
                ),
                source_coverage,
                (
                    "Use the exact lesson and unit topology from the approved Blueprint content architecture."
                    if request.blueprint_architecture is not None
                    else f"Use exactly one lesson and at most {STAGED_LESSON_AUTHOR_RECOVERY_UNITS} units."
                ),
                "Assign every checklist source_fact_id exactly once. Keep source-page order and group adjacent facts by topic.",
                (
                    "For each unit preserve the exact approved component_plan types and assign the unit source_fact_ids to every plan entry."
                    if request.blueprint_architecture is not None
                    else "For each unit use component_plan with exactly one html entry, a rationale of at most eight words, and the same source_fact_ids. The server will choose additional evidence-supported components later."
                ),
                "Titles must be semantic, concise, unnumbered, and must not contain slide/page ranges.",
                "Return one JSON object only. Do not generate lesson content.",
            ]
            if part
        )
        try:
            recovery_text, recovery_usage = await provider.generate_content(
                request.api_key,
                request.model,
                recovery_prompt,
                max_output_tokens=min(STAGED_LESSON_AUTHOR_SKELETON_TOKENS, request.max_output_tokens),
                json_mode=True,
                response_schema=build_lesson_author_skeleton_response_schema(),
                thinking_config=types.ThinkingConfig(include_thoughts=False),
            )
            total_usage = combine_usage(total_usage, recovery_usage)
            skeleton, batches = parse_and_validate_skeleton(recovery_text, "skeleton recovery")
            logger.info(
                "lesson_author_staged_skeleton_recovered mode=provider units=%s",
                sum(len(batch) for batch in batches),
            )
        except (HTTPException, LessonAuthorProposalValidationError) as error:
            if isinstance(error, HTTPException) and is_non_retryable_provider_error(error):
                raise
            logger.warning(
                "lesson_author_staged_skeleton_recovery_fallback initial_reason=%s recovery_reason=%s",
                skeleton_failure,
                str(getattr(error, "detail", None) or error),
            )
            skeleton = consolidate_staged_thin_units(
                build_source_locked_staged_skeleton(request, source_coverage_manifest),
                source_coverage_manifest,
                request.locale,
            )
            batches = extract_lesson_author_unit_batches(
                skeleton,
                source_coverage_manifest,
                request.locale,
            )
            validate_staged_skeleton_source_facts(batches, source_coverage_manifest)
            logger.warning(
                "lesson_author_staged_skeleton_recovered mode=source_locked units=%s",
                sum(len(batch) for batch in batches),
            )

    # A detailed unit is an independently bounded provider request. Do not
    # compress it to a fixed 4K ceiling: dense source pages and HTML plus an
    # interaction routinely need more room. The configured request ceiling is
    # still authoritative, with a fact-density floor that avoids truncation.
    all_batches = batches
    content_batch_count = len(checkpoint_expected_units(all_batches)) if checkpoint_unit_index is not None else len(batches)
    if checkpoint_unit_index is not None:
        batches = [[select_checkpoint_unit(all_batches, checkpoint_unit_index)]]
    max_facts_per_unit = max(
        (len(unit.get("source_fact_ids", [])) for batch in batches for unit in batch),
        default=1,
    )
    content_output_tokens = staged_lesson_content_output_tokens(
        request_max_output_tokens=request.max_output_tokens,
        max_facts_per_unit=max_facts_per_unit,
    )
    logger.info(
        "lesson_author_staged_plan skeleton_tokens=%s batches=%s units=%s content_tokens_per_batch=%s",
        min(STAGED_LESSON_AUTHOR_SKELETON_TOKENS, request.max_output_tokens),
        content_batch_count,
        sum(len(batch) for batch in batches),
        content_output_tokens,
    )
    content_map: dict[str, dict[str, Any]] = {}
    for selected_batch_index, batch in enumerate(batches, start=1):
        batch_index = checkpoint_unit_index + 1 if checkpoint_unit_index is not None else selected_batch_index
        logger.info(
            "lesson_author_staged_content_batch_start batch=%s total_batches=%s units=%s output_tokens=%s",
            batch_index,
            content_batch_count,
            len(batch),
            content_output_tokens,
        )
        unit_lines = "\n".join(
            f"{index + 1}. Path: {item['unit_path']} | Chương: {item['chapter_title']} > Bài: {item['lesson_title']} > Mục: {item['unit_title']} | "
                f"approved component plan: {json.dumps(item.get('component_plan', []), ensure_ascii=False)} | "
            f"source_fact_ids: {', '.join(item.get('source_fact_ids', [])) or 'none'} | "
            f"supporting_evidence_fact_ids: {', '.join(item.get('supporting_evidence_fact_ids', [])) or 'none'}"
            for index, item in enumerate(batch)
        )
        expected = {
            **batch[0],
            "learner_content_purity": build_learner_content_purity_context(
                batch[0],
                source_coverage_manifest,
                source_rows or [],
            ),
        }
        instance_output = checkpoint_unit_index is not None and bool(expected.get("component_plan")) and all(
            p.get("component_plan_id") for p in expected["component_plan"]
        )
        unit_coverage, unit_context = staged_unit_source_material(
            expected,
            source_rows or [],
            source_coverage_manifest,
        )
        instructional_contract = json.dumps(
            build_staged_instructional_contract(expected, source_coverage_manifest),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        content_prompt = "\n\n".join(
            [
                "SERVER STAGE 2: Generate complete content for exactly one unit.",
                staged_component_contract_prompt(expected.get("component_types", [])),
                unit_lines,
                STAGED_UNIT_UNTRUSTED_RULE,
                f"Mandatory facts for this unit:\n{untrusted_block('MANDATORY_FACTS', unit_coverage)}",
                f"Relevant source material:\n{untrusted_block('SOURCE_MATERIAL', unit_context or context[:STAGED_LESSON_AUTHOR_UNIT_CONTEXT_CHARS])}",
                f"APPROVED INSTRUCTIONAL CONTRACT (hard scope; do not redesign):\n{instructional_contract}",
                (
                    'OUTPUT CONTRACT component-instance-payload-1: Return {"components":{"c0":{...payload...},"c1":{...payload...}}}. '
                    'Each exact cN key is bound to approved component plan index N, NOT a choice of type or evidence owner. '
                    'Return every slot exactly once with only its selected payload fields plus covered_source_fact_ids. '
                    'Never emit unit title/envelope, type, component_plan_id, source_fact_ids, supporting_evidence_fact_ids or any other provenance. '
                    'The server preserves those fields from the approved contract. covered_source_fact_ids is a claim about facts actually represented: '
                    'fully express every assigned owned fact through that slot\'s approved treatment and declare only those exact IDs; supporting-only slots return an empty coverage array. '
                    'Do not merely list IDs without teaching, visualizing, practising, assessing, or clarifying their content as appropriate. '
                    'Slot mapping: ' + json.dumps({f"c{i}": {"type": p["type"], "component_plan_id": p["component_plan_id"]} for i, p in enumerate(expected["component_plan"])})
                    if instance_output else
                    'Return exactly one JSON object for the requested unit, not an array. Use the exact unit title provided.\n\n'
                    'The object must be {"title":"exact unit title","source_fact_ids":["<exact assigned fact_id>"],"supporting_evidence_fact_ids":[],"components":[...]} with real content. Copy only exact canonical source_fact_ids into ownership/coverage fields. If read-only supporting evidence is supplied, return its exact IDs only in supporting_evidence_fact_ids; never copy them into source_fact_ids or covered_source_fact_ids. Components must match the approved component plan exactly. Every component must return source_fact_ids equal to its canonical contract ownership and covered_source_fact_ids that include each owned fact. Every owning component must substantively express its assigned facts using its approved component semantics; do not summarize away source details.'
                ),
                'Use only the selected component contracts above. Never populate content for other component types. Never generate media assets, URLs, storage paths or presentation styles.',
                'Structured evidence rule: source_evidence_bundle is immutable read-only evidence, not new ownership. Preserve table headers, row/column positions, notes and conditions; preserve process order constraints and exceptions; preserve hierarchy parent/priority relations. For visual evidence, use only observed_facts as factual claims. Treat candidate_unverified regions and inference_claims as review cues, never as established domain facts. Never increase source_fact_ids or covered_source_fact_ids because structured/supporting evidence is present.',
                'Quality rules: satisfy every mapped objective with a substantive approved treatment before any related check. Respect instructional_output_budget when present; split decisions were already made upstream, so never inflate this unit into a handbook-sized HTML page. Preserve every required_artifact in the approved component plan using its matching semantic block. A problem must assess evidence represented by a preceding instructional treatment when the approved plan includes one. Do not repeat an explanation, FAQ answer, or question already present in this unit. Do not use an interaction merely for variety. Keep procedures explanatory unless the approved plan explicitly calls for ordering practice. Do not fabricate factual examples; source material is the only source of domain claims.',
                'Ownership boundary: each component may own only the exact canonical facts assigned by its approved plan. HTML teaches prose and structured tables; diagram teaches relationships; sortable practises explicit sequence; crossword reinforces definitions; problem assesses; FAQ clarifies. Supporting-only components must not claim canonical coverage. Components must not duplicate one another, and HTML must not contain interaction questions, answers, choices, instructions or FAQ presentation.',
                'Learner-content purity: use evidence facts for teaching, but keep provenance metadata private. Never write source filenames, citations, page/slide/chunk locators, internal fact/source/component/block IDs, or phrases such as "theo tài liệu nguồn" in any learner-facing field.',
                (f"Every listed source fact must be taught: {', '.join(expected.get('source_fact_ids', [])) or 'none'}." if instance_output else
                 f"Every listed source_fact_id is mandatory for this unit: {', '.join(expected.get('source_fact_ids', [])) or 'none'}. Include all of them in the response."),
                (f"Read-only supporting evidence for grounding, never canonical coverage: {', '.join(expected.get('supporting_evidence_fact_ids', [])) or 'none'}." if instance_output else
                 f"Read-only supporting evidence for this unit: {', '.join(expected.get('supporting_evidence_fact_ids', [])) or 'none'}. Keep it separate from canonical ownership.\n\n"
                 "Preserve each approved component_plan_id on its matching generated component, including repeated types. Never merge or drop instances. Supporting-only components keep canonical source_fact_ids and covered_source_fact_ids empty."),
                "Do not invent facts outside the relevant source material. Do not include markdown or prose outside the JSON object.",
            ]
        )
        content_text, content_usage = await generate_stage_two_content(
            generation_stage="staged_lesson_content",
            batch_index=batch_index,
            batch_count=content_batch_count,
            unit_count=len(batch),
            prompt=content_prompt,
            response_schema=build_staged_instance_response_model(expected["component_plan"]) if instance_output else build_staged_lesson_content_response_model([
                component_type
                for expected in batch
                for component_type in expected.get("component_types", [])
            ], expected_unit_title=batch[0]["unit_title"] if len(batch) == 1 else None),
        )
        total_usage = combine_usage(total_usage, content_usage)
        try:
            parsed = bind_provider_semantic_versions(parse_lesson_author_json_value(content_text, f"content batch {batch_index}"))
        except ProviderSemanticVersionError as error:
            raise WorkflowFailure("LESSON_VALIDATION_FAILED", "Invalid staged HTML format.",
                                  internal_code=str(error), failure_stage="staged_content_validation") from None
        except HTTPException:
            logger.warning(
                "lesson_author_staged_content_batch_unparsed batch=%s response_chars=%s",
                batch_index,
                len(content_text),
            )
            parsed = []
        if instance_output:
            parsed = bind_staged_instance_payload(parsed, expected)
            logger.info("lesson_author_component_binding %s", json.dumps({"correlation_id": request.correlation_id,
                        "contract_version": STAGED_INSTANCE_OUTPUT_CONTRACT_VERSION, "batch_index": batch_index,
                        "component_count": len(expected["component_plan"]), "status": "PASS", "ownership_source": "server"}))
        generated_units = staged_unit_candidates(parsed)
        for expected in batch:
            generated = match_staged_unit_by_title(generated_units, expected["unit_title"])
            expected_fact_ids = set(expected.get("source_fact_ids", []))
            generated_fact_ids = set(generated.get("source_fact_ids", [])) if isinstance(generated, dict) else set()
            generated_supporting_evidence_fact_ids = set(generated.get("supporting_evidence_fact_ids", [])) if isinstance(generated, dict) else set()
            expected_supporting_evidence_fact_ids = set(expected.get("supporting_evidence_fact_ids", []))
            generated_validation_reason = (
                validate_staged_unit_content(generated, expected, strict_payload=any(p.get("component_plan_id") for p in expected.get("component_plan", [])))
                if isinstance(generated, dict)
                else StagedUnitFinding("Unit content is missing or not an object.", "UNIT_OUTPUT_UNRESOLVED")
            )
            if (
                generated is None
                or generated_fact_ids != expected_fact_ids
                or generated_supporting_evidence_fact_ids != expected_supporting_evidence_fact_ids
                or generated_validation_reason
            ):
                repair_baseline = deepcopy(generated)
                repair_guard_finding = staged_component_repair_guard(generated, expected, allow_partial_coverage=True)
                repair_targets = staged_component_repair_targets(generated, expected)
                coverage_findings = staged_coverage_repair_diagnostics(generated, repair_targets)
                coverage_targets = [f["component_index"] for f in coverage_findings]
                repair_contract_version = (STAGED_COMPONENT_COVERAGE_REPAIR_CONTRACT_VERSION if coverage_targets
                                           else STAGED_COMPONENT_REPAIR_CONTRACT_VERSION)
                logger.warning("lesson_author_component_validation %s", json.dumps({
                    "correlation_id": request.correlation_id,
                    "contract_version": STAGED_COMPONENT_CONTRACT_VERSION,
                    "batch_index": batch_index,
                    "stage": "staged_content_validation",
                    "generation_stage": "staged_lesson_content",
                    "validation_finding": generated_validation_reason.diagnostic() if isinstance(generated_validation_reason, StagedUnitFinding) else None,
                    "repair_guard_finding": repair_guard_finding,
                    "findings": staged_payload_diagnostics(generated),
                    "instructional_findings": staged_instructional_diagnostics(generated),
                    "evidence_scope_findings": staged_evidence_scope_diagnostics(generated, expected),
                    "repair_scope": "components" if repair_targets else ("none" if checkpoint_unit_index is not None else "unit"),
                    "repair_component_indices": repair_targets,
                    "coverage_findings": coverage_findings,
                    "coverage_claim_findings": staged_coverage_claim_diagnostics(generated),
                    "coverage_repair_component_indices": coverage_targets,
                    "unit_match": staged_unit_match_diagnostics(generated_units, expected["unit_title"]),
                }))
                if checkpoint_unit_index is not None and (not repair_targets or not isinstance(generated_validation_reason, StagedUnitFinding)
                                                         or not generated_validation_reason.repairable):
                    # No stable authorized payload target: never regenerate the
                    # whole checkpoint unit or let a model fix source ownership.
                    raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The unit has no safe component repair target.",
                                          internal_code="CHAPTER_UNIT_CONTRACT_REJECTED", failure_stage="chapter_component_repair_preflight",
                                          diagnostics={"validation_finding": generated_validation_reason.diagnostic() if isinstance(generated_validation_reason, StagedUnitFinding) else None,
                                                       "repair_guard_finding": repair_guard_finding,
                                                       "coverage_claim_findings": staged_coverage_claim_diagnostics(generated),
                                                       "repair_scope": "none", "repair_component_indices": []})
                recovery_types = [generated["components"][i]["type"] for i in repair_targets] if repair_targets else expected.get("component_types", [])
                multi_repair_slots = len(repair_targets) > 1
                if multi_repair_slots:
                    repair_contract_version = STAGED_MULTI_REPAIR_CONTRACT_VERSION
                expected_line = (
                    f"Chương: {expected['chapter_title']} > "
                    f"Bài: {expected['lesson_title']} > "
                    f"Mục: {expected['unit_title']} | "
                    f"approved component plan: {json.dumps(expected.get('component_plan', []), ensure_ascii=False)}"
                )
                recovery_prompt = "\n\n".join(
                    [
                        "SERVER STAGE 2 RECOVERY: Generate complete content for exactly one unit.",
                        staged_component_contract_prompt(recovery_types),
                        (
                            "SCOPED COMPONENT REPAIR: Return the unit envelope with ONLY the following failed components, in listed order. Do not regenerate other components. Preserve type, component_plan_id, owned/covered/supporting fact IDs exactly. The server preserves all other components. Targets:\n"
                            + json.dumps([repair_baseline["components"][i] for i in repair_targets], ensure_ascii=False)
                            if repair_targets else "Repair the requested unit contract."
                        ),
                        f"Target unit: {expected_line}",
                        f"Approved instructional contract (read-only):\n{instructional_contract}",
                        STAGED_UNIT_UNTRUSTED_RULE,
                        f"Mandatory facts for this unit:\n{untrusted_block('MANDATORY_FACTS', unit_coverage)}",
                        f"Relevant source material:\n{untrusted_block('SOURCE_MATERIAL', unit_context or context[:STAGED_LESSON_AUTHOR_UNIT_CONTEXT_CHARS])}",
                        f"Validation feedback from the previous unit: {generated_validation_reason or 'The unit omitted mandatory source facts.'} Fix this exact issue.",
                        "Return exactly one JSON object, not an array. The title must exactly match the target unit title and no other unit may be returned.",
                        'The object must include exact source_fact_ids and supporting_evidence_fact_ids. Keep supporting evidence read-only: never copy it into canonical ownership or covered_source_fact_ids. Return only authorized failed instances for scoped repair, otherwise match the approved component plan exactly.',
                        f"Approved component plan: {json.dumps(expected.get('component_plan', []), ensure_ascii=False)}",
                        'Use only the selected component contracts above. Never populate content for other component types. Never generate media assets, URLs, storage paths or presentation styles.',
                        f"Include every mandatory source_fact_id for this unit: {', '.join(expected.get('source_fact_ids', [])) or 'none'}.",
                        f"Include read-only supporting evidence separately: {', '.join(expected.get('supporting_evidence_fact_ids', [])) or 'none'}.",
                        "Do not invent facts outside the source material. Do not include markdown or prose outside the JSON object.",
                        "Keep provenance metadata private: no source filename, citation, page/slide/chunk locator, internal ID, or source-attribution phrase may appear in learner-facing content.",
                    ]
                )
                if repair_targets:
                    recovery_prompt = "\n\n".join([
                        "SCOPED COMPONENT REPAIR: " + repair_contract_version,
                        ('Return exactly a components OBJECT with these required slots: '
                         + json.dumps({f"c{i}": {"fixed_type": repair_baseline["components"][i]["type"]} for i in repair_targets})
                         + '. Each slot value contains only the repaired payload fields for its fixed type, NOT fixed_type. '
                         'Return every slot exactly once. Do not return an array or component_index. Server binds each slot to its approved target. '
                         if multi_repair_slots else
                         'Return exactly {"components":[{"component_index":0,...payload fields...}]}. '
                        "Use the exact listed component_index once each, no other addresses. Return only payload fields for each target's fixed type. "
                        ) +
                        "Do NOT return a unit title/envelope, type, component_plan_id, source_fact_ids, "
                        "supporting_evidence_fact_ids, learning blocks, metadata or source references. The server preserves them and all good components unchanged.",
                        (("CONTENT COVERAGE REPAIR: Only these slots may additionally return covered_source_fact_ids: " if multi_repair_slots
                          else "CONTENT COVERAGE REPAIR: Only these component indices may additionally return covered_source_fact_ids: ")
                         + json.dumps([f"c{i}" for i in coverage_targets] if multi_repair_slots else coverage_targets)
                         + ". Rewrite the affected component payload to actually teach or reinforce every assigned fact using the provided evidence. "
                         "Return a truthful complete coverage claim within that component's existing owned IDs only. "
                         "Return an array of non-empty exact references, each at most once. Replace malformed claims rather than copying them. "
                         "Do not merely append IDs; an ID-only change is rejected. Do not invent new terms or exceed the selected component limits. "
                         "If source-grounded complete content is impossible, do not claim coverage. Other targets must omit covered_source_fact_ids."
                         if coverage_targets else "Do NOT return covered_source_fact_ids; the server preserves the existing valid claim."),
                        staged_component_contract_prompt(recovery_types),
                        "Authorized targets (read-only baseline; not the response shape):\n" + json.dumps([
                            {("slot" if multi_repair_slots else "component_index"): f"c{i}" if multi_repair_slots else i,
                             "baseline": repair_baseline["components"][i]}
                            for i in repair_targets
                        ], ensure_ascii=False),
                        "Deterministic payload findings:\n" + json.dumps(staged_payload_diagnostics(repair_baseline)),
                        "Deterministic coverage findings:\n" + json.dumps(coverage_findings),
                        "Coverage claim shape findings:\n" + json.dumps(staged_coverage_claim_diagnostics(repair_baseline)),
                        "Deterministic instructional findings:\n" + json.dumps([
                            f.diagnostic() for i in repair_targets if (f := staged_instructional_finding(
                                repair_baseline["components"][i],
                                i,
                                expected.get("component_plan", [])[i],
                                expected.get("instructional_output_budget"),
                                expected.get("learner_content_purity"),
                            ))
                        ]),
                        f"Approved instructional contract (read-only):\n{instructional_contract}",
                        STAGED_UNIT_UNTRUSTED_RULE,
                        f"Mandatory evidence:\n{untrusted_block('MANDATORY_FACTS', unit_coverage)}",
                        f"Relevant source material:\n{untrusted_block('SOURCE_MATERIAL', unit_context or context[:STAGED_LESSON_AUTHOR_UNIT_CONTEXT_CHARS])}",
                        "Repair the invalid payload shape while retaining instructional completeness and teaching/check alignment. "
                        "No new source claims or media assets. Remove all source filenames, citations, page/slide/chunk locators, internal IDs, and source-attribution phrases from learner-facing fields. Do not emit markdown or text outside the JSON object.",
                    ])
                recovery_validation_reason: str | None = None
                recovery_evidence_findings: list[dict[str, Any]] = []
                recovery_payload_findings: list[dict[str, Any]] = []
                recovery_unit_match: dict[str, Any] | None = None
                recovery_claim_findings: list[dict[str, Any]] = []
                recovery_slot_diagnostics: dict[str, Any] = {}
                try:
                    recovery_text, recovery_usage = await generate_stage_two_content(
                        generation_stage="staged_lesson_content_recovery",
                        batch_index=batch_index,
                        batch_count=content_batch_count,
                        unit_count=1,
                        prompt=recovery_prompt,
                        response_schema=build_staged_multi_repair_model(repair_baseline, repair_targets, coverage_targets) if multi_repair_slots else build_staged_lesson_content_response_model(
                            recovery_types,
                            payload_only=bool(repair_targets),
                            coverage_repair=bool(coverage_targets),
                            coverage_allowed_ids=[fact for i in coverage_targets for fact in repair_baseline["components"][i]["source_fact_ids"]],
                            expected_unit_title=expected["unit_title"],
                        ),
                    )
                    total_usage = combine_usage(total_usage, recovery_usage)
                    recovery_value = bind_provider_semantic_versions(decode_staged_multi_repair(
                        recovery_text, repair_baseline, repair_targets, coverage_targets, recovery_slot_diagnostics,
                    ) if multi_repair_slots else parse_lesson_author_json_value(
                        recovery_text,
                        f"content unit recovery {batch_index}",
                    ))
                    if repair_targets:
                        # Capture safe shape diagnostics before atomic merge
                        # can reject the delta. Never log replacement content.
                        delta_components = recovery_value.get("components", []) if isinstance(recovery_value, dict) else []
                        projected = {"components": []}
                        projected_indices: list[int] = []
                        for change in delta_components if isinstance(delta_components, list) else []:
                            index = change.get("component_index") if isinstance(change, dict) else None
                            if type(index) is int and index in repair_targets:
                                baseline_component = repair_baseline["components"][index]
                                projected["components"].append({**baseline_component, **change, "type": baseline_component["type"],
                                                                "source_fact_ids": baseline_component.get("source_fact_ids", [])})
                                projected_indices.append(index)
                        recovery_payload_findings = staged_payload_diagnostics(projected)
                        recovery_claim_findings = staged_coverage_claim_diagnostics(projected)
                        for finding in recovery_claim_findings:
                            finding["component_index"] = projected_indices[finding["component_index"]]
                            finding["path"] = f"components[{finding['component_index']}].covered_source_fact_ids"
                        for finding in recovery_payload_findings:
                            finding["component_index"] = projected_indices[finding["component_index"]]
                        generated = merge_staged_component_payload_delta(repair_baseline, recovery_value, repair_targets,
                                                                         coverage_targets=coverage_targets)
                    else:
                        recovery_candidates = staged_unit_candidates(recovery_value)
                        recovery_unit_match = staged_unit_match_diagnostics(recovery_candidates, expected["unit_title"])
                        generated = match_staged_unit_by_title(recovery_candidates, expected["unit_title"])
                    recovery_evidence_findings = staged_evidence_scope_diagnostics(generated, expected)
                    recovery_payload_findings = staged_payload_diagnostics(generated)
                    generated_fact_ids = set(generated.get("source_fact_ids", [])) if isinstance(generated, dict) else set()
                    generated_supporting_evidence_fact_ids = set(generated.get("supporting_evidence_fact_ids", [])) if isinstance(generated, dict) else set()
                    recovery_validation_reason = (
                        validate_staged_unit_content(generated, expected, strict_payload=any(p.get("component_plan_id") for p in expected.get("component_plan", [])))
                        if isinstance(generated, dict)
                        else recovery_unit_match["reason"] if recovery_unit_match else "MISSING_OR_INVALID_UNIT"
                    )
                except (HTTPException, LessonAuthorProposalValidationError, ProviderSemanticVersionError) as error:
                    generated = None
                    generated_fact_ids = set()
                    generated_supporting_evidence_fact_ids = set()
                    recovery_validation_reason = str(getattr(error, "detail", None) or error)
                recovery_failure_code = (
                    recovery_validation_reason.code if isinstance(recovery_validation_reason, StagedUnitFinding)
                    else recovery_validation_reason if recovery_validation_reason and re.fullmatch(r"[A-Z_]+", recovery_validation_reason)
                    else "UNIT_REVALIDATION_FAILED" if recovery_validation_reason else None
                )
                logger.info("lesson_author_component_validation %s", json.dumps({
                    "correlation_id": request.correlation_id,
                    "contract_version": STAGED_COMPONENT_CONTRACT_VERSION,
                    "batch_index": batch_index,
                    "stage": "staged_repair_revalidation",
                    "generation_stage": "staged_lesson_content_recovery",
                    "status": "FAIL" if recovery_validation_reason else "PASS",
                    "validation_finding": recovery_validation_reason.diagnostic() if isinstance(recovery_validation_reason, StagedUnitFinding) else None,
                    "repair_scope": "components" if repair_targets else "unit",
                    "repair_component_indices": repair_targets,
                    "repair_contract_version": repair_contract_version if repair_targets else "legacy-unit-recovery",
                    "repair_slot_diagnostics": recovery_slot_diagnostics,
                    "coverage_repair_component_indices": coverage_targets,
                    "coverage_claim_findings": recovery_claim_findings,
                    "findings": recovery_payload_findings,
                    "evidence_scope_findings": recovery_evidence_findings,
                    "unit_match": recovery_unit_match,
                    "failure_code": recovery_failure_code,
                }))
                if checkpoint_unit_index is not None and (generated is None or generated_fact_ids != expected_fact_ids
                        or generated_supporting_evidence_fact_ids != expected_supporting_evidence_fact_ids or recovery_validation_reason):
                    fail_soft_targets = repair_targets or list(range(len(repair_baseline.get("components", []))))
                    try:
                        fail_soft_unit = merge_checkpoint_component_fallback(
                            repair_baseline,
                            checkpoint_component_fallback_unit,
                            fail_soft_targets,
                            expected,
                        )
                        fail_soft_validation = validate_staged_unit_content(
                            fail_soft_unit,
                            expected,
                            strict_payload=True,
                        )
                    except LessonAuthorProposalValidationError:
                        fail_soft_unit = None
                        fail_soft_validation = StagedUnitFinding(
                            "The server component fallback did not preserve the approved unit contract.",
                            "COMPONENT_FAIL_SOFT_INVALID",
                            "unit.components",
                            False,
                        )
                    if fail_soft_unit is not None and fail_soft_validation is None:
                        generated = fail_soft_unit
                        generated_fact_ids = set(expected_fact_ids)
                        generated_supporting_evidence_fact_ids = set(expected_supporting_evidence_fact_ids)
                        recovery_validation_reason = None
                        checkpoint_fallback_component_indices[:] = fail_soft_targets
                        logger.warning("lesson_author_component_fail_soft_recovered %s", json.dumps({
                            "correlation_id": request.correlation_id,
                            "conversation_id": request.conversation_id,
                            "batch_index": batch_index,
                            "unit_path": expected.get("unit_path"),
                            "fallback_component_indices": fail_soft_targets,
                            "preserved_provider_component_count": max(0, len(repair_baseline.get("components", [])) - len(fail_soft_targets)),
                            "repair_failure_code": recovery_failure_code,
                        }, sort_keys=True))
                if checkpoint_unit_index is not None and (generated is None or generated_fact_ids != expected_fact_ids
                        or generated_supporting_evidence_fact_ids != expected_supporting_evidence_fact_ids or recovery_validation_reason):
                    raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The scoped component repair did not pass acceptance.",
                                          internal_code="CHAPTER_COMPONENT_REPAIR_EXHAUSTED", failure_stage="chapter_component_repair_revalidation",
                                          diagnostics={"validation_finding": recovery_validation_reason.diagnostic() if isinstance(recovery_validation_reason, StagedUnitFinding) else None,
                                                       "repair_scope": "components", "repair_component_indices": repair_targets,
                                                       "repair_failure_code": recovery_failure_code,
                                                       "repair_contract_version": repair_contract_version,
                                                       "repair_slot_diagnostics": recovery_slot_diagnostics,
                                                       "coverage_claim_findings": recovery_claim_findings,
                                                       "payload_findings": recovery_payload_findings})
                if (
                    generated is None
                    or generated_fact_ids != expected_fact_ids
                    or generated_supporting_evidence_fact_ids != expected_supporting_evidence_fact_ids
                    or recovery_validation_reason
                ):
                    fallback_expected, dropped_types = prepare_source_locked_expected(
                        expected,
                        source_coverage_manifest,
                    )
                    fallback_unit = staged_source_locked.build_source_locked_html_unit(
                        fallback_expected,
                        source_coverage_manifest,
                    )
                    if fallback_unit is not None:
                        fallback_validation_reason = validate_staged_unit_content(
                            fallback_unit,
                            fallback_expected,
                        )
                        if fallback_validation_reason is None:
                            logger.warning(
                                "lesson_author_staged_unit_source_locked_fallback batch=%s unit_path=%s dropped_types=%s",
                                batch_index,
                                expected["unit_path"],
                                ",".join(dropped_types) or "none",
                            )
                            expected = fallback_expected
                            generated = fallback_unit
                            expected_fact_ids = {
                                str(fact_id).strip()
                                for fact_id in expected.get("source_fact_ids", [])
                                if str(fact_id).strip()
                            }
                            generated_fact_ids = set(expected_fact_ids)
                            generated_supporting_evidence_fact_ids = set(expected.get("supporting_evidence_fact_ids", []))
                            recovery_validation_reason = None
                        else:
                            recovery_validation_reason = (
                                f"{recovery_validation_reason or 'unit không hợp lệ'}; "
                                f"source fallback: {fallback_validation_reason}"
                            )
                if (
                    generated is None
                    or generated_fact_ids != expected_fact_ids
                    or generated_supporting_evidence_fact_ids != expected_supporting_evidence_fact_ids
                    or recovery_validation_reason
                ):
                    raise LessonAuthorProposalValidationError(
                        f"Content batch {batch_index} không trả unit hợp lệ cho '{expected['unit_title']}': "
                        f"{recovery_validation_reason or 'thiếu source fact bắt buộc'}.",
                    )
            if not isinstance(generated, dict):
                raise LessonAuthorProposalValidationError(f"Content batch {batch_index} có unit không hợp lệ.")
            generated["component_plan"] = expected.get("component_plan", [])
            generated["source_fact_ids"] = [
                str(fact_id).strip()
                for fact_id in expected.get("source_fact_ids", [])
                if str(fact_id).strip()
            ]
            generated["supporting_evidence_fact_ids"] = [
                str(fact_id).strip()
                for fact_id in expected.get("supporting_evidence_fact_ids", [])
                if str(fact_id).strip()
            ]
            # Unit titles are human-facing labels and can legitimately repeat
            # in a chapter. The structural path is the only safe assembly key.
            content_map[expected["unit_path"]] = generated
            # Non-blocking observations about the approved contract, never a
            # declaration that provider prose is factually/semantically verified.
            logger.info("lesson_author_instructional_contract_review %s", json.dumps({
                "correlation_id": request.correlation_id,
                "conversation_id": request.conversation_id,
                "batch_index": batch_index,
                **instructional_contract_review_signals(expected),
            }, sort_keys=True))
            # Shadow-only: do not let an observation failure invalidate an
            # otherwise accepted checkpoint or trigger another paid dispatch.
            try:
                observation = observe_lesson_content(generated, expected, source_coverage_manifest)
            except Exception:
                observation = {"observation_version": "lesson-content-observation-1", "mode": "shadow",
                               "blocking": False, "code": "CONTENT_OBSERVATION_UNAVAILABLE",
                               "semantic_fidelity": "not_measured", "semantic_coverage": "not_measured"}
            logger.info("lesson_author_content_observation %s", json.dumps({
                "correlation_id": request.correlation_id, "conversation_id": request.conversation_id,
                "unit_path": expected["unit_path"], "checkpoint_unit_index": checkpoint_unit_index,
                "batch_index": batch_index, **observation,
            }, sort_keys=True))

    if checkpoint_unit_index is not None:
        approved = select_checkpoint_unit(all_batches, checkpoint_unit_index)
        unit = content_map[approved["unit_path"]]
        # Revalidate against the ORIGINAL approved plan, not any recovery/fallback
        # variant. A checkpoint cannot permanently omit a required component.
        reason = validate_staged_unit_content(unit, approved, strict_payload=True)
        if reason:
            raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The unit failed checkpoint acceptance.",
                                  internal_code="CHAPTER_CHECKPOINT_UNIT_INVALID", failure_stage="chapter_checkpoint_unit_validation")
        ChapterCheckpointUnit(unit_index=checkpoint_unit_index, unit=unit)
        return {"checkpoint_version": 1, "unit_index": checkpoint_unit_index, "unit_path": approved["unit_path"],
                "unit": unit, "provider_usage_complete": checkpoint_provider_usage_complete,
                "fallback_component_indices": checkpoint_fallback_component_indices,
                "attempt_trace": checkpoint_attempt_events}, total_usage

    assembled = dict(skeleton)
    assembled_chapters: list[dict[str, Any]] = []
    for chapter_index, chapter_value in enumerate(skeleton.get("chapters", [])[:1], start=1):
        if not isinstance(chapter_value, dict):
            continue
        chapter = dict(chapter_value)
        next_lessons: list[dict[str, Any]] = []
        for lesson_index, lesson_value in enumerate(
            chapter.get("lessons", []) if isinstance(chapter.get("lessons"), list) else [],
            start=1,
        ):
            if not isinstance(lesson_value, dict):
                continue
            lesson = dict(lesson_value)
            next_units: list[dict[str, Any]] = []
            for unit_index, unit_value in enumerate(
                lesson.get("units", []) if isinstance(lesson.get("units"), list) else [],
                start=1,
            ):
                if not isinstance(unit_value, dict):
                    continue
                unit_path = f"chapter_{chapter_index}.lesson_{lesson_index}.unit_{unit_index}"
                unit = content_map.get(unit_path)
                if not unit:
                    raise LessonAuthorProposalValidationError("Không thể ghép đủ nội dung vào skeleton.")
                next_units.append(unit)
            lesson["units"] = next_units
            next_lessons.append(lesson)
        chapter["lessons"] = next_lessons
        assembled_chapters.append(chapter)
    assembled["chapters"] = assembled_chapters
    assembled = normalize_lesson_author_proposal_tree(assembled)
    validate_lesson_author_proposal_shape(assembled)
    quality_metrics = lesson_author_proposal_quality_metrics(assembled)
    logger.info(
        "lesson_author_staged_proposal_quality units=%s component_counts=%s min_html_text_chars=%s source_locked_components=%s",
        quality_metrics["units"],
        json.dumps(quality_metrics["component_counts"], ensure_ascii=False, sort_keys=True),
        quality_metrics["min_html_text_chars"],
        quality_metrics["source_locked_components"],
    )
    return assembled, total_usage
