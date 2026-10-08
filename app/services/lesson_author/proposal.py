"""Legacy lesson proposal workflow (whole-request deadline, staged writer, validation)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import asyncpg
from fastapi import HTTPException
from google.genai import types

from app.core.config import settings
from app.core.logging import SERVICE_LOGGER_NAME
from app.lesson_quality import duplicate_validation_result, pedagogical_validation_result
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorCheckpointRequest, RagLessonAuthorRequest
from app.services import provider
from app.services.deadlines import run_with_deadline
from app.services.lesson_author import checkpoint as checkpoint_service
from app.services.lesson_author.architecture_shape import _workflow_issue
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.lesson_generation import (
    apply_lesson_generation_repair_patches,
    build_lesson_generation_repair_prompt,
    validate_lesson_generation_workflow,
)
from app.services.lesson_author.prompts import build_lesson_author_prompt
from app.services.lesson_author.proposal_validation import (
    normalize_lesson_author_proposal_tree,
    parse_lesson_author_json,
    validate_lesson_author_proposal_shape,
)
from app.services.lesson_author.source_refs import (
    drop_invalid_lesson_author_proposal_source_refs,
    validate_lesson_author_proposal_source_refs,
)
from app.services.lesson_author.staged import writer as staged_writer
from app.services.lesson_author.staged.provider_schemas import build_lesson_author_proposal_response_schema
from app.services.lesson_author.staged.skeleton import should_stage_lesson_author_proposal
from app.services.provider import combine_usage, is_non_retryable_provider_error
from app.services.retrieval import search as retrieval_search
from app.services.retrieval.query import retrieval_limits
from app.services.retrieval.search import build_retrieval_diagnostics, format_sources
from app.services.retrieval.source_coverage import (
    format_source_coverage_manifest,
    validate_lesson_author_source_coverage,
)
from app.workflows.contracts import RepairTarget, WorkflowFailure, WorkflowGenerationResult, WorkflowValidationResult
from app.workflows.lesson_generation import LessonGenerationWorkflowCallbacks, run_lesson_generation_workflow

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def lesson_author_proposal(
    request: RagLessonAuthorRequest,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    return await run_with_deadline(
        "/v1/lesson-author/proposal",
        settings.lesson_author_deadline_ms,
        _lesson_author_proposal(request, pool),
    )


async def _lesson_author_proposal(
    request: RagLessonAuthorRequest,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    workflow_started = perf_counter()

    def emit_lesson_diagnostic(metadata: dict[str, Any]) -> None:
        """Log request-correlated, metadata-only lesson workflow diagnostics."""

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "lesson_generation",
            "workflow_version": "chapter-checkpoint-1" if isinstance(request, RagLessonAuthorCheckpointRequest) else "langgraph-v1",
            "architecture_contract_version": (
                request.blueprint_architecture.architecture_contract_version
                if request.blueprint_architecture is not None else None
            ),
            "correlation_id": request.correlation_id,
            "tenant_id": request.tenant_id,
            "kb_id": request.kb_id,
            "conversation_id": request.conversation_id,
            "source_document_ids": [document.document_id for document in request.source_documents],
            **metadata,
        }
        logger.info("lesson_author_proposal_diagnostic %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))

    emit_lesson_diagnostic({
        "stage": "lesson_author_request",
        "event": "received",
        "repair_pass_number": 0,
        "output_locale": request.locale,
        "language_policy_version": "lesson-language-1",
    })
    rows, retrieval_usage, structure_context = await retrieval_search.retrieve_chunks(pool, request)
    emit_lesson_diagnostic({
        "stage": "rag_retrieval",
        "event": "completed",
        "repair_pass_number": 0,
        "retrieved_chunk_count": len(rows),
        "target_source_scope_hard_locked": bool(structure_context.get("target_source_scope_hard_locked")),
        "target_source_scope_truncated": bool(structure_context.get("target_source_scope_truncated")),
    })
    context, sources = format_sources(rows, max_context_chars=retrieval_limits(request)["max_context_chars"])
    retrieval = build_retrieval_diagnostics(request, rows, sources, structure_context)
    source_coverage_manifest = structure_context.get("source_coverage_manifest")
    source_coverage = format_source_coverage_manifest(source_coverage_manifest)
    missing_blueprint_fact_ids = structure_context.get("blueprint_draft_missing_source_fact_ids", [])
    if missing_blueprint_fact_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LESSON_AUTHOR_BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE",
                "message": "Không thể soạn chương vì một phần nguồn đã khóa trong Bản thiết kế không còn khả dụng. Vui lòng tạo lại Bản thiết kế khóa học.",
                "missing_source_fact_ids": missing_blueprint_fact_ids[:24],
                "retrieval": retrieval,
            },
        )
    if retrieval_search.target_source_scope_is_incomplete(structure_context, rows, sources, source_coverage_manifest):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LESSON_AUTHOR_SOURCE_SCOPE_INCOMPLETE",
                "message": "Không thể soạn chương vì phạm vi tài liệu nguồn chưa được tải đầy đủ. Vui lòng re-index tài liệu rồi thử lại.",
                "retrieval": retrieval,
            },
        )
    if source_coverage_manifest and source_coverage_manifest.get("scope_unresolved"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LESSON_AUTHOR_INFERRED_SCOPE_UNRESOLVED",
                "message": "Không thể ánh xạ đầy đủ các mục của chương vào tài liệu nguồn không có mục lục. Vui lòng re-index tài liệu rồi thử lại.",
                "missing_source_refs": sorted(
                    set(source_coverage_manifest.get("target_source_refs", []))
                    - set(source_coverage_manifest.get("resolved_source_refs", []))
                )[:20],
                "retrieval": retrieval,
            },
        )
    if source_coverage_manifest and source_coverage_manifest.get("truncated"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LESSON_AUTHOR_SOURCE_COVERAGE_TRUNCATED",
                "message": "Phạm vi nội dung nguồn vượt giới hạn kiểm tra đầy đủ; hệ thống không tạo bài học để tránh lược bỏ dữ liệu.",
                "retrieval": retrieval,
            },
        )
    if isinstance(request, RagLessonAuthorCheckpointRequest):
        # The normal endpoint applies this gate in its LangGraph evidence node.
        # Checkpoint mode branches before that graph, so retain the same gate here
        # and require the canonical manifest used to validate cached ownership.
        if not rows or not isinstance(source_coverage_manifest, dict) or not (
            source_coverage_manifest.get("facts") or source_coverage_manifest.get("supporting_evidence_facts")
        ):
            raise WorkflowFailure("SOURCE_EVIDENCE_INSUFFICIENT", "The approved chapter evidence is unavailable.",
                                  internal_code="SOURCE_EVIDENCE_INSUFFICIENT", failure_stage="chapter_checkpoint_evidence_validation")
        return await checkpoint_service.build_lesson_author_checkpoint_result(
            request, context=context, source_outline=structure_context.get("outline", ""), source_coverage=source_coverage,
            rows=rows, manifest=source_coverage_manifest, known_source_refs=set(structure_context.get("known_source_refs", set())),
            retrieval=retrieval, retrieval_usage=retrieval_usage, elapsed_ms=round((perf_counter() - workflow_started) * 1000),
            emit=emit_lesson_diagnostic,
        )
    prompt = build_lesson_author_prompt(
        request,
        context,
        structure_context.get("outline", ""),
        source_coverage,
    )
    async def retrieve_lesson_evidence() -> dict[str, Any]:
        # Retrieval occurred immediately before this request-local graph is
        # entered. Keep raw chunks in the endpoint closure, not graph state.
        return {
            "source_document_count": len(request.source_documents),
            "retrieved_count": len(rows),
            "returned_source_count": len(sources),
            "target_source_scope_hard_locked": bool(structure_context.get("target_source_scope_hard_locked")),
            "target_source_scope_truncated": bool(structure_context.get("target_source_scope_truncated")),
        }

    def validate_lesson_evidence(evidence: dict[str, Any]) -> WorkflowValidationResult:
        if not evidence.get("retrieved_count") or retrieval_search.target_source_scope_is_incomplete(structure_context, rows, sources, source_coverage_manifest):
            return WorkflowValidationResult([
                _workflow_issue("SOURCE_EVIDENCE_INSUFFICIENT", "The approved lesson source scope has insufficient retrieved evidence.", path="lesson"),
            ])
        return WorkflowValidationResult([], {"source_coverage": None})

    async def generate_lesson_candidate() -> WorkflowGenerationResult:
        """The generation node preserves existing staged and JSON safeguards."""
        total_generation_usage = AiUsage()
        proposal: dict[str, Any] | None = None
        last_error: str | None = None
        should_stage = should_stage_lesson_author_proposal(request, context)
        logger.info(
            "lesson_author_proposal_generation_mode conversation_id=%s operation=%s target_type=%s mode=%s context_chars=%s max_output_tokens=%s has_outline_context=%s has_target_scope=%s",
            request.conversation_id,
            request.operation,
            request.target_type or "none",
            "staged" if should_stage else "single",
            len(context),
            request.max_output_tokens,
            bool(request.outline_context.strip()),
            bool(request.target_scope_instruction.strip()),
        )
        if should_stage:
            try:
                staged_candidate, staged_usage = await staged_writer.generate_staged_lesson_author_proposal(
                    request,
                    context,
                    structure_context.get("outline", ""),
                    source_coverage,
                    source_rows=rows,
                    source_coverage_manifest=source_coverage_manifest,
                )
                total_generation_usage = combine_usage(total_generation_usage, staged_usage)
                allowed_source_refs = set(structure_context.get("known_source_refs", set()))
                staged_candidate, dropped_refs = drop_invalid_lesson_author_proposal_source_refs(staged_candidate, allowed_source_refs)
                if dropped_refs:
                    logger.warning("lesson_author_staged_proposal_dropped_unknown_refs refs=%s", ",".join(dropped_refs[:8]))
                validate_lesson_author_proposal_source_refs(staged_candidate, allowed_source_refs)
                validate_lesson_author_source_coverage(staged_candidate, source_coverage_manifest)
                proposal = staged_candidate
                logger.info("lesson_author_proposal_staged_valid context_chars=%s", len(context))
            except (HTTPException, LessonAuthorProposalValidationError) as error:
                if isinstance(error, HTTPException) and is_non_retryable_provider_error(error):
                    raise WorkflowFailure("PROVIDER_ERROR", "The provider rejected the staged lesson generation request.") from error
                last_error = re.sub(r"\s+", " ", str(error)).strip()[:240]
                logger.warning("lesson_author_proposal_staged_failed conversation_id=%s reason=%s", request.conversation_id, last_error or "unknown")
                raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The staged lesson candidate did not satisfy its generation contract.") from error

        if proposal is None:
            for attempt in range(request.max_attempts):
                attempt_prompt = prompt if attempt == 0 else "\n\n".join([
                    prompt,
                    "The previous proposal did not satisfy the server validation contract.",
                    "Regenerate one complete, compact proposal now. Preserve the requested scope, include every chapter, lesson, unit and component required by the schema, ensure every lesson has at least one non-empty unit, use plain title fields without structural numbering, and return only one JSON object.",
                    f"Validation feedback from the previous response: {last_error}. Correct this exact issue in the new JSON; do not repeat the invalid component." if last_error else "",
                ])
                text, generation_usage = await provider.generate_content(
                    request.api_key,
                    request.model,
                    attempt_prompt,
                    max_output_tokens=request.max_output_tokens,
                    json_mode=True,
                    response_schema=build_lesson_author_proposal_response_schema(),
                    thinking_config=types.ThinkingConfig(include_thoughts=False),
                )
                total_generation_usage = combine_usage(total_generation_usage, generation_usage)
                try:
                    candidate = normalize_lesson_author_proposal_tree(parse_lesson_author_json(text, "proposal"))
                    validate_lesson_author_proposal_shape(candidate)
                    allowed_source_refs = set(structure_context.get("known_source_refs", set()))
                    candidate, dropped_refs = drop_invalid_lesson_author_proposal_source_refs(candidate, allowed_source_refs)
                    if dropped_refs:
                        logger.warning("lesson_author_proposal_dropped_unknown_refs refs=%s", ",".join(dropped_refs[:8]))
                    validate_lesson_author_proposal_source_refs(candidate, allowed_source_refs)
                    validate_lesson_author_source_coverage(candidate, source_coverage_manifest)
                    proposal = candidate
                    break
                except HTTPException as error:
                    if is_non_retryable_provider_error(error):
                        raise WorkflowFailure("PROVIDER_ERROR", "The provider rejected lesson proposal generation.") from error
                    last_error = re.sub(r"\s+", " ", str(error)).strip()[:240]
                except LessonAuthorProposalValidationError as error:
                    last_error = re.sub(r"\s+", " ", str(error)).strip()[:240]
                logger.warning("lesson_author_proposal_invalid attempt=%s reason=%s", attempt + 1, last_error or "unknown")
        if proposal is None:
            raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The provider did not return a valid lesson proposal after bounded attempts.")
        return WorkflowGenerationResult(proposal, total_generation_usage.model_dump())

    async def repair_lesson_candidate(
        candidate: dict[str, Any],
        targets: list[RepairTarget],
    ) -> WorkflowGenerationResult:
        repair_prompt = build_lesson_generation_repair_prompt(
            proposal=candidate,
            targets=targets,
            evidence_context=context,
            locale=request.locale,
        )
        text, repair_usage = await provider.generate_content(
            request.api_key,
            request.model,
            repair_prompt,
            max_output_tokens=min(request.max_output_tokens, 16_384),
            json_mode=True,
            thinking_config=types.ThinkingConfig(include_thoughts=False),
        )
        try:
            payload = parse_lesson_author_json(text, "lesson repair")
            repaired = apply_lesson_generation_repair_patches(candidate, targets, payload)
        except WorkflowFailure as error:
            logger.warning(
                "lesson_author_lesson_repair_patch_rejected failure_code=%s",
                error.internal_code or error.code,
            )
            raise
        except HTTPException as error:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Provider returned invalid JSON for scoped lesson repair.",
                internal_code="LESSON_REPAIR_JSON_INVALID",
                failure_stage="lesson_repair_json_parser",
                diagnostics={"provider_status_code": error.status_code},
            ) from error
        except LessonAuthorProposalValidationError as error:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Provider returned a repair payload that failed the lesson contract.",
                internal_code="LESSON_REPAIR_PATCH_INVALID",
                failure_stage="lesson_repair_patch_contract",
            ) from error
        return WorkflowGenerationResult(repaired, repair_usage.model_dump())

    try:
        proposal, workflow = await run_lesson_generation_workflow(
            LessonGenerationWorkflowCallbacks(
                validate_contract=lambda: [],
                retrieve_evidence=retrieve_lesson_evidence,
                validate_evidence=validate_lesson_evidence,
                generate_proposal=generate_lesson_candidate,
                validate_content=lambda candidate: validate_lesson_generation_workflow(
                    candidate,
                    source_coverage_manifest,
                    set(structure_context.get("known_source_refs", set())),
                ),
                validate_pedagogy=lambda candidate: pedagogical_validation_result(
                    candidate,
                    request.blueprint_architecture.model_dump() if request.blueprint_architecture else None,
                ),
                validate_duplicates=duplicate_validation_result,
                repair_content=repair_lesson_candidate,
                emit_diagnostic=emit_lesson_diagnostic,
            ),
            request_context={
                "correlation_id": request.correlation_id,
                "tenant_id": request.tenant_id,
                "kb_id": request.kb_id,
                "conversation_id": request.conversation_id,
                "operation": request.operation,
                "target_type": request.target_type,
                "source_document_count": len(request.source_documents),
            },
            max_repair_attempts=min(2, max(0, settings.lesson_workflow_max_repair_attempts)),
        )
    except WorkflowFailure as error:
        emit_lesson_diagnostic({
            "stage": error.failure_stage or "lesson_validation",
            "event": "final_failure",
            "repair_pass_number": int(error.diagnostics.get("repair_pass_number") or 0),
            "failure_stage": error.failure_stage or "lesson_validation",
            "internal_failure_code": error.internal_code,
            "external_failure_code": error.code,
            "duration_ms": max(0, round((perf_counter() - workflow_started) * 1000)),
        })
        raise HTTPException(
            status_code=422 if error.code in {"SOURCE_EVIDENCE_INSUFFICIENT", "SOURCE_SCOPE_INCOMPLETE"} else 502,
            detail={
                "code": error.code,
                "message": "AI chưa thể tạo đề xuất nội dung bài học hợp lệ.",
                "workflow_issues": error.issues[:20],
            },
        ) from error
    coverage_metrics = validate_lesson_author_source_coverage(proposal, source_coverage_manifest)
    retrieval.update(
        {
            "source_coverage_required_count": coverage_metrics["required_count"],
            "source_coverage_covered_count": coverage_metrics["covered_count"],
            "source_coverage_ratio": coverage_metrics["coverage_ratio"],
            "source_coverage_missing_fact_ids": coverage_metrics["missing_fact_ids"],
            "source_coverage_status": coverage_metrics["status"],
        },
    )
    usage = combine_usage(retrieval_usage, AiUsage(**(workflow.get("usage") or {})))
    return {"proposal": proposal, "usage": usage.model_dump(), "sources": sources, "retrieval": retrieval, "workflow": workflow}
