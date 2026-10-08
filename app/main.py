from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from time import monotonic, perf_counter
from typing import Any, Literal

import asyncpg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.api.deps import get_db, require_internal_token
from app.api.routes import chat as chat_routes
from app.api.routes import health
from app.api.routes import kb as kb_routes
from app.assessment_planner import compile_v5_assessment_plan, materialize_assessment_source_refs
from app.assessment_selection_contract import VERSION as ASSESSMENT_SELECTION_CONTRACT_VERSION
from app.assessment_selection_contract import build_assessment_selection_contract
from app.core.config import settings
from app.core.errors import AppError, app_error_handler, request_validation_error_handler, unhandled_error_handler
from app.core.lifespan import build_lifespan
from app.core.logging import configure_application_logging
from app.core.middleware import RequestBodyLimitMiddleware
from app.core.request_context import DisconnectCancellationMiddleware, RequestContextMiddleware
from app.idm.course_design import run_idm_course_design
from app.idm.diagram import idm_source_step_diagram
from app.idm.module_design import run_idm_module_design
from app.idm.runtime import (
    RUN_STOPPING_PROVIDER_CODES,
    TRANSIENT_PROVIDER_CODES,
    IdmError,
    IdmProviderError,
    IdmRuntime,
    IdmStageError,
    IdmTokenAllowance,
)
from app.idm.source_locked import idm_source_faq, idm_source_grounded_single_choice, render_idm_source_locked_html
from app.idm.storyboard import IdmUnitDeps, run_idm_unit
from app.instructional_density import (
    INSTRUCTIONAL_DENSITY_POLICY_VERSION,
    SOURCE_SCOPE_CHUNKS_PER_GROUP,
    partition_chunk_facts_for_density,
)
from app.instructional_opportunities import VERSION as EVIDENCE_TREATMENT_VERSION
from app.instructional_opportunities import compile_evidence_treatments
from app.instructional_quality import source_relationship_pairs
from app.learner_content_purity import build_learner_content_purity_context
from app.lesson_author_blueprint import (
    SEMANTIC_REPAIR_CONTRACT_VERSION,
    LessonAuthorBlueprintValidationError,
    build_course_architecture_repair_response_schema,
    build_v5_semantic_delta_repair_response_schema,
    validate_lesson_author_blueprint,
)
from app.lesson_author_orchestration_v2 import OrchestrationContractError
from app.lesson_author_orchestration_v2 import canonical_hash as orchestration_v2_canonical_hash
from app.lesson_author_orchestration_v2_provider import (
    CHAPTER_SHARD_PROVIDER_SCHEMA_METADATA,
    COURSE_SKELETON_PROVIDER_SCHEMA_METADATA,
    ChapterShardProviderWireV2,
    CourseSkeletonDraftV2,
    CourseSkeletonProviderWireV2,
    SourceOutlineAuthorityV2,
    SourceOutlineChapterV2,
    SourceSnapshotFactV2,
    UnitGenerationContractV2,
    bind_chapter_shard_v2,
    bind_course_skeleton_v2,
    chapter_shard_prompt_v2,
    fallback_chapter_shard_draft_v2,
    fallback_course_skeleton_draft_v2,
    parse_chapter_shard_draft_v2,
    salvage_chapter_shard_draft_v2,
    skeleton_prompt_v2,
    unit_contract_manifest_v2,
    unit_contract_v5_architecture_v2,
)
from app.lesson_quality import duplicate_validation_result, pedagogical_validation_result
from app.schemas.chat import RagChatRequest
from app.schemas.common import AiUsage
from app.schemas.lesson_author import (
    RagLessonAuthorBlueprintRequest,
    RagLessonAuthorCheckpointRequest,
    RagLessonAuthorRequest,
)
from app.schemas.orchestration_v2 import (
    RagLessonAuthorChapterShardV2Request,
    RagLessonAuthorCourseSkeletonV2Request,
    RagLessonAuthorSourceSnapshotV2Request,
    RagLessonAuthorUnitV2Request,
)
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
from app.services import runtime as service_runtime
from app.services.deadlines import record_fallback, run_with_deadline
from app.services.ingestion.chunking import SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED, SOURCE_EVIDENCE_READY
from app.services.lesson_author import blueprint as blueprint_service
from app.services.lesson_author import checkpoint as checkpoint_service
from app.services.lesson_author.architecture_shape import (
    _workflow_issue,
    _workflow_issue_from_blueprint_validation_error,
)
from app.services.lesson_author.architecture_validation import (
    validate_course_architecture_workflow,
    validate_v5_instructional_coherence,
)
from app.services.lesson_author.blueprint import (
    CourseArchitectSemanticScopeError,
    LessonAuthorBlueprintGenerationError,
    lesson_author_blueprint_failure_message,
    validate_lesson_author_source_refs,
)
from app.services.lesson_author.checkpoint import (
    build_lesson_author_checkpoint_result as build_lesson_author_checkpoint_result,
)
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.evidence_scope import (
    allocate_source_map_architecture_facts,
    source_fact_allocation_diagnostics,
    validate_course_architecture_evidence_scope,
)
from app.services.lesson_author.granularity import (
    allocate_blueprint_source_fact_ids,
    ensure_blueprint_source_granularity,
)
from app.services.lesson_author.lesson_generation import (
    apply_lesson_generation_repair_patches,
    build_lesson_generation_repair_prompt,
    validate_lesson_generation_workflow,
)
from app.services.lesson_author.media_review import MEDIA_REVIEW_VERSION, enrich_lesson_author_blueprint_media_review
from app.services.lesson_author.prompts import build_lesson_author_blueprint_prompt, build_lesson_author_prompt
from app.services.lesson_author.proposal_validation import (
    normalize_lesson_author_proposal_tree,
    parse_course_architecture_repair_payload,
    parse_lesson_author_json,
    validate_lesson_author_proposal_shape,
)
from app.services.lesson_author.proposal_validation import (
    semantic_learning_visible_text as semantic_learning_visible_text,
)
from app.services.lesson_author.repair.candidate import (
    _deterministic_factless_unit_removal_candidate,
    _v5_deterministic_evidence_alignment_candidate,
)
from app.services.lesson_author.repair.prompt import (
    _v5_semantic_delta_operations_for_targets,
    build_course_architecture_repair_prompt,
    build_v5_scoped_repair_source_context,
)
from app.services.lesson_author.repair.semantic_delta import apply_course_architecture_repair_patches
from app.services.lesson_author.repair.targets import (
    _v5_deterministic_assessment_alignment_payload,
    _v5_prepare_semantic_repair_targets,
    _v5_safe_action_intent_repair_metadata,
)
from app.services.lesson_author.source_context import (
    MAX_WORKFLOW_REPAIR_TARGET_CHARS,
    assert_v5_immutable_source_context,
    assert_v5_scoped_repair_target_bound,
    create_v5_immutable_source_context,
)
from app.services.lesson_author.source_refs import (
    drop_invalid_lesson_author_proposal_source_refs,
    update_blueprint_source_coverage,
    validate_lesson_author_proposal_source_refs,
)
from app.services.lesson_author.staged import writer as staged_writer
from app.services.lesson_author.staged.provider_schemas import (
    bind_staged_instance_payload,
    build_lesson_author_proposal_response_schema,
    build_staged_instance_response_model,
    build_staged_multi_repair_model,
    decode_staged_multi_repair,
)
from app.services.lesson_author.staged.provider_schemas import (
    staged_component_payload_code as staged_component_payload_code,
)
from app.services.lesson_author.staged.skeleton import should_stage_lesson_author_proposal
from app.services.lesson_author.staged.validation import (
    StagedUnitFinding,
    merge_staged_component_payload_delta,
    validate_staged_unit_content,
)
from app.services.meta import API_VERSION
from app.services.orchestration_v2.source_locked import (
    build_orchestration_v2_source_locked_components,
    build_orchestration_v2_source_locked_unit,
)
from app.services.provider import (
    PROVIDER_TRANSIENT_MAX_ATTEMPTS,
    combine_usage,
    is_non_retryable_provider_error,
    normalize_embedding_model,
    provider_http_error_status,
    require_provider_api_key,
    safe_provider_error_diagnostics,
)
from app.services.provider import generate_content as generate_content
from app.services.retrieval import search as retrieval_search
from app.services.retrieval.query import decode_json_object, retrieval_limits
from app.services.retrieval.search import build_retrieval_diagnostics, format_sources
from app.services.retrieval.source_coverage import (
    extract_source_coverage_facts,
    format_course_architecture_coverage_contract,
    format_source_coverage_manifest,
    validate_lesson_author_source_coverage,
)
from app.services.text import clean_text
from app.source_chapter_policy import bind_source_chapters, resolve_source_chapter_policy
from app.source_map import build_course_architect_context, build_source_map
from app.workflows.contracts import (
    RepairTarget,
    WorkflowFailure,
    WorkflowGenerationResult,
    WorkflowIssue,
    WorkflowValidationResult,
)
from app.workflows.course_architecture import (
    V5_MAX_PROVIDER_REPAIR_CALLS,
    V5_MAX_REPAIR_ATTEMPTS_PER_LAYER,
    CourseArchitectureWorkflowCallbacks,
    run_course_architecture_workflow,
)
from app.workflows.lesson_generation import LessonGenerationWorkflowCallbacks, run_lesson_generation_workflow

app = FastAPI(
    title="Internal AI RAG Service",
    version=API_VERSION,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None,
    # Resolved at call time: startup/shutdown are defined further down.
    lifespan=build_lifespan(
        state=service_runtime.runtime_state,
        startup=lambda: service_runtime.startup(),
        shutdown=lambda: service_runtime.shutdown(),
        grace_seconds=settings.shutdown_grace_seconds,
    ),
)


# add_middleware wraps outward: request context (outermost) -> body limit ->
# disconnect cancellation (buffers the already size-checked body) -> routes.
app.add_middleware(DisconnectCancellationMiddleware)


app.add_middleware(
    RequestBodyLimitMiddleware,
    max_request_bytes=settings.max_request_bytes,
    idm_max_request_bytes=settings.idm_max_request_bytes,
)


app.add_middleware(RequestContextMiddleware, state=service_runtime.runtime_state)


app.add_exception_handler(AppError, app_error_handler)


app.add_exception_handler(Exception, unhandled_error_handler)


app.add_exception_handler(RequestValidationError, request_validation_error_handler)


app.include_router(health.router)


app.include_router(kb_routes.router)


app.include_router(chat_routes.router)


# Configure the package logger so every app.* module (app.idm, app.core.request_context, ...)
# emits through the JSON handler, not only this module.
configure_application_logging("app")


logger = logging.getLogger(__name__)


# Keep the outer Node -> Python request alive long enough to serialize and
# return a deterministic source-locked fallback after an inner provider
# deadline. Without this gap, a provider timeout and the HTTP client timeout
# race at the same millisecond and turn a valid fallback into outcome_unknown.
ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS = 5_000


@app.post("/v1/lesson-author/chapter-checkpoint", dependencies=[Depends(require_internal_token)])
async def lesson_author_chapter_checkpoint(
    request: RagLessonAuthorCheckpointRequest,
    pool: asyncpg.Pool = Depends(get_db),
) -> dict[str, Any]:
    return await run_with_deadline(
        "/v1/lesson-author/chapter-checkpoint",
        settings.lesson_author_deadline_ms,
        _lesson_author_chapter_checkpoint(request, pool),
    )


async def _lesson_author_chapter_checkpoint(
    request: RagLessonAuthorCheckpointRequest,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        # Covers retrieval and final validation too. Cancellation of a to_thread
        # await is NOT proof that the synchronous Google HTTP request was stopped.
        return await asyncio.wait_for(lesson_author_proposal(request, pool), request.remaining_workflow_budget_ms / 1000)
    except asyncio.TimeoutError as error:
        failure = WorkflowFailure("PROVIDER_ERROR", "The chapter invocation deadline expired.",
                                  internal_code="AI_STAGED_LESSON_WORKFLOW_TIMEOUT", failure_stage="chapter_checkpoint_deadline")
        cause: Exception = error
    except WorkflowFailure as error:
        failure, cause = error, error
    except HTTPException as error:
        detail = error.detail if isinstance(error.detail, dict) else {}
        code = str(detail.get("code") or "SOURCE_SCOPE_INCOMPLETE")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", code):
            code = "SOURCE_SCOPE_INCOMPLETE"
        provider_error = code.startswith("AI_PROVIDER_") or code == "AI_STAGED_LESSON_WORKFLOW_TIMEOUT"
        failure = WorkflowFailure("PROVIDER_ERROR" if provider_error else "SOURCE_SCOPE_INCOMPLETE",
                                  "The chapter checkpoint request could not complete.", internal_code=code,
                                  failure_stage="chapter_checkpoint_provider" if provider_error else "chapter_checkpoint_source_validation")
        cause = error
    except (ValueError, LessonAuthorProposalValidationError) as error:
        failure = WorkflowFailure("LESSON_VALIDATION_FAILED", "The chapter checkpoint contract is invalid.",
                                  internal_code="CHAPTER_CHECKPOINT_CONTRACT_INVALID", failure_stage="chapter_checkpoint_contract")
        cause = error
    except Exception as error:
        failure = WorkflowFailure("LESSON_VALIDATION_FAILED", "The chapter checkpoint operation failed.",
                                  internal_code="CHAPTER_CHECKPOINT_INTERNAL_ERROR", failure_stage="chapter_checkpoint_internal")
        cause = error
    diagnostics = {"workflow": "lesson_generation", "event": "final_failure",
                   "correlation_id": request.correlation_id, "conversation_id": request.conversation_id,
                   "checkpoint_action": request.checkpoint_action, "unit_index": request.checkpoint_unit_index,
                   "failure_stage": failure.failure_stage, "internal_failure_code": failure.internal_code,
                   "external_failure_code": failure.code, "duration_ms": round((perf_counter() - started) * 1000)}
    if failure.internal_code in {"CHAPTER_UNIT_CONTRACT_REJECTED", "CHAPTER_COMPONENT_REPAIR_EXHAUSTED"}:
        diagnostics.update(failure.diagnostics)
    if failure.code == "PROVIDER_ERROR":
        provider_keys = {"provider_http_status", "provider_error_type", "provider_status", "provider_error_category",
                         "provider_message_class", "provider_error_detail_count",
                         "provider_schema_constraint", "provider_error_markers", "usage_source", "provider_input_tokens", "provider_output_tokens",
                         "provider_total_tokens", "provider_finish_reason", "provider_schema_fingerprint"}
        diagnostics.update({key: value for key, value in failure.diagnostics.items() if key in provider_keys})
    logger.info("lesson_author_checkpoint_diagnostic %s", json.dumps(diagnostics, sort_keys=True))
    raise HTTPException(status_code=502 if failure.code == "PROVIDER_ERROR" else 422, detail={
        "code": failure.code, "message": "The chapter could not complete its validated checkpoint operation.",
        "internal_failure_code": failure.internal_code, "failure_stage": failure.failure_stage,
        "correlation_id": request.correlation_id,
    }) from cause


@app.post("/v1/lesson-author/proposal", dependencies=[Depends(require_internal_token)])
async def lesson_author_proposal(
    request: RagLessonAuthorRequest,
    pool: asyncpg.Pool = Depends(get_db),
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


# Legacy V2 skeleton/shard providers fall back on these codes; AI_PROVIDER_RATE_LIMITED was
# reported as AI_PROVIDER_QUOTA_EXHAUSTED before the 429 classification and keeps that branch.
ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES = frozenset({
    "AI_PROVIDER_TIMEOUT", "AI_PROVIDER_UNAVAILABLE", "AI_PROVIDER_QUOTA_EXHAUSTED", "AI_PROVIDER_RATE_LIMITED",
})


def _orchestration_v2_http_error(code: str, message: str, *, status_code: int = 422) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


async def _orchestration_v2_generate_content(
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_output_tokens: int,
    response_schema: type[BaseModel],
    correlation_id: str,
    generation_stage: str,
) -> tuple[str, AiUsage]:
    """Call Gemini with a safe, terminal classification for definitive refusals.

    A provider HTTP 400/401/403 proves that this request was rejected rather
    than accepted for generation. Expose only a stable internal code so the
    durable worker can release its reservation and fail immediately. Network,
    timeout, and 5xx ambiguity retain the existing outcome-unknown path.
    """

    try:
        return await provider.generate_content(
            api_key,
            model,
            prompt,
            max_output_tokens=max_output_tokens,
            json_mode=True,
            response_schema=response_schema,
        )
    except HTTPException:
        raise
    except Exception as error:
        status = provider_http_error_status(error)
        diagnostics = safe_provider_error_diagnostics(error)
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "provider_request_rejected" if status in {400, 401, 403} else "provider_request_failed",
            "correlation_id": correlation_id,
            "generation_stage": generation_stage,
            "model": model,
            **diagnostics,
        }, sort_keys=True))
        if status == 400:
            raise _orchestration_v2_http_error(
                "AI_PROVIDER_REQUEST_REJECTED",
                "AI provider rejected the structured generation request.",
                status_code=502,
            ) from error
        if status in {401, 403}:
            raise _orchestration_v2_http_error(
                "AI_PROVIDER_AUTH_REJECTED",
                "AI provider rejected the configured credentials or access policy.",
                status_code=502,
            ) from error
        raise


@app.post("/v1/lesson-author/orchestration-v2/source-snapshot", dependencies=[Depends(require_internal_token)])
async def lesson_author_orchestration_v2_source_snapshot(
    request: RagLessonAuthorSourceSnapshotV2Request,
    pool: asyncpg.Pool = Depends(get_db),
) -> dict[str, Any]:
    """Return one deterministic source page without a provider call or whole-source materialization."""

    document_ids = sorted(document.document_id for document in request.source_documents)
    cursor = request.cursor
    if cursor is not None and str(cursor["document_id"]) not in set(document_ids):
        raise _orchestration_v2_http_error(
            "SOURCE_CURSOR_INVALID", "The source cursor is outside the selected source authority.",
        )
    indexes = await pool.fetch(
        """
        SELECT r.id::text AS index_id,r.document_id::text AS document_id,r.content_sha256,
               r.chunk_count,r.embedding_model,r.embedding_dimensions
        FROM rag_document_indexes r
        WHERE r.tenant_id=$1::uuid AND r.kb_id=$2::uuid AND r.document_id=ANY($3::uuid[])
          AND r.engine='self_built_rag' AND r.status='learned' AND r.is_active=true
          AND r.embedding_model=$4 AND r.embedding_dimensions=$5::int
        ORDER BY r.document_id
        """,
        request.tenant_id, request.kb_id, document_ids,
        normalize_embedding_model(request.embedding_model), request.embedding_dimensions,
    )
    if len(indexes) != len(document_ids) or {str(row["document_id"]) for row in indexes} != set(document_ids):
        raise _orchestration_v2_http_error(
            "SOURCE_REVISION_UNAVAILABLE", "The selected learned source revision is unavailable.",
        )
    structure_rows = await pool.fetch(
        """
        SELECT c.document_id::text AS document_id,
               (jsonb_agg(c.metadata->'source_structure' ORDER BY c.chunk_no)
                 FILTER (WHERE c.metadata ? 'source_structure'))->0 AS source_structure,
               min(lower(c.metadata->>'source_evidence_revision')) FILTER (
                 WHERE coalesce(c.metadata->>'source_evidence_revision','') ~ '^[0-9a-fA-F]{64}$'
               ) AS source_evidence_revision,
               count(*)::integer AS actual_chunk_count,
               count(*) FILTER (
                 WHERE coalesce(c.metadata->>'source_evidence_revision','') ~ '^[0-9a-fA-F]{64}$'
               )::integer AS evidence_revision_chunk_count,
               count(DISTINCT lower(c.metadata->>'source_evidence_revision')) FILTER (
                 WHERE coalesce(c.metadata->>'source_evidence_revision','') ~ '^[0-9a-fA-F]{64}$'
               )::integer AS evidence_revision_distinct_count
        FROM rag_chunks c
        WHERE c.tenant_id=$1::uuid AND c.kb_id=$2::uuid AND c.index_id=ANY($3::uuid[])
        GROUP BY c.document_id
        ORDER BY c.document_id
        """,
        request.tenant_id, request.kb_id, [str(row["index_id"]) for row in indexes],
    )
    structures_by_document = {
        str(row["document_id"]): decode_json_object(row["source_structure"])
        for row in structure_rows
        if row.get("source_structure") is not None
    }
    structure_rows_by_document = {str(row["document_id"]): row for row in structure_rows}
    evidence_revisions_by_document: dict[str, str] = {}
    evidence_status_by_document: dict[str, str] = {}
    for index_row in indexes:
        document_id = str(index_row["document_id"])
        declared_chunks = int(index_row["chunk_count"] or 0)
        summary = structure_rows_by_document.get(document_id) or {}
        revision = str(summary.get("source_evidence_revision") or "").strip().casefold()
        valid_revision = revision if re.fullmatch(r"[0-9a-f]{64}", revision) else None
        actual_chunks = int(summary.get("actual_chunk_count") if summary.get("actual_chunk_count") is not None
                            else declared_chunks)
        revision_chunks = int(summary.get("evidence_revision_chunk_count")
                              if summary.get("evidence_revision_chunk_count") is not None
                              else declared_chunks if valid_revision else 0)
        revision_count = int(summary.get("evidence_revision_distinct_count")
                             if summary.get("evidence_revision_distinct_count") is not None
                             else 1 if valid_revision else 0)
        if declared_chunks < 1 or actual_chunks != declared_chunks:
            raise _orchestration_v2_http_error(
                "SOURCE_EVIDENCE_REVISION_INCONSISTENT",
                "The learned source index does not match its declared chunk inventory.",
                status_code=409,
            )
        if revision_chunks == 0 and revision_count == 0 and valid_revision is None:
            evidence_status_by_document[document_id] = SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED
        elif (revision_chunks == declared_chunks and revision_count == 1
              and valid_revision is not None):
            evidence_status_by_document[document_id] = SOURCE_EVIDENCE_READY
            evidence_revisions_by_document[document_id] = valid_revision
        else:
            raise _orchestration_v2_http_error(
                "SOURCE_EVIDENCE_REVISION_INCONSISTENT",
                "The learned source index contains mixed structured-evidence revisions.",
                status_code=409,
            )
    policy_documents = [{
        "document_id": document_id,
        "structure": structures_by_document.get(document_id),
    } for document_id in document_ids]
    chapter_policy = resolve_source_chapter_policy(policy_documents)
    policy_mode = str(chapter_policy.get("mode") or "NEEDS_STRUCTURE_REVIEW")
    authority_mode: Literal["locked", "model_designed", "needs_review"] = (
        "locked" if policy_mode.startswith("SOURCE_LOCKED")
        else "model_designed" if policy_mode == "MODEL_DESIGNED"
        else "needs_review"
    )
    authority_source: Literal["toc", "headings", "none", "ambiguous"] = (
        "toc" if policy_mode == "SOURCE_LOCKED_TOC"
        else "headings" if policy_mode == "SOURCE_LOCKED_HEADINGS"
        else "none" if authority_mode == "model_designed"
        else "ambiguous"
    )
    chapters = [SourceOutlineChapterV2(
        order=index,
        document_id=str(chapter["document_id"]),
        source_ref=str(chapter["source_ref"]),
        title=str(chapter["title"]),
    ) for index, chapter in enumerate(chapter_policy.get("chapters", []))] if authority_mode == "locked" else []
    confidences = [float(structure.get("confidence") or 0) for structure in structures_by_document.values()
                   if isinstance(structure, dict)]
    reason_codes = [str(value)[:120] for value in chapter_policy.get("reason_codes", [])[:32]]
    legacy_document_count = sum(
        status == SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED
        for status in evidence_status_by_document.values()
    )
    if legacy_document_count and "STRUCTURED_EVIDENCE_REVISION_MISSING" not in reason_codes:
        reason_codes = [*reason_codes[:31], "STRUCTURED_EVIDENCE_REVISION_MISSING"]
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "source_snapshot_legacy_evidence_review_required",
            "correlation_id": request.correlation_id,
            "source_snapshot_hash": request.source_snapshot_hash,
            "legacy_document_count": legacy_document_count,
            "selected_document_count": len(document_ids),
        }, sort_keys=True))
    authority_payload = {
        "mode": authority_mode,
        "source": authority_source,
        "complete": bool(chapter_policy.get("complete")),
        "confidence": min(confidences) if confidences else 0.0,
        "reason_codes": reason_codes,
        "chapters": [chapter.model_dump(mode="json") for chapter in chapters],
    }
    source_authority = SourceOutlineAuthorityV2(
        **authority_payload,
        structure_hash=orchestration_v2_canonical_hash(authority_payload),
    )
    revision_payload: list[dict[str, Any]] = []
    for row in indexes:
        document_id = str(row["document_id"])
        revision_entry: dict[str, Any] = {
            "document_id": document_id,
            "index_id": str(row["index_id"]),
            "content_sha256": str(row["content_sha256"] or ""),
            "chunk_count": int(row["chunk_count"] or 0),
            "embedding_model": str(row["embedding_model"]),
            "embedding_dimensions": int(row["embedding_dimensions"]),
            "source_evidence_status": evidence_status_by_document[document_id],
        }
        # Preserve the exact legacy revision for already learned documents.
        # A new revision is materialized only after re-indexing has produced
        # explicit structured-evidence metadata.
        evidence_revision = evidence_revisions_by_document.get(document_id)
        if evidence_revision is not None:
            revision_entry["source_evidence_revision"] = evidence_revision
        revision_payload.append(revision_entry)
    source_revision = orchestration_v2_canonical_hash(revision_payload)
    if request.expected_source_revision and request.expected_source_revision != source_revision:
        raise _orchestration_v2_http_error(
            "SOURCE_REVISION_CHANGED", "The learned source revision changed while paging.", status_code=409,
        )

    index_ids = [str(row["index_id"]) for row in indexes]
    after_document_id = str(cursor["document_id"]) if cursor else None
    after_chunk_no = int(cursor["chunk_no"]) if cursor else -1
    row_limit = 128
    fetched = await pool.fetch(
        """
        SELECT c.content,c.source_page,c.source_section,c.metadata,c.chunk_no,c.index_id::text AS index_id,
               c.document_id::text AS document_id,d.name AS document_name
        FROM rag_chunks c
        JOIN kb_documents d ON d.id=c.document_id AND d.tenant_id=c.tenant_id AND d.kb_id=c.kb_id
        WHERE c.tenant_id=$1::uuid AND c.kb_id=$2::uuid AND c.index_id=ANY($3::uuid[])
          AND ($4::uuid IS NULL OR c.document_id>$4::uuid
               OR (c.document_id=$4::uuid AND c.chunk_no>$5::int))
        ORDER BY c.document_id,c.chunk_no
        LIMIT $6::int
        """,
        request.tenant_id, request.kb_id, index_ids, after_document_id, after_chunk_no, row_limit + 1,
    )
    document_order = {document_id: index + 1 for index, document_id in enumerate(document_ids)}
    facts: list[SourceSnapshotFactV2] = []
    content_bytes = 0
    last_cursor: dict[str, Any] | None = None
    has_more = len(fetched) > row_limit
    for raw_row in fetched[:row_limit]:
        row = dict(raw_row)
        document_id = str(row["document_id"])
        chunk_no = int(row["chunk_no"])
        metadata = decode_json_object(row.get("metadata")) or {}
        content_kinds = {
            str(value).strip().casefold()
            for value in metadata.get("content_kinds", [])
            if isinstance(value, str) and value.strip()
        } if isinstance(metadata.get("content_kinds"), list) else set()
        table_count = max(0, min(100, int(metadata.get("table_count") or 0))) \
            if str(metadata.get("table_count") or "0").isdigit() else 0
        fact_texts = extract_source_coverage_facts(
            clean_text(str(row.get("content") or "")),
            preserve_table_numeric=table_count > 0 or "table" in content_kinds,
        )
        page = row.get("source_page") if isinstance(row.get("source_page"), int) and row.get("source_page") > 0 else None
        source_ref = str(metadata.get("source_ref") or row.get("source_section") or "").strip()[:255] or None
        heading = metadata.get("heading_path")
        if isinstance(heading, list):
            heading = " › ".join(
                str(value).strip()[:120] for value in heading[:8] if str(value).strip()
            )
        scope_title = str(heading or row.get("source_section") or "").strip()
        if not scope_title:
            suffix = f" · page {page}" if page is not None else f" · chunk {chunk_no + 1}"
            scope_title = f"{str(row.get('document_name') or 'Source document')}{suffix}"
        scope_title = scope_title[:500]
        visual_regions = metadata.get("visual_regions")
        if not isinstance(visual_regions, list):
            visual_regions = []
        visual_regions = [region for region in visual_regions[:4] if isinstance(region, dict)]
        visual_prompt_text = str(metadata.get("visual_prompt_text") or "").strip()[:600] or None
        # Scope V3 is a deterministic density lane. Four adjacent chunks share
        # one scope only when their same-index fact partitions remain within a
        # complete instructional-unit budget. This is pagination independent:
        # no cross-request mutable counter can change scope identity.
        scope_group = chunk_no // SOURCE_SCOPE_CHUNKS_PER_GROUP
        if source_ref:
            scope_owner = f"source-ref:{source_ref}"
        else:
            scope_owner = "title:" + hashlib.sha256(scope_title.encode("utf-8")).hexdigest()[:24]
        candidates: list[SourceSnapshotFactV2] = []
        fact_index = 0
        try:
            fact_partitions = partition_chunk_facts_for_density(fact_texts)
        except ValueError as error:
            raise _orchestration_v2_http_error(
                "SOURCE_FACT_DENSITY_INVALID",
                "One canonical source fact exceeds the instructional density contract.",
            ) from error
        for lane_index, partition in enumerate(fact_partitions):
            scope_locator = f"{scope_owner}:chunk-group:{scope_group}:lane:{lane_index}"
            scope_key = "scope3_" + hashlib.sha256(
                f"{document_id}\x1e{scope_locator}".encode("utf-8"),
            ).hexdigest()[:32]
            # Chunk metadata describes every structure found anywhere in the
            # chunk. Density partitioning creates independent scopes, so
            # copying a chunk-level table marker onto a prose-only lane makes
            # downstream contracts require a table that cannot be faithfully
            # reconstructed from that lane. Keep only structure represented by
            # the canonical facts in this partition.
            partition_table_row_count = sum(
                bool(re.match(r"^Row\s+\d+\s*:\s*.+\|.+$", text, re.IGNORECASE))
                for text in partition
            )
            partition_has_table = partition_table_row_count >= 2
            partition_content_kinds = set(content_kinds)
            if partition_has_table:
                partition_content_kinds.add("table")
            else:
                partition_content_kinds.discard("table")
            partition_table_count = max(1, table_count) if partition_has_table else 0
            for text in partition:
                fact_index += 1
                candidates.append(SourceSnapshotFactV2(
                    document_id=document_id,
                    fact_key=f"d{document_order[document_id]}-c{chunk_no + 1}-f{fact_index}",
                    scope_key=scope_key,
                    fact_text=text,
                    source_ref=source_ref,
                    source_page=page,
                    source_chunk=chunk_no,
                    locator={
                        "index_id": str(row["index_id"]),
                        "source_revision": source_revision,
                        "scope_title": scope_title,
                        "parser_version": str(metadata.get("parser_version") or "")[:80] or None,
                        "content_kinds": sorted(partition_content_kinds)[:8],
                        "table_count": partition_table_count,
                        "source_evidence_revision": (
                            str(metadata.get("source_evidence_revision") or "").strip().casefold()
                            if re.fullmatch(
                                r"[0-9a-f]{64}",
                                str(metadata.get("source_evidence_revision") or "").strip().casefold(),
                            )
                            else None
                        ),
                        "source_evidence_status": evidence_status_by_document[document_id],
                        "structured_evidence_contract_version": str(
                            metadata.get("structured_evidence_contract_version") or ""
                        )[:80] or None,
                        "visual_regions": visual_regions,
                        "visual_prompt_text": visual_prompt_text,
                        "instructional_density_policy_version": INSTRUCTIONAL_DENSITY_POLICY_VERSION,
                        "scope_partition_lane": lane_index,
                    },
                ))
        candidate_bytes = sum(len(fact.fact_text.encode("utf-8")) for fact in candidates)
        exceeds = (len(facts) + len(candidates) > request.page_max_facts
                   or content_bytes + candidate_bytes > request.page_max_bytes)
        if exceeds and facts:
            has_more = True
            break
        if exceeds:
            raise _orchestration_v2_http_error(
                "SOURCE_CHUNK_EXCEEDS_PAGE", "One source chunk exceeds the bounded page contract.",
            )
        facts.extend(candidates)
        content_bytes += candidate_bytes
        last_cursor = {"document_id": document_id, "chunk_no": chunk_no}

    if has_more and last_cursor is None:
        raise _orchestration_v2_http_error(
            "SOURCE_CURSOR_STALLED", "The source cursor could not advance.",
        )
    facts_wire = [fact.model_dump(mode="json") for fact in facts]
    logger.info("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "source_snapshot_page_ready", "correlation_id": request.correlation_id,
        "source_snapshot_hash": request.source_snapshot_hash, "source_revision": source_revision,
        "source_fact_count": len(facts_wire), "source_content_bytes": content_bytes,
        "source_evidence_ready_document_count": len(document_ids) - legacy_document_count,
        "source_evidence_legacy_document_count": legacy_document_count,
        "has_more": has_more, "provider_call_count": 0,
    }, sort_keys=True))
    return {
        "contract_version": 2,
        "source_snapshot_hash": request.source_snapshot_hash,
        "source_revision": source_revision,
        "source_authority": source_authority.model_dump(mode="json"),
        "facts": facts_wire,
        "next_cursor": last_cursor if has_more else None,
        "has_more": has_more,
        "page_content_bytes": content_bytes,
        "usage": AiUsage().model_dump(),
    }


async def _idm_generate(api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, AiUsage]:
    """Provider transport for ``app.idm``: map service errors onto ``IdmProviderError``.

    An exhausted key or a rate limit that outlived the bounded wait is terminal for the
    stage (``RUN_STOPPING_PROVIDER_CODES``): it reaches Node as a 503 with that code
    instead of turning into deterministic course content.
    """

    try:
        return await provider.generate_content(api_key, model, prompt, **options)
    except HTTPException as error:
        code = str((error.detail if isinstance(error.detail, dict) else {}).get("code") or "AI_PROVIDER_UNAVAILABLE")
        if code in RUN_STOPPING_PROVIDER_CODES:
            raise IdmProviderError(code, terminal=True, http_status=503) from error
        raise IdmProviderError(code, terminal=code not in TRANSIENT_PROVIDER_CODES,
                               http_status=error.status_code) from error
    except AppError as error:
        raise IdmProviderError(error.code, terminal=False, http_status=error.http_status) from error
    except Exception as error:
        status = provider_http_error_status(error)
        logger.warning("lesson_author_idm %s", json.dumps({
            "event": "provider_request_failed", "model": model, **safe_provider_error_diagnostics(error),
        }, sort_keys=True))
        if status == 400:
            raise IdmProviderError("AI_PROVIDER_REQUEST_REJECTED", terminal=True) from error
        if status in {401, 403}:
            raise IdmProviderError("AI_PROVIDER_AUTH_REJECTED", terminal=True) from error
        raise IdmProviderError("AI_PROVIDER_UNAVAILABLE", terminal=False) from error


def _idm_runtime(request: RagChatRequest, *, budget_ms: int, allowance: Any | None) -> IdmRuntime:
    return IdmRuntime(
        generate=_idm_generate, api_key=require_provider_api_key(request.api_key), model=request.model,
        locale=request.locale, deadline=monotonic() + budget_ms / 1000, correlation_id=request.correlation_id,
        token_allowance=(IdmTokenAllowance(allowance.input_tokens, allowance.output_tokens)
                         if allowance is not None else None),
        provider_call_timeout_ms=settings.idm_provider_call_timeout_ms,
        rate_limit_max_wait_ms=settings.provider_rate_limit_max_wait_ms,
    )


def _idm_http_error(error: IdmError) -> HTTPException:
    if isinstance(error, IdmProviderError):
        return _orchestration_v2_http_error(error.code, "AI provider failed for the IDM stage.",
                                            status_code=error.http_status)
    return _orchestration_v2_http_error(error.code, "The IDM stage cannot be completed for this source.")


@app.post("/v1/lesson-author/orchestration-v2/course-skeleton", dependencies=[Depends(require_internal_token)])
async def lesson_author_orchestration_v2_course_skeleton(
    request: RagLessonAuthorCourseSkeletonV2Request,
) -> dict[str, Any]:
    if request.idm is not None:
        try:
            result = await run_idm_course_design(
                request.idm, source_snapshot_hash=request.source_snapshot_hash,
                runtime=_idm_runtime(request, budget_ms=request.idm.remaining_budget_ms,
                                     allowance=request.idm.token_allowance),
                parallelism=settings.idm_w1_parallelism, max_sections=settings.idm_max_sections,
            )
        except IdmError as error:
            raise _idm_http_error(error) from error
        except ValidationError as error:
            raise _idm_http_error(IdmStageError("IDM_STAGE_OUTPUT_INVALID")) from error
    else:
        result = await _lesson_author_orchestration_v2_course_skeleton(request=request)
    record_fallback("course_skeleton", result)
    return result


async def _lesson_author_orchestration_v2_course_skeleton(
    request: RagLessonAuthorCourseSkeletonV2Request,
) -> dict[str, Any]:
    """Generate only global structure; chapter content is delegated to bounded shards."""

    prompt = skeleton_prompt_v2(request.locale, request.scope_catalog, request.source_authority)
    logger.info("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "provider_schema_projection_ready", "correlation_id": request.correlation_id,
        "generation_stage": "course_skeleton", **COURSE_SKELETON_PROVIDER_SCHEMA_METADATA,
    }, sort_keys=True))
    usage = AiUsage()
    last_code = "ARCHITECTURE_SKELETON_INVALID"
    provider_failure_code: str | None = None
    for attempt in range(1, request.max_attempts + 1):
        try:
            text, attempt_usage = await _orchestration_v2_generate_content(
                request.api_key, request.model, prompt,
                max_output_tokens=request.max_output_tokens,
                response_schema=CourseSkeletonProviderWireV2,
                correlation_id=request.correlation_id,
                generation_stage="course_skeleton",
            )
        except HTTPException as error:
            detail = error.detail if isinstance(error.detail, dict) else {}
            code = detail.get("code")
            if code not in ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES:
                raise
            provider_failure_code = str(code)
            last_code = provider_failure_code
            break
        except Exception as error:
            # The Node worker persists its dispatch fence before this HTTP
            # request. Unknown SDK/transport failures must therefore settle
            # conservatively instead of turning a valid source fallback into a
            # blank 5xx response.
            provider_failure_code = "AI_PROVIDER_UNAVAILABLE"
            last_code = provider_failure_code
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_provider_fallback",
                "correlation_id": request.correlation_id,
                "failure_code": provider_failure_code,
                "exception_type": type(error).__name__,
            }, sort_keys=True))
            break
        usage = combine_usage(usage, attempt_usage)
        try:
            draft = CourseSkeletonDraftV2.model_validate_json(text)
            skeleton = bind_course_skeleton_v2(
                draft,
                source_snapshot_hash=request.source_snapshot_hash,
                locale=request.locale,
                scope_catalog=request.scope_catalog,
                source_authority=request.source_authority,
            )
            logger.info("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_ready", "correlation_id": request.correlation_id,
                "source_snapshot_hash": request.source_snapshot_hash,
                "scope_count": len(request.scope_catalog), "chapter_count": len(skeleton.chapters),
                "provider_attempt": attempt,
            }, sort_keys=True))
            return {"contract_version": 2, "skeleton": skeleton.model_dump(mode="json"),
                    "usage": usage.model_dump(), "usage_complete": True,
                    "usage_source": "provider", "content_origin": "provider_validated",
                    "quality_state": "validated"}
        except OrchestrationContractError as error:
            last_code = error.code
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_validation_failed",
                "correlation_id": request.correlation_id,
                "provider_attempt": attempt,
                "failure_code": last_code,
            }, sort_keys=True))
        except ValidationError as error:
            last_code = "ARCHITECTURE_SKELETON_INVALID"
            safe_errors = [{"type": item.get("type"), "loc": list(item.get("loc") or ())}
                           for item in error.errors(include_url=False, include_input=False)[:10]]
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_validation_failed",
                "correlation_id": request.correlation_id,
                "provider_attempt": attempt,
                "failure_code": last_code,
                "validation_errors": safe_errors,
            }, sort_keys=True))
    fallback = fallback_course_skeleton_draft_v2(
        request.locale, request.scope_catalog, request.source_authority,
    )
    skeleton = bind_course_skeleton_v2(
        fallback,
        source_snapshot_hash=request.source_snapshot_hash,
        locale=request.locale,
        scope_catalog=request.scope_catalog,
        source_authority=request.source_authority,
    )
    logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "course_skeleton_deterministic_fallback_ready",
        "correlation_id": request.correlation_id,
        "source_snapshot_hash": request.source_snapshot_hash,
        "scope_count": len(request.scope_catalog),
        "chapter_count": len(skeleton.chapters),
        "provider_attempts": request.max_attempts,
        "last_failure_code": last_code,
    }, sort_keys=True))
    return {"contract_version": 2, "skeleton": skeleton.model_dump(mode="json"),
            "usage": usage.model_dump(), "usage_complete": provider_failure_code is None,
            "usage_source": "reserved_upper_bound" if provider_failure_code else "provider",
            "content_origin": "structured_fallback", "quality_state": "review_required"}


@app.post("/v1/lesson-author/orchestration-v2/chapter-shard", dependencies=[Depends(require_internal_token)])
async def lesson_author_orchestration_v2_chapter_shard(
    request: RagLessonAuthorChapterShardV2Request,
) -> dict[str, Any]:
    context = request.idm_module_context
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
        code = error.code if isinstance(error, OrchestrationContractError) else "ARCHITECTURE_SHARD_IDENTITY_MISMATCH"
        raise _orchestration_v2_http_error(code, "The chapter shard context is invalid.") from error
    logger.info("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "provider_schema_projection_ready", "correlation_id": request.correlation_id,
        "generation_stage": "chapter_shard", "chapter_key": request.shard_plan.chapter_key,
        "shard_index": request.shard_plan.shard_index, **CHAPTER_SHARD_PROVIDER_SCHEMA_METADATA,
    }, sort_keys=True))
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
        except HTTPException as error:
            detail = error.detail if isinstance(error.detail, dict) else {}
            code = detail.get("code")
            if code not in ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES:
                raise
            provider_failure_code = str(code)
            last_code = provider_failure_code
            break
        except Exception as error:
            provider_failure_code = "AI_PROVIDER_UNAVAILABLE"
            last_code = provider_failure_code
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "chapter_shard_provider_fallback",
                "correlation_id": request.correlation_id,
                "chapter_key": request.shard_plan.chapter_key,
                "shard_index": request.shard_plan.shard_index,
                "failure_code": provider_failure_code,
                "exception_type": type(error).__name__,
            }, sort_keys=True))
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
            logger.info("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "chapter_shard_ready", "correlation_id": request.correlation_id,
                "source_snapshot_hash": request.skeleton.source_snapshot_hash,
                "chapter_key": request.shard_plan.chapter_key,
                "shard_index": request.shard_plan.shard_index,
                "shard_count": request.shard_plan.shard_count,
                "source_fact_count": len(request.source_facts), "provider_attempt": attempt,
                "component_compiler": compiler_diagnostics[:24],
                "omitted_component_compiler_unit_count": max(0, len(compiler_diagnostics) - 24),
            }, sort_keys=True))
            return {"contract_version": 2, "shard": shard.model_dump(mode="json"),
                    "usage": usage.model_dump(), "usage_complete": True,
                    "usage_source": "provider", "content_origin": "provider_validated",
                    "quality_state": "validated"}
        except OrchestrationContractError as error:
            last_code = error.code
            safe_errors = [{"type": "contract_error", "loc": [], "code": error.code}]
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            last_code = "ARCHITECTURE_SHARD_INVALID"
            safe_errors = ([{"type": item.get("type"), "loc": list(item.get("loc") or ())}
                            for item in error.errors(include_url=False, include_input=False)[:12]]
                           if isinstance(error, ValidationError)
                           else [{"type": type(error).__name__, "loc": []}])
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "chapter_shard_validation_failed", "correlation_id": request.correlation_id,
            "chapter_key": request.shard_plan.chapter_key, "shard_index": request.shard_plan.shard_index,
            "provider_attempt": attempt, "failure_code": last_code, "validation_errors": safe_errors,
        }, sort_keys=True))
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
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "chapter_shard_partial_salvage_rejected",
                "correlation_id": request.correlation_id,
                "chapter_key": request.shard_plan.chapter_key,
                "shard_index": request.shard_plan.shard_index,
                "failure_code": error.code if isinstance(error, OrchestrationContractError) else type(error).__name__,
            }, sort_keys=True))
            best_salvage = None
    if best_salvage is not None:
        shard, diagnostics = best_salvage
        component_decisions = diagnostics.pop("component_decisions", [])
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
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
        }, sort_keys=True))
        return {"contract_version": 2, "shard": shard.model_dump(mode="json"),
                "usage": usage.model_dump(), "usage_complete": provider_failure_code is None,
                "usage_source": "reserved_upper_bound" if provider_failure_code else "provider",
                "content_origin": "structured_fallback", "quality_state": "review_required"}

    fallback = fallback_chapter_shard_draft_v2(
        request.skeleton, request.shard_plan, request.source_facts,
    )
    compiler_diagnostics: list[dict[str, Any]] = []
    shard = bind_chapter_shard_v2(
        fallback,
        skeleton=request.skeleton,
        plan=request.shard_plan,
        source_facts=request.source_facts,
        compiler_diagnostics=compiler_diagnostics,
    )
    logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "chapter_shard_deterministic_fallback_ready",
        "correlation_id": request.correlation_id,
        "chapter_key": request.shard_plan.chapter_key,
        "shard_index": request.shard_plan.shard_index,
        "source_fact_count": len(request.source_facts),
        "provider_attempts": request.max_attempts,
        "last_failure_code": last_code,
        "component_compiler": compiler_diagnostics[:24],
        "omitted_component_compiler_unit_count": max(0, len(compiler_diagnostics) - 24),
    }, sort_keys=True))
    return {"contract_version": 2, "shard": shard.model_dump(mode="json"),
            "usage": usage.model_dump(), "usage_complete": provider_failure_code is None,
            "usage_source": "reserved_upper_bound" if provider_failure_code else "provider",
            "content_origin": "structured_fallback", "quality_state": "review_required"}


@app.post("/v1/lesson-author/orchestration-v2/unit", dependencies=[Depends(require_internal_token)])
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


@app.post("/v1/lesson-author/blueprint", dependencies=[Depends(require_internal_token)])
async def lesson_author_blueprint(
    request: RagLessonAuthorBlueprintRequest,
    pool: asyncpg.Pool = Depends(get_db),
) -> dict[str, Any]:
    return await run_with_deadline(
        "/v1/lesson-author/blueprint",
        settings.lesson_author_deadline_ms,
        _lesson_author_blueprint(request, pool),
    )


async def _lesson_author_blueprint(
    request: RagLessonAuthorBlueprintRequest,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    workflow_started = perf_counter()

    def emit_blueprint_diagnostic(metadata: dict[str, Any]) -> None:
        """Emit only structured operational metadata for the correlated run."""

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "course_architecture",
            "workflow_version": "langgraph-v1",
            "architecture_contract_version": 5,
            "repair_provider_call_cap": V5_MAX_PROVIDER_REPAIR_CALLS,
            "semantic_repair_contract_version": SEMANTIC_REPAIR_CONTRACT_VERSION,
            "repair_attempt_cap_per_layer": V5_MAX_REPAIR_ATTEMPTS_PER_LAYER,
            "correlation_id": request.correlation_id,
            "tenant_id": request.tenant_id,
            "kb_id": request.kb_id,
            "conversation_id": request.conversation_id,
            "course_id": request.course_id,
            "source_document_ids": [document.document_id for document in request.source_documents],
            **metadata,
        }
        logger.info("lesson_author_blueprint_diagnostic %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))

    emit_blueprint_diagnostic({
        "stage": "lesson_author_request",
        "event": "received",
        "repair_pass_number": 0,
    })
    rows, retrieval_usage, structure_context = await retrieval_search.retrieve_chunks(pool, request)
    emit_blueprint_diagnostic({
        "stage": "rag_retrieval",
        "event": "completed",
        "repair_pass_number": 0,
        "retrieved_chunk_count": len(rows),
        "source_scope_truncated": bool(structure_context.get("course_blueprint_source_scope_truncated")),
    })
    context, sources = format_sources(rows, max_context_chars=retrieval_limits(request)["max_context_chars"])
    retrieval = build_retrieval_diagnostics(request, rows, sources, structure_context)
    source_coverage_manifest = structure_context.get("source_coverage_manifest")
    source_coverage = format_course_architecture_coverage_contract(source_coverage_manifest)
    runtime: dict[str, Any] = {}

    def validate_source_scope() -> list[WorkflowIssue]:
        if structure_context.get("course_blueprint_source_scope_truncated"):
            return [_workflow_issue(
                "SOURCE_SCOPE_INCOMPLETE",
                "The complete source scope is unavailable for global course architecture.",
            )]
        policy = structure_context.get("source_chapter_policy")
        if policy is not None:
            represented_documents = {str(node.get("document_id")) for node in structure_context.get("source_structure_nodes", [])}
            missing_documents = {doc.document_id for doc in request.source_documents} - represented_documents
            emit_blueprint_diagnostic({
                "stage": "source_chapter_policy", "event": "resolved",
                "mode": policy["mode"], "complete": policy["complete"],
                "expected_chapter_count": len(policy["chapters"]),
                "reason_codes": policy["reason_codes"],
                "missing_document_count": len(missing_documents),
                "source_structure_node_count": structure_context.get("structure_node_count", 0),
                "source_outline_display_truncated": bool(structure_context.get("source_outline_display_truncated")),
                "source_outline_display_chars": len(structure_context.get("outline") or ""),
            })
            if not policy["complete"] or missing_documents:
                raise WorkflowFailure(
                    "SOURCE_STRUCTURE_REVIEW_REQUIRED", "Source chapter authority is incomplete or ambiguous.",
                    internal_code="SOURCE_STRUCTURE_REVIEW_REQUIRED", failure_stage="source_chapter_policy",
                    diagnostics={"reason_codes": policy["reason_codes"], "missing_document_count": len(missing_documents)},
                )
        return []

    def build_global_source_map() -> dict[str, Any]:
        source_map = build_source_map(
            structure_context.get("source_structure_nodes", []),
            source_coverage_manifest,
            locale=request.locale,
        )
        architect_context = build_course_architect_context(
            source_map,
            source_coverage_manifest,
            max_chars=max(1, settings.source_map_architect_context_max_chars),
        )
        coverage = source_map.get("coverage") if isinstance(source_map.get("coverage"), dict) else {}
        if not coverage.get("section_scope_complete"):
            raise WorkflowFailure(
                "SOURCE_MAP_SCOPE_INCOMPLETE",
                "The global Source Map does not represent the complete selected source scope.",
            )
        if not coverage.get("fact_scope_complete"):
            incomplete_reason = str(coverage.get("incomplete_reason") or "SOURCE_MAP_SCOPE_INCOMPLETE")
            raise WorkflowFailure(
                incomplete_reason if incomplete_reason == "SOURCE_FACT_EXTRACTION_CAPACITY_EXCEEDED" else "SOURCE_MAP_SCOPE_INCOMPLETE",
                "The global Source Map does not represent every canonical source fact.",
            )
        if not architect_context.get("context_complete"):
            raise WorkflowFailure(
                str(architect_context.get("error_code") or "ARCHITECT_CONTEXT_CANNOT_REPRESENT_SOURCE"),
                "The global Source Map hierarchy cannot fit within the configured architect context budget.",
            )
        v5_source_context = create_v5_immutable_source_context(
            source_map,
            source_coverage_manifest,
        )
        runtime["v5_source_context"] = v5_source_context
        runtime["source_map_context"] = architect_context["context"]
        # Never give a graph node the immutable object itself. The graph gets
        # a disposable copy while validation/allocation always return to the
        # request-scoped source authority below.
        runtime["source_map"] = v5_source_context.source_map_copy()
        runtime["source_map_diagnostics"] = architect_context["diagnostics"]
        emit_blueprint_diagnostic({
            "stage": "source_map_build",
            "event": "completed",
            "repair_pass_number": 0,
            "canonical_fact_count": int(coverage.get("total_fact_count") or 0),
            "represented_fact_count": int(coverage.get("represented_fact_count") or 0),
            "source_section_count": int(coverage.get("section_count") or 0),
            "source_concept_count": int(coverage.get("concept_count") or 0),
            "fact_scope_complete": bool(coverage.get("fact_scope_complete")),
            "source_map_complete": bool(coverage.get("fact_scope_complete")) and bool(coverage.get("section_scope_complete")),
            "architect_context_mode": architect_context["diagnostics"].get("architect_context_mode"),
            "architect_context_chars": architect_context["diagnostics"].get("architect_context_size"),
            "architect_hierarchy_encoding": architect_context["diagnostics"].get("architect_hierarchy_encoding"),
            "architect_uncompressed_context_chars": architect_context["diagnostics"].get("architect_uncompressed_context_chars"),
            "architect_detail_fact_count": architect_context["diagnostics"].get("architect_detail_fact_count"),
            "evidence_scope_count": v5_source_context.evidence_scope_count,
            "source_context_fingerprint": v5_source_context.fingerprint,
        })
        return v5_source_context.source_map_copy()

    async def architect_course(source_map: dict[str, Any]) -> WorkflowGenerationResult:
        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="course_architect_generation",
        )
        prompt = build_lesson_author_blueprint_prompt(
            request,
            context,
            structure_context.get("outline", ""),
            source_coverage,
            runtime["source_map_context"],
            structure_source=structure_context.get("structure_source"),
            authoritative_source_nodes=structure_context.get("authoritative_source_nodes"),
            source_chapter_policy=structure_context.get("source_chapter_policy"),
        )
        emit_blueprint_diagnostic({
            "stage": "course_architect_policy", "event": "composed",
            "policy_mode": "self_built_rag_v5",
            "policy_version": "v5-blueprint-policy-1" if request.system_prompt.startswith("SERVER POLICY VERSION: v5-blueprint-policy-1.") else "legacy_internal_caller",
            "policy_chars": len(request.system_prompt.strip()),
            "policy_sha256": hashlib.sha256(request.system_prompt.strip().encode("utf-8")).hexdigest(),
            "output_locale": request.locale,
        })
        try:
            blueprint, generation_usage = await blueprint_service.generate_validated_lesson_author_blueprint(
                request,
                prompt,
                set(structure_context.get("known_source_refs", set())),
                structure_source=structure_context.get("structure_source"),
                authoritative_source_nodes=structure_context.get("authoritative_source_nodes"),
                source_chapter_policy=structure_context.get("source_chapter_policy"),
                source_structure_nodes=structure_context.get("source_structure_nodes"),
                allow_server_fact_allocation=True,
                # V5 semantic findings are intentionally returned to the
                # graph as local patch targets. V3/V4 callers retain the
                # existing whole-candidate compatibility behavior.
                v5_immutable_source_context=v5_source_context,
                defer_semantic_scope_validation=True,
                emit_diagnostic=emit_blueprint_diagnostic,
            )
            if blueprint.get("architecture_contract_version") not in {3, 4, 5}:
                blueprint = ensure_blueprint_source_granularity(
                    blueprint,
                    source_coverage_manifest,
                    structure_context.get("source_structure_nodes"),
                    request.locale,
                )
            return WorkflowGenerationResult(blueprint, generation_usage.model_dump())
        except CourseArchitectSemanticScopeError as error:
            raise WorkflowFailure(
                "ARCHITECTURE_SCOPE_INCOMPLETE",
                "Course Architect output did not satisfy canonical semantic ownership.",
                issues=error.issues,
                internal_code="ARCH_SEMANTIC_SCOPE_ATTEMPTS_EXHAUSTED",
                failure_stage="course_architect_semantic_scope_validation",
                diagnostics={"architect_attempt_count": error.attempt_count},
            ) from error
        except LessonAuthorBlueprintGenerationError as error:
            raise WorkflowFailure(
                "PROVIDER_ERROR",
                error.reason or error.code,
                internal_code="ARCH_PROVIDER_OUTPUT_INVALID",
                failure_stage="course_architect_output_validation",
                diagnostics={"architect_attempt_count": request.max_attempts},
            ) from error
        except LessonAuthorBlueprintValidationError as error:
            raise WorkflowFailure(
                error.code,
                str(error),
                internal_code="ARCH_FACT_ALLOCATION_FAILED",
                failure_stage="canonical_fact_allocation",
            ) from error
        except HTTPException as error:
            raise WorkflowFailure(
                "PROVIDER_ERROR",
                "Course Architect provider call failed.",
                internal_code="AI_PROVIDER_TIMEOUT" if isinstance(error.detail, dict) and error.detail.get("code") == "AI_PROVIDER_TIMEOUT" else "ARCH_PROVIDER_ERROR",
                failure_stage="course_architect_provider",
                diagnostics={
                    "provider_http_status": None if isinstance(error.detail, dict) and error.detail.get("code") == "AI_PROVIDER_TIMEOUT" else error.status_code,
                    "application_http_status": error.status_code,
                },
            ) from error

    def validate_blueprint_layers(
        candidate: dict[str, Any],
        _workflow_source_map: dict[str, Any],
    ) -> WorkflowValidationResult:
        """Run V5 validation in repairable layers against immutable source state.

        Schema, semantic ownership and instructional coherence are evaluated
        before server allocation. This prevents a four-scope local omission
        from being inflated into hundreds of fact findings or a whole-course
        repair. Only a semantically complete candidate can be allocated.
        """

        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="blueprint_validation",
        )
        def emit_layer(event: str, issues: list[WorkflowIssue] | None = None) -> None:
            emit_blueprint_diagnostic({
                "stage": "v5_blueprint_layer_validation",
                "event": event,
                "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
                "architecture_contract_version": 5,
                "canonical_fact_count": v5_source_context.canonical_fact_count,
                "evidence_scope_count": v5_source_context.evidence_scope_count,
                "candidate_status": event,
                "validation_codes": sorted({
                    str(issue.get("code") or "")
                    for issue in (issues or [])
                    if issue.get("code")
                }),
            })

        def mark_repair_layer(
            result: WorkflowValidationResult,
            repair_layer: Literal[
                "SCHEMA",
                "EVIDENCE_SEMANTIC",
                "PRE_ALLOCATION_COHERENCE",
                "POST_ALLOCATION_INSTRUCTIONAL_DEPTH",
            ],
        ) -> WorkflowValidationResult:
            """Annotate deterministic V5 findings for the graph scheduler only."""

            for issue in result.issues:
                if issue.get("severity") == "error":
                    issue["repair_layer"] = repair_layer
            return result

        if candidate.get("architecture_contract_version") != 5:
            result = WorkflowValidationResult([{
                "code": "V5_SOURCE_CONTEXT_INVARIANT_FAILED",
                "severity": "error",
                "message": "The V5 Course Architect candidate did not retain the required V5 contract.",
                "path": "course",
                "repairable": False,
            }])
            emit_layer("contract_failed", result.issues)
            return result
        try:
            normalized = validate_lesson_author_blueprint(
                candidate,
                require_source_fact_ownership=False,
                forbid_provider_fact_ownership=True,
            )
            validate_lesson_author_source_refs(
                normalized,
                set(structure_context.get("known_source_refs", set())),
            )
            if structure_context.get("source_chapter_policy") is not None:
                normalized = bind_source_chapters(normalized, structure_context["source_chapter_policy"], v5_source_context.source_map_copy())
        except LessonAuthorBlueprintValidationError as error:
            issue = _workflow_issue_from_blueprint_validation_error(error)
            # A bounded lesson array violation (including the provider's four
            # units) is local. Root/chapter failures are globally unusable and
            # must not become an oversized repair request.
            issue["repairable"] = bool(re.fullmatch(
                r"chapters\[\d+\]\.lessons\[\d+\](?:\.units(?:\[\d+\])?)?",
                error.path or "",
            ))
            result = WorkflowValidationResult([issue])
            emit_layer("schema_failed", result.issues)
            return mark_repair_layer(result, "SCHEMA")
        except LessonAuthorProposalValidationError:
            result = WorkflowValidationResult([{
                "code": "INVALID_SOURCE_REF",
                "severity": "error",
                "message": "Blueprint source references are outside the approved source scope.",
                "path": "course",
                "repairable": False,
            }])
            emit_layer("source_reference_failed", result.issues)
            return result

        # The normalizer creates a provider-safe semantic candidate; any old
        # allocation must be absent at this stage and is re-created below.
        candidate.clear()
        candidate.update(normalized)

        # Provider output never grants capability; only the internal Node request does.
        if request.component_capabilities is not None:
            candidate["component_capabilities"] = request.component_capabilities.model_dump()

        semantic = validate_course_architecture_evidence_scope(
            candidate,
            v5_source_context.source_map_copy(),
        )
        if semantic.errors:
            emit_layer("semantic_scope_failed", semantic.issues)
            return mark_repair_layer(semantic, "EVIDENCE_SEMANTIC")
        emit_layer("evidence_semantic_passed", [])

        # Freeze direct assessment references from the same canonical scopes
        # already accepted above, before candidate fingerprints/mutation guards.
        # Explicit provider refs and all ownership remain protected unchanged.
        projected, provenance_diagnostics, provenance_issues = materialize_assessment_source_refs(
            candidate, v5_source_context.source_map_copy(),
        )
        emit_blueprint_diagnostic({
            "stage": "v5_assessment_provenance", "event": "failed" if provenance_issues else "resolved",
            "materialized_block_count": 0 if provenance_issues else len(provenance_diagnostics),
            "blocks": provenance_diagnostics[:24],
            "omitted_block_count": max(0, len(provenance_diagnostics) - 24),
            "reason_codes": sorted({issue.safe_reason for issue in provenance_issues}),
        })
        if provenance_issues:
            result = WorkflowValidationResult([issue.workflow_issue() for issue in provenance_issues])
            return mark_repair_layer(result, "PRE_ALLOCATION_COHERENCE")
        projected_semantic = validate_course_architecture_evidence_scope(projected, v5_source_context.source_map_copy())
        if projected_semantic.errors:
            emit_layer("semantic_scope_failed", projected_semantic.issues)
            return mark_repair_layer(projected_semantic, "EVIDENCE_SEMANTIC")
        candidate.clear()
        candidate.update(projected)

        # Assessment intent is compiled before coherence, one local objective
        # at a time. The Architect declares *that* assessment is required; the
        # compiler is the server authority that creates factless supporting
        # knowledge-check blocks from existing teaching provenance.  It never
        # assigns canonical facts or expands source scope.
        assessment_plan = compile_v5_assessment_plan(candidate)
        unit_diagnostics = assessment_plan.safe_unit_diagnostics(candidate)
        emit_blueprint_diagnostic({
            "stage": "v5_assessment_plan_compiler",
            "event": assessment_plan.status.lower(),
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
            "architecture_contract_version": 5,
            **assessment_plan.metrics,
            "units": unit_diagnostics[:24],
            "omitted_unit_count": max(0, len(unit_diagnostics) - 24),
            "objective_diagnostics": assessment_plan.objective_diagnostics,
            "objective_diagnostic_count": assessment_plan.objective_diagnostic_count,
            "omitted_objective_diagnostic_count": max(
                0, assessment_plan.objective_diagnostic_count - len(assessment_plan.objective_diagnostics),
            ),
        })
        if assessment_plan.status == "TERMINAL_GAP":
            result = WorkflowValidationResult(
                [issue.workflow_issue() for issue in assessment_plan.issues],
                assessment_plan.metrics,
            )
            emit_layer("assessment_plan_terminal_gap", result.issues)
            return mark_repair_layer(result, "PRE_ALLOCATION_COHERENCE")
        # The compiler is copy-on-write. Applying this safe candidate here
        # does not persist anything; final review persistence remains after
        # all evidence/coherence/allocation validation passes.
        candidate.clear()
        candidate.update(assessment_plan.blueprint)
        if assessment_plan.status == "NEEDS_SEMANTIC_RESOLUTION":
            result = WorkflowValidationResult(
                [issue.workflow_issue() for issue in assessment_plan.issues],
                assessment_plan.metrics,
            )
            emit_layer("assessment_plan_semantic_resolution_required", result.issues)
            return mark_repair_layer(result, "PRE_ALLOCATION_COHERENCE")

        treatment_candidate, treatment_diagnostics = compile_evidence_treatments(
            candidate, v5_source_context.source_map_copy(), v5_source_context.manifest_copy(),
        )
        # Both the existing evidence and coherence validators still decide
        # acceptance. Treatment discovery cannot grant ownership or bypass them.
        treatment_semantic = validate_course_architecture_evidence_scope(
            treatment_candidate, v5_source_context.source_map_copy(),
        )
        if treatment_semantic.errors:
            emit_layer("semantic_scope_failed", treatment_semantic.issues)
            return mark_repair_layer(treatment_semantic, "EVIDENCE_SEMANTIC")
        candidate.clear()
        candidate.update(treatment_candidate)
        emit_blueprint_diagnostic({
            "stage": "evidence_treatment_discovery", "event": "completed",
            "version": EVIDENCE_TREATMENT_VERSION,
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
            "units": treatment_diagnostics[:24],
            "omitted_unit_count": max(0, len(treatment_diagnostics) - 24),
        })

        coherence = validate_v5_instructional_coherence(candidate)
        if coherence.errors:
            emit_layer("pre_allocation_coherence_failed", coherence.issues)
            return mark_repair_layer(coherence, "PRE_ALLOCATION_COHERENCE")
        emit_layer("instructional_coherence_passed", [])

        allocated = allocate_source_map_architecture_facts(
            candidate,
            v5_source_context.source_map_copy(),
            v5_source_context.manifest_copy(),
        )
        allocation = allocated.get("source_fact_allocation")
        if (
            not isinstance(allocation, dict)
            or int(allocation.get("required_count") or -1) != v5_source_context.canonical_fact_count
        ):
            result = WorkflowValidationResult([{
                "code": "V5_SOURCE_CONTEXT_INVARIANT_FAILED",
                "severity": "error",
                "message": "V5 allocation did not retain the immutable canonical fact manifest.",
                "path": "course",
                "repairable": False,
            }])
            emit_layer("allocation_context_failed", result.issues)
            return result
        if allocation.get("complete"):
            allocated = allocate_blueprint_source_fact_ids(
                allocated,
                v5_source_context.manifest_copy(),
                structure_context.get("source_structure_nodes"),
            )
        candidate.clear()
        candidate.update(allocated)
        emit_blueprint_diagnostic({
            "stage": "canonical_fact_allocation",
            "event": "completed",
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
            "canonical_fact_count": v5_source_context.canonical_fact_count,
            "evidence_scope_count": v5_source_context.evidence_scope_count,
            **source_fact_allocation_diagnostics(candidate),
        })
        result = validate_course_architecture_workflow(
            candidate,
            v5_source_context.source_map_copy(),
            v5_source_context.manifest_copy(),
            set(structure_context.get("known_source_refs", set())),
        )
        depth_only = bool(result.errors) and all(
            str(issue.get("code") or "") == "INSTRUCTIONAL_DEPTH_INSUFFICIENT"
            for issue in result.errors
        )
        emit_layer(
            "final_validation_passed" if not result.errors
            else "post_allocation_instructional_depth_failed" if depth_only
            else "final_validation_failed",
            result.issues,
        )
        if depth_only:
            return mark_repair_layer(result, "POST_ALLOCATION_INSTRUCTIONAL_DEPTH")
        return result

    async def repair_course(
        blueprint: dict[str, Any],
        targets: list[RepairTarget],
        _source_map: dict[str, Any],
    ) -> WorkflowGenerationResult:
        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="architecture_repair_target_snapshot",
        )
        assert_v5_scoped_repair_target_bound(targets, v5_source_context)
        targets = _v5_prepare_semantic_repair_targets(
            blueprint,
            targets,
            v5_source_context.source_map_copy(),
        )
        deterministic_targets = [
            target for target in targets
            if target.get("deterministic_semantic_delta") is True
        ]
        provider_targets = [
            target for target in targets
            if target.get("deterministic_semantic_delta") is not True
        ]
        working_blueprint = blueprint
        deterministic_payload = _v5_deterministic_assessment_alignment_payload(deterministic_targets)
        if deterministic_payload is not None:
            working_blueprint = apply_course_architecture_repair_patches(
                working_blueprint,
                deterministic_targets,
                deterministic_payload,
            )
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_deterministic",
                "event": "completed",
                "repair_pass_number": int(runtime.get("repair_provider_pass") or 0) + 1,
                "repair_target_count": len(deterministic_targets),
                "provider_called": False,
                "semantic_operation": "repair_assessment_alignment",
                "deterministic_reason": "EXACT_BASE_ELIGIBLE_TEACHING_ANCHOR",
            })
        if not provider_targets:
            return WorkflowGenerationResult(working_blueprint)
        blueprint = working_blueprint
        targets = provider_targets
        repair_layers = {
            str(target.get("repair_layer") or "")
            for target in targets
            if isinstance(target, dict)
        }
        repair_layer = next(iter(repair_layers)) if len(repair_layers) == 1 else "UNCLASSIFIED"
        semantic_delta_operations = _v5_semantic_delta_operations_for_targets(targets)
        layer_attempt_number = max(
            (int(target.get("layer_attempt_number") or 0) for target in targets),
            default=0,
        )
        total_repair_provider_calls = max(
            (int(target.get("total_repair_provider_calls") or 0) for target in targets),
            default=0,
        )
        if semantic_delta_operations:
            operation_groups: list[tuple[str | None, list[RepairTarget]]] = []
            for operation in sorted(semantic_delta_operations):
                group = [
                    target for target in targets
                    if list(target.get("semantic_operations") or []) == [operation]
                ]
                if group:
                    operation_groups.append((operation, group))
            if sum(len(group) for _operation, group in operation_groups) != len(targets):
                raise WorkflowFailure(
                    "ARCHITECTURE_REPAIR_INVALID",
                    "A V5 coherence target did not map to one exact semantic-delta operation.",
                    internal_code="ARCH_REPAIR_SEMANTIC_OPERATION_CONFLICT",
                    failure_stage="architecture_repair_target_snapshot",
                    diagnostics={"repair_target_count": len(targets)},
                )
        else:
            operation_groups = [(None, targets)]
        repair_pass_number = int(runtime.get("repair_provider_pass") or 0) + 1
        runtime["repair_provider_pass"] = repair_pass_number
        repair_output_tokens = min(request.max_output_tokens, 16_384)
        scoped_contexts: list[tuple[str | None, list[RepairTarget], str, str]] = []
        for operation, operation_targets in operation_groups:
            scoped_repair_context = build_v5_scoped_repair_source_context(
                blueprint=blueprint,
                targets=operation_targets,
                source_map=v5_source_context.source_map_copy(),
            )
            scoped_contexts.append((
                operation,
                operation_targets,
                scoped_repair_context,
                build_course_architecture_repair_prompt(
                    blueprint=blueprint,
                    targets=operation_targets,
                    source_map_context=scoped_repair_context,
                    locale=request.locale,
                ),
            ))
        emit_blueprint_diagnostic({
            "stage": "architecture_repair_target_snapshot",
            "event": "completed",
            "repair_pass_number": repair_pass_number,
            "repair_target_count": len(targets),
            "repair_scope": sorted({target["scope"] for target in targets}),
            **({"semantic_delta_operations": sorted(semantic_delta_operations)} if semantic_delta_operations else {}),
            "repair_layer": repair_layer,
            "layer_attempt_number": layer_attempt_number,
            "total_repair_provider_calls": total_repair_provider_calls,
            "serialized_target_chars": sum(len(context) for _operation, _targets, context, _prompt in scoped_contexts),
            "requested_output_tokens": repair_output_tokens,
            "canonical_fact_count": v5_source_context.canonical_fact_count,
            "evidence_scope_count": v5_source_context.evidence_scope_count,
            "provider_operation_count": len(scoped_contexts),
        })
        repair_usages: list[AiUsage] = []
        all_patches: list[dict[str, Any]] = []
        executed_provider_call_count = 0
        # Targets carry the actual count BEFORE dispatch, not a reservation.
        prior_repair_provider_calls = total_repair_provider_calls
        try:
            for operation_index, (operation, operation_targets, _context, prompt) in enumerate(scoped_contexts, start=1):
                repair_provider_telemetry: dict[str, Any] = {}
                selection_contract = None
                if operation == "select_assessment_teaching_alignment":
                    selection_contract = build_assessment_selection_contract(
                        operation_targets, max_context_chars=MAX_WORKFLOW_REPAIR_TARGET_CHARS,
                        blueprint=blueprint, source_map=v5_source_context.source_map_copy(),
                        manifest=v5_source_context.manifest_copy(),
                    )
                    prompt = selection_contract.prompt(request.locale)
                    emit_blueprint_diagnostic({
                        "stage": "architecture_repair_selection_contract", "event": "prepared",
                        "repair_pass_number": repair_pass_number, "repair_layer": repair_layer,
                        "semantic_operation": operation, "selection_contract_version": ASSESSMENT_SELECTION_CONTRACT_VERSION,
                        "selection_slot_count": len(selection_contract.bindings), "repair_target_count": len(operation_targets),
                        **selection_contract.diagnostics,
                        "total_repair_provider_calls": total_repair_provider_calls,
                    })

                def record_repair_provider_telemetry(telemetry: dict[str, Any]) -> None:
                    repair_provider_telemetry.update(telemetry)
                    emit_blueprint_diagnostic({
                        "stage": "architecture_repair_provider",
                        "event": "completed",
                        "repair_pass_number": repair_pass_number,
                        "repair_layer": repair_layer,
                        "layer_attempt_number": layer_attempt_number,
                        "total_repair_provider_calls": prior_repair_provider_calls + operation_index,
                        "provider_operation_index": operation_index,
                        "provider_operation_count": len(scoped_contexts),
                        **({"semantic_operation": operation} if operation else {}),
                        **telemetry,
                    })

                try:
                    # Count the provider boundary before awaiting it: an HTTP
                    # failure, parser failure or mutation-guard rejection must
                    # still be visible as an executed call in terminal state.
                    executed_provider_call_count = operation_index
                    total_repair_provider_calls = prior_repair_provider_calls + executed_provider_call_count
                    text, usage = await provider.generate_content(
                        request.api_key,
                        request.model,
                        prompt,
                        max_output_tokens=repair_output_tokens,
                        json_mode=True,
                        response_schema=(
                            selection_contract.schema if selection_contract else build_v5_semantic_delta_repair_response_schema({operation})
                            if operation else build_course_architecture_repair_response_schema(
                                set().union(*(
                                    set(target.get("allowed_fields", []))
                                    for target in operation_targets
                                ))
                            )
                        ),
                        thinking_config=types.ThinkingConfig(include_thoughts=False),
                        request_timeout_ms=settings.blueprint_provider_request_timeout_ms,
                        on_provider_telemetry=record_repair_provider_telemetry,
                    )
                except HTTPException as error:
                    emit_blueprint_diagnostic({
                        "stage": "architecture_repair_provider",
                        "event": "failed",
                        "repair_pass_number": repair_pass_number,
                        "repair_layer": repair_layer,
                        "layer_attempt_number": layer_attempt_number,
                        "total_repair_provider_calls": prior_repair_provider_calls + operation_index,
                        "provider_operation_index": operation_index,
                        "provider_operation_count": len(scoped_contexts),
                        **({"semantic_operation": operation} if operation else {}),
                        "provider_http_status": error.status_code,
                        "provider_finish_reason": None,
                        "provider_finish_reason_available": False,
                        "usage_source": "unavailable",
                    })
                    raise WorkflowFailure(
                        "ARCHITECTURE_REPAIR_INVALID",
                        "Course architecture repair provider call failed.",
                        internal_code="ARCH_REPAIR_PROVIDER_ERROR",
                        failure_stage="architecture_repair_provider",
                        diagnostics={"provider_http_status": error.status_code, "semantic_operation": operation},
                    ) from error
                finish_reason = str(repair_provider_telemetry.get("provider_finish_reason") or "").upper()
                if finish_reason.endswith("MAX_TOKENS"):
                    raise WorkflowFailure(
                        "ARCHITECTURE_REPAIR_INVALID",
                        "Course architecture repair was truncated by the provider.",
                        internal_code="ARCH_REPAIR_PROVIDER_TRUNCATED",
                        failure_stage="architecture_repair_provider",
                        diagnostics={
                            "provider_finish_reason": repair_provider_telemetry.get("provider_finish_reason"),
                            "response_chars": repair_provider_telemetry.get("response_chars"),
                            "response_bytes": repair_provider_telemetry.get("response_bytes"),
                            "semantic_operation": operation,
                        },
                    )
                payload = selection_contract.decode_text(text) if selection_contract else parse_course_architecture_repair_payload(text)
                emit_blueprint_diagnostic({
                    "stage": "architecture_repair_json_parser",
                    "event": "passed",
                    "repair_pass_number": repair_pass_number,
                    "repair_layer": repair_layer,
                    "layer_attempt_number": layer_attempt_number,
                    "total_repair_provider_calls": prior_repair_provider_calls + operation_index,
                    "provider_operation_index": operation_index,
                    "provider_operation_count": len(scoped_contexts),
                    **({"semantic_operation": operation} if operation else {}),
                    "response_chars": len(text),
                })
                all_patches.extend(
                    patch for patch in payload.get("patches", []) if isinstance(patch, dict)
                )
                repair_usages.append(usage)

            action_intent_metadata = _v5_safe_action_intent_repair_metadata(
                blueprint,
                targets,
                all_patches,
            )
            repaired = apply_course_architecture_repair_patches(blueprint, targets, {"patches": all_patches})
            assert_v5_immutable_source_context(
                v5_source_context,
                stage="architecture_repair_patch_apply",
            )
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_patch_apply",
                "event": "completed",
                "repair_pass_number": repair_pass_number,
                "repair_layer": repair_layer,
                "layer_attempt_number": layer_attempt_number,
                "total_repair_provider_calls": total_repair_provider_calls,
                "patch_count": len(all_patches),
                "accepted_patch_count": len(targets),
                **action_intent_metadata,
            })
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_candidate_validation",
                "event": "deferred_to_layered_validation",
                "repair_pass_number": repair_pass_number,
                "repair_layer": repair_layer,
                "layer_attempt_number": layer_attempt_number,
                "total_repair_provider_calls": total_repair_provider_calls,
                "repair_target_count": len(targets),
                "repair_validation_result": "PENDING_SCHEMA_SEMANTIC_COHERENCE_ALLOCATION",
            })
        except WorkflowFailure as error:
            # Preserve only operational call accounting. The exception and
            # logs intentionally never retain a provider patch or source text.
            error.diagnostics["executed_provider_call_count"] = executed_provider_call_count
            if error.failure_stage == "architecture_repair_semantic_delta_guard":
                emit_blueprint_diagnostic({
                    "stage": error.failure_stage,
                    "event": "rejected",
                    "repair_pass_number": repair_pass_number,
                    "repair_layer": repair_layer,
                    "layer_attempt_number": layer_attempt_number,
                    "total_repair_provider_calls": prior_repair_provider_calls + executed_provider_call_count,
                    "provider_calls_executed": executed_provider_call_count,
                    "external_failure_code": error.code,
                    "internal_failure_code": error.internal_code,
                    **error.diagnostics,
                })
            raise
        except LessonAuthorBlueprintValidationError as error:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Course architecture repair could not be applied to the canonical source scope.",
                internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                failure_stage="architecture_repair_patch_apply",
            ) from error
        except HTTPException as error:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Provider returned an invalid scoped architecture repair.",
                internal_code="ARCH_REPAIR_JSON_INVALID",
                failure_stage="architecture_repair_json_parser",
                diagnostics={"provider_http_status": error.status_code},
            ) from error
        return WorkflowGenerationResult(
            repaired,
            combine_usage(*repair_usages).model_dump(),
            provider_call_count=len(scoped_contexts),
        )

    async def deterministic_repair_course(
        blueprint: dict[str, Any],
        targets: list[RepairTarget],
        _source_map: dict[str, Any],
    ) -> WorkflowGenerationResult | None:
        """Apply server-proven factless-unit removals before any provider call."""

        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="architecture_repair_target_snapshot",
        )
        prepared_targets = _v5_prepare_semantic_repair_targets(
            blueprint,
            targets,
            v5_source_context.source_map_copy(),
        )
        deterministic_assessment = _v5_deterministic_assessment_alignment_payload(prepared_targets)
        if deterministic_assessment is not None and all(
            target.get("deterministic_semantic_delta") is True
            for target in prepared_targets
        ):
            repaired = apply_course_architecture_repair_patches(
                blueprint,
                prepared_targets,
                deterministic_assessment,
            )
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_deterministic",
                "event": "completed",
                "repair_pass_number": int(runtime.get("repair_provider_pass") or 0) + 1,
                "repair_target_count": len(prepared_targets),
                "provider_called": False,
                "semantic_operation": "repair_assessment_alignment",
                "deterministic_reason": "EXACT_BASE_ELIGIBLE_TEACHING_ANCHOR",
            })
            return WorkflowGenerationResult(repaired)
        evidence_alignment = _v5_deterministic_evidence_alignment_candidate(
            blueprint,
            prepared_targets,
        )
        if evidence_alignment is not None:
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_deterministic",
                "event": "completed",
                "repair_pass_number": int(runtime.get("repair_provider_pass") or 0) + 1,
                "repair_target_count": len(prepared_targets),
                "provider_called": False,
                "semantic_operation": "align_concepts_to_evidence",
                "canonical_fact_count": v5_source_context.canonical_fact_count,
                "evidence_scope_count": v5_source_context.evidence_scope_count,
            })
            return WorkflowGenerationResult(evidence_alignment)

        outcome = _deterministic_factless_unit_removal_candidate(
            blueprint,
            prepared_targets,
            source_map=v5_source_context.source_map_copy(),
            source_coverage_manifest=source_coverage_manifest,
            known_source_refs=set(structure_context.get("known_source_refs", set())),
            source_structure_nodes=structure_context.get("source_structure_nodes"),
        )
        if outcome is None:
            return None
        repaired, repair_metadata = outcome
        emit_blueprint_diagnostic({
            "stage": "architecture_repair_deterministic",
            "event": "completed",
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0) + 1,
            "repair_target_count": len(targets),
            "provider_called": False,
            "baseline": source_fact_allocation_diagnostics(blueprint),
            "candidate": source_fact_allocation_diagnostics(repaired),
            **repair_metadata,
        })
        return WorkflowGenerationResult(repaired)

    def prepare_course_repair_targets(
        blueprint: dict[str, Any],
        targets: list[RepairTarget],
        _source_map: dict[str, Any],
    ) -> list[RepairTarget]:
        """Preflight V5 target authority before scheduler provider accounting."""

        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="architecture_repair_assessment_preflight",
        )
        assert_v5_scoped_repair_target_bound(targets, v5_source_context)
        prepared = _v5_prepare_semantic_repair_targets(
            blueprint,
            targets,
            v5_source_context.source_map_copy(),
        )
        assessment_targets = [
            target for target in prepared
            if target.get("semantic_operations") == ["repair_assessment_alignment"]
        ]
        action_intent_targets = [
            target for target in prepared
            if target.get("semantic_operations") == ["set_block_intent"]
        ]
        emit_blueprint_diagnostic({
            "stage": "architecture_repair_assessment_preflight",
            "event": "completed",
            "repair_target_count": len(prepared),
            "assessment_alignment_target_count": len(assessment_targets),
            "deterministic_assessment_alignment_count": sum(
                target.get("deterministic_semantic_delta") is True
                for target in assessment_targets
            ),
            "provider_assessment_choice_count": sum(
                target.get("deterministic_semantic_delta") is not True
                for target in assessment_targets
            ),
            "assessment_intent_repair_target_count": sum(
                any(candidate.get("allowed_intents") for candidate in target.get("assessment_alignment_candidates", []))
                for target in assessment_targets
            ),
            "action_intent_repair_target_count": len(action_intent_targets),
            "action_intent_allowed_choice_count": sum(
                len({
                    intent
                    for intents in (target.get("allowed_intents_by_block_id") or {}).values()
                    if isinstance(intents, list)
                    for intent in intents
                    if isinstance(intent, str) and intent.strip()
                })
                for target in action_intent_targets
            ),
            "action_intent_assessment_dependency_objective_count": sum(
                max(0, int(target.get("assessment_dependency_objective_count") or 0))
                for target in action_intent_targets
            ),
        })
        return prepared

    try:
        blueprint, workflow = await run_course_architecture_workflow(
            CourseArchitectureWorkflowCallbacks(
                validate_source_scope=validate_source_scope,
                build_source_map=build_global_source_map,
                architect=architect_course,
                validate_blueprint=validate_blueprint_layers,
                repair_blueprint=repair_course,
                prepare_repair_targets=prepare_course_repair_targets,
                deterministic_repair=deterministic_repair_course,
                emit_diagnostic=emit_blueprint_diagnostic,
                run_blocking=service_runtime.concurrency.run_cpu,
            ),
            request_context={
                "correlation_id": request.correlation_id,
                "tenant_id": request.tenant_id,
                "kb_id": request.kb_id,
                "conversation_id": request.conversation_id,
                "course_id": request.course_id,
                "source_document_count": len(request.source_documents),
            },
            max_repair_attempts=min(2, max(0, settings.course_workflow_max_repair_attempts)),
        )
        coverage_metrics = validate_lesson_author_source_coverage(
            {"chapters": blueprint.get("chapters", [])},
            source_coverage_manifest,
        )
        if request.component_capabilities is not None:
            blueprint["component_capabilities"] = request.component_capabilities.model_dump()
        if structure_context.get("source_chapter_policy") is not None:
            try:
                blueprint = bind_source_chapters(blueprint, structure_context["source_chapter_policy"], runtime.get("source_map"))
            except LessonAuthorBlueprintValidationError as error:
                raise WorkflowFailure(
                    error.code, "Final Blueprint violates source chapter authority.",
                    internal_code=error.code, failure_stage="source_chapter_policy_final",
                    diagnostics=error.safe_diagnostic(),
                ) from error
        source_map = runtime.get("source_map")
        if not isinstance(source_map, dict):
            # The graph owns the map in state; recompute deterministically only
            # for its response contract, never from top-K retrieval.
            source_map = await service_runtime.concurrency.run_cpu(build_global_source_map)
        source_map_coverage = source_map.get("coverage") if isinstance(source_map.get("coverage"), dict) else {}
        retrieval.update({
            "source_map_version": source_map.get("version"),
            "source_map_section_count": source_map_coverage.get("section_count", 0),
            "source_map_concept_count": len(source_map.get("concepts", [])),
            "source_map_fact_count": source_map_coverage.get("source_fact_count", 0),
            **{
                key: value
                for key, value in (runtime.get("source_map_diagnostics") or {}).items()
                if key in {
                    "source_total_facts", "source_total_sections", "source_total_concepts",
                    "source_map_complete", "architect_context_mode", "architect_context_size",
                    "architect_detail_fact_count",
                }
            },
        })
        retrieval.update({
            "source_coverage_required_count": coverage_metrics["required_count"],
            "source_coverage_covered_count": coverage_metrics["covered_count"],
            "source_coverage_status": coverage_metrics["status"],
        })
        retrieval = update_blueprint_source_coverage(retrieval, blueprint, structure_context)
        try:
            blueprint, media_metrics = enrich_lesson_author_blueprint_media_review(
                blueprint,
                source_coverage_manifest,
                request.locale,
            )
            emit_blueprint_diagnostic({
                "stage": "media_recommendation_evaluation",
                "event": "completed",
                "provider_called": False,
                "media_review_version": MEDIA_REVIEW_VERSION,
                **media_metrics,
            })
        except Exception as error:
            # Media is additive review metadata. A failure must not erase a
            # source-valid Blueprint or imply that no media is needed.
            emit_blueprint_diagnostic({
                "stage": "media_recommendation_evaluation",
                "event": "failed",
                "provider_called": False,
                "error_type": type(error).__name__,
                "media_review_available": False,
            })
    except WorkflowFailure as error:
        emit_blueprint_diagnostic({
            "stage": "endpoint_response",
            "event": "failed",
            "failure_stage": error.failure_stage or "blueprint_validation",
            "internal_failure_code": error.internal_code,
            "external_failure_code": error.code,
            "total_repair_provider_calls": int(error.diagnostics.get("repair_provider_calls", error.diagnostics.get("total_repair_provider_calls", 0))),
            "repair_pass_number": int((error.diagnostics or {}).get("repair_count") or runtime.get("repair_provider_pass") or 0),
            "duration_ms": max(0, round((perf_counter() - workflow_started) * 1000)),
        })
        raise HTTPException(
            status_code=422 if error.code in {
                "SOURCE_SCOPE_INCOMPLETE", "SOURCE_MAP_SCOPE_INCOMPLETE", "SOURCE_EVIDENCE_INSUFFICIENT",
                "SOURCE_FACT_EXTRACTION_CAPACITY_EXCEEDED", "ARCHITECT_CONTEXT_CANNOT_REPRESENT_SOURCE",
                "SOURCE_STRUCTURE_REVIEW_REQUIRED",
            } else 502,
            detail={
                "code": error.code,
                "message": (
                    ("Source chapter structure requires review before generation." if request.locale == "en"
                     else "Cần kiểm tra cấu trúc chương của nguồn trước khi tạo Bản thiết kế.")
                    if error.code == "SOURCE_STRUCTURE_REVIEW_REQUIRED"
                    else lesson_author_blueprint_failure_message(request.locale)
                ),
                "failure_stage": error.failure_stage or "blueprint_validation",
                "internal_failure_code": error.internal_code,
                "correlation_id": request.correlation_id,
                "total_repair_provider_calls": int(error.diagnostics.get("repair_provider_calls", error.diagnostics.get("total_repair_provider_calls", 0))),
                "workflow_issues": error.issues[:20],
            },
        ) from error
    except LessonAuthorBlueprintValidationError as error:
        emit_blueprint_diagnostic({
            "stage": "endpoint_response",
            "event": "failed",
            "failure_stage": "blueprint_source_coverage_validation",
            "internal_failure_code": "BLUEPRINT_SOURCE_COVERAGE_INVALID",
            "external_failure_code": "LESSON_AUTHOR_BLUEPRINT_SOURCE_COVERAGE_INVALID",
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
            "duration_ms": max(0, round((perf_counter() - workflow_started) * 1000)),
            "validation_code": error.code,
        })
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LESSON_AUTHOR_BLUEPRINT_SOURCE_COVERAGE_INVALID",
                "message": lesson_author_blueprint_failure_message(request.locale),
                "usage": retrieval_usage.model_dump(),
            },
        ) from error
    usage = combine_usage(retrieval_usage, AiUsage(**(workflow.get("usage") or {})))
    emit_blueprint_diagnostic({
        "stage": "endpoint_response",
        "event": "completed",
        "repair_pass_number": int(workflow.get("repair_count") or 0),
        "duration_ms": max(0, round((perf_counter() - workflow_started) * 1000)),
        "workflow_status": workflow.get("status"),
        "workflow_validation_codes": list(workflow.get("validation_codes") or []),
        **source_fact_allocation_diagnostics(blueprint),
    })
    return {
        "blueprint": blueprint,
        "source_map": source_map,
        "usage": usage.model_dump(),
        "sources": sources,
        "retrieval": retrieval,
        "workflow": workflow,
    }
