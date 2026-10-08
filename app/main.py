from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from copy import deepcopy
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
from app.api.routes import lesson_author_legacy as lesson_author_legacy_routes
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
from app.instructional_quality import source_relationship_pairs
from app.learner_content_purity import build_learner_content_purity_context
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
from app.schemas.chat import RagChatRequest
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorCheckpointRequest as RagLessonAuthorCheckpointRequest
from app.schemas.lesson_author import RagLessonAuthorRequest
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
from app.services.deadlines import record_fallback
from app.services.ingestion.chunking import SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED, SOURCE_EVIDENCE_READY
from app.services.lesson_author.architecture_validation import (
    validate_v5_instructional_coherence as validate_v5_instructional_coherence,
)
from app.services.lesson_author.checkpoint import (
    build_lesson_author_checkpoint_result as build_lesson_author_checkpoint_result,
)
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.proposal_validation import (
    semantic_learning_visible_text as semantic_learning_visible_text,
)
from app.services.lesson_author.staged import writer as staged_writer
from app.services.lesson_author.staged.provider_schemas import (
    bind_staged_instance_payload,
    build_staged_instance_response_model,
    build_staged_multi_repair_model,
    decode_staged_multi_repair,
)
from app.services.lesson_author.staged.provider_schemas import (
    staged_component_payload_code as staged_component_payload_code,
)
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
    normalize_embedding_model,
    provider_http_error_status,
    require_provider_api_key,
    safe_provider_error_diagnostics,
)
from app.services.provider import generate_content as generate_content
from app.services.retrieval.query import decode_json_object
from app.services.retrieval.source_coverage import extract_source_coverage_facts, format_source_coverage_manifest
from app.services.text import clean_text
from app.source_chapter_policy import resolve_source_chapter_policy
from app.workflows.contracts import WorkflowFailure

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
app.include_router(lesson_author_legacy_routes.router)


# Configure the package logger so every app.* module (app.idm, app.core.request_context, ...)
# emits through the JSON handler, not only this module.
configure_application_logging("app")


logger = logging.getLogger(__name__)


# Keep the outer Node -> Python request alive long enough to serialize and
# return a deterministic source-locked fallback after an inner provider
# deadline. Without this gap, a provider timeout and the HTTP client timeout
# race at the same millisecond and turn a valid fallback into outcome_unknown.
ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS = 5_000


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
