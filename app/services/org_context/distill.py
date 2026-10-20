"""``POST /v1/lesson-author/org-context/distill``: draft the organization profile from the tenant's context documents.

The backend's distill worker sends the learned context documents of one tenant (ORG_CONTEXT spec §3). This service
reads their learned chunks (``app.repositories.org_context``: sizes first, then only the texts it reads), samples them
evenly when they exceed ``max_source_chars`` and lets ``app.idm.org_distill`` map-reduce them through the IDM runtime:
the provider slots of the process limiter (``SERVICE_BUSY`` when saturated), a per-call timeout, the usage ledger and
the whole-request deadline (``AI_RAG_LESSON_AUTHOR_DEADLINE_MS``, 504). Logs carry counts and codes only.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from app.core.config import settings
from app.core.errors import AppError, ServiceError
from app.idm.org_distill import OrgSourceBatch, OrgSourceChunk, batch_source, distill_profile, sample_chunks
from app.idm.policy import ORG_DISTILL_BATCH_MAX_CHARS
from app.idm.runtime import IdmBudgetError, IdmProviderError, IdmResponseInvalidError, IdmRuntime, log_stage
from app.repositories import org_context as org_context_repository
from app.repositories.executor import SqlExecutor
from app.schemas.org_context import OrgContextDistillResponse, RagOrgContextDistillRequest
from app.services.deadlines import run_with_deadline
from app.services.orchestration_v2.common import _orchestration_v2_http_error
from app.services.orchestration_v2.idm import _idm_runtime

DISTILL_ROUTE: Final = "/v1/lesson-author/org-context/distill"
# Keyset page of chunk sizes (two integers per row) and batch of chunk texts (a chunk is at most a few thousand
# characters, so one text batch stays near a megabyte).
_SIZE_PAGE: Final = 2_000
_TEXT_PAGE: Final = 200
_MS: Final = 1000


@dataclass(frozen=True)
class _Source:
    document_ids: list[str]
    batches: list[OrgSourceBatch]
    truncated: bool

    @property
    def chars(self) -> int:
        return sum(batch.chars for batch in self.batches)


async def org_context_distill(request: RagOrgContextDistillRequest, pool: SqlExecutor) -> OrgContextDistillResponse:
    deadline_ms = settings.lesson_author_deadline_ms
    runtime = _idm_runtime(request, budget_ms=deadline_ms, allowance=None)
    return await run_with_deadline(DISTILL_ROUTE, deadline_ms, _distill(request, pool, runtime))


async def _distill(request: RagOrgContextDistillRequest, pool: SqlExecutor,
                   runtime: IdmRuntime) -> OrgContextDistillResponse:
    started = time.perf_counter()
    source = await read_source(pool, request)
    try:
        profile = await distill_profile(runtime, source.batches, parallelism=settings.idm_w1_parallelism,
                                        excerpts=source.truncated)
    except IdmBudgetError as error:
        raise AppError("REQUEST_DEADLINE_EXCEEDED", 504, "The request exceeded its processing deadline.") from error
    except IdmProviderError as error:
        raise _provider_error(error) from error
    except IdmResponseInvalidError as error:
        raise AppError("ORG_CONTEXT_DISTILL_OUTPUT_INVALID", 502,
                       "The AI provider returned no valid organization profile.") from error
    usage_complete = runtime.usage.complete and runtime.provider_failure_code is None
    log_stage("org_context_distill_completed", {
        "correlation_id": request.correlation_id, "document_count": len(source.document_ids),
        "batch_count": len(source.batches), "source_chars": source.chars, "truncated": source.truncated,
        "call_count": runtime.usage.calls, "usage": runtime.usage.as_usage(), "usage_complete": usage_complete,
        "duration_ms": int((time.perf_counter() - started) * _MS),
        "response_adjustments": dict(sorted(runtime.adjustments.items())),
    })
    return OrgContextDistillResponse(
        profile=profile, source_document_ids=source.document_ids, source_chars=source.chars,
        truncated=source.truncated, usage=runtime.usage.as_usage(), usage_complete=usage_complete,
        provider_calls=runtime.usage.calls)


def _provider_error(error: IdmProviderError) -> ServiceError:
    return _orchestration_v2_http_error(error.code, "AI provider failed while drafting the organization profile.",
                                        status_code=error.http_status)


async def read_source(pool: SqlExecutor, request: RagOrgContextDistillRequest) -> _Source:
    """The learned text of exactly the requested documents of the tenant's knowledge base, in request and source
    order, sampled evenly over the whole source when it exceeds ``max_source_chars``."""

    rows = await org_context_repository.fetch_context_indexes(pool, request.tenant_id, request.kb_id,
                                                              request.document_ids)
    index_of = {str(row["document_id"]): str(row["index_id"]) for row in rows}
    chunks: list[OrgSourceChunk] = []
    for document_id in request.document_ids:
        if document_id in index_of:
            chunks.extend(await _chunk_sizes(pool, request, document_id, index_of[document_id]))
    selected, truncated = sample_chunks([chunk for chunk in chunks if chunk.chars > 0], request.max_source_chars)
    documents: dict[str, list[str]] = {}
    # A document learned again between the two reads (its chunks are gone) is left out, never read half.
    gone: set[str] = set()
    for document_id in dict.fromkeys(chunk.document_id for chunk in selected):
        numbers = [chunk.chunk_no for chunk in selected if chunk.document_id == document_id]
        texts = await _chunk_texts(pool, request, document_id, index_of[document_id], numbers)
        if texts:
            documents[document_id] = texts
        else:
            gone.add(document_id)
    if not documents:
        log_stage("org_context_distill_no_source", {
            "correlation_id": request.correlation_id, "requested_count": len(request.document_ids),
            "learned_count": len(index_of)})
        raise AppError("ORG_CONTEXT_NO_LEARNED_SOURCE", 409, "None of the documents has learned text.")
    batches = batch_source(list(documents.values()), max_chars=request.max_source_chars,
                           batch_chars=ORG_DISTILL_BATCH_MAX_CHARS)
    # Every document with learned text was considered; a sampled-out one is part of the source the sample stands for.
    considered = [document_id for document_id in dict.fromkeys(chunk.document_id for chunk in chunks if chunk.chars > 0)
                  if document_id not in gone]
    return _Source(considered, batches, truncated)


async def _chunk_sizes(pool: SqlExecutor, request: RagOrgContextDistillRequest, document_id: str,
                       index_id: str) -> list[OrgSourceChunk]:
    found: list[OrgSourceChunk] = []
    after = -1
    while True:
        rows = await org_context_repository.fetch_context_chunk_sizes(
            pool, request.tenant_id, request.kb_id, document_id, index_id, after, _SIZE_PAGE)
        found.extend(OrgSourceChunk(document_id, int(row["chunk_no"]), int(row["chars"] or 0)) for row in rows)
        if len(rows) < _SIZE_PAGE:
            return found
        after = found[-1].chunk_no


async def _chunk_texts(pool: SqlExecutor, request: RagOrgContextDistillRequest, document_id: str, index_id: str,
                       numbers: Sequence[int]) -> list[str]:
    texts: list[str] = []
    for start in range(0, len(numbers), _TEXT_PAGE):
        rows = await org_context_repository.fetch_context_chunk_texts(
            pool, request.tenant_id, request.kb_id, document_id, index_id, list(numbers[start:start + _TEXT_PAGE]))
        texts.extend(text for text in (str(row["content"] or "").strip() for row in rows) if text)
    return texts
