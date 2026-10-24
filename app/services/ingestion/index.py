"""Document indexing: extract, chunk, embed and persist one knowledge-base document."""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import asyncpg

from app.content.source_structure import analyze_source_structure
from app.core import metrics
from app.core.concurrency import deadline_seconds
from app.core.config import settings
from app.core.document_limits import (
    UnsupportedDocumentTypeError,
    assert_document_size,
    assert_tenant_storage_path,
    document_source_suffix,
)
from app.core.errors import AppError, DocumentLimitError, ServiceError
from app.core.handled_exceptions import count_handled_exception, report_handled_exception
from app.core.logging import SERVICE_LOGGER_NAME
from app.infra.gemini.calls import normalize_embedding_model
from app.infra.gemini.errors import http_error_detail
from app.infra.gemini.usage import AiUsage
from app.repositories import indexing as index_repository
from app.repositories.indexing import delete_previous_structure_nodes_if_available, persist_structure_nodes_if_available
from app.repositories.pgvector import vector_values
from app.schemas.kb import INDEX_SKIP_REASON_UNCHANGED, RagIndexRequest
from app.services import provider
from app.services import runtime as service_runtime
from app.services.ingestion import extract_isolated
from app.services.ingestion.chunking import build_chunks, build_index_diagnostics
from app.services.ingestion.extract import ExtractedSection, index_document_temp_path
from app.services.ingestion.extract_isolated import IsolatedExtractionError
from app.services.ingestion.fingerprint import (
    EMBEDDING_TASK_TYPE,
    StoredChunk,
    index_fingerprint,
    stored_chunk_values,
)
from app.services.ingestion.storage import fetch_index_source
from app.services.text import clean_text

logger = logging.getLogger(SERVICE_LOGGER_NAME)


SAFE_ERROR_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,95}$")


# Provider answers after which the same index request may succeed later (F5).
RETRYABLE_PROVIDER_HTTP_STATUSES = frozenset({429, 502, 503, 504})

# A newer index of the same document started while this one ran: the newer one wins.
SUPERSEDED_INDEX_MESSAGE = "Phiên học tài liệu đã bị thay thế bởi phiên mới hơn."
# error_reason of an index row whose request was cancelled (caller disconnected, service stopping).
INDEX_CANCELLED_REASON = "INDEX_CANCELLED"
CANCEL_RECORD_TIMEOUT_SECONDS = 10.0


def retryable_provider_failure(exc: BaseException) -> AppError | None:
    """Map a transient provider failure (quota/rate limit/unavailable/timeout) to a retryable 503."""
    if not isinstance(exc, ServiceError) or exc.status_code not in RETRYABLE_PROVIDER_HTTP_STATUSES:
        return None
    detail = http_error_detail(exc)
    code = str(detail.get("code") or "")
    message = detail.get("message")
    return AppError(
        code if SAFE_ERROR_CODE_PATTERN.fullmatch(code) else "AI_PROVIDER_UNAVAILABLE",
        503,
        message if isinstance(message, str) and message.strip()
        else "AI provider hiện không khả dụng. Vui lòng thử lại sau.",
    )


async def _reusable_index_id(
    pool: asyncpg.Pool, row: asyncpg.Record, request: RagIndexRequest, embedding_model: str,
) -> str | None:
    """The active index this request could turn out to match, or ``None`` when it must rebuild anyway.

    Only an idle document qualifies: while another index runs, this request takes its row at once and
    supersedes it, exactly as before.
    """
    active = await index_repository.load_active_index(pool, row, embedding_model, request.embedding_dimensions)
    if active is None or active["index_running"]:
        return None
    return str(active["index_id"])


async def _claim_deferred_index(
    pool: asyncpg.Pool,
    row: asyncpg.Record,
    request: RagIndexRequest,
    embedding_model: str,
    *,
    reusable_index_id: str | None,
    fingerprint: str,
    chunk_count: int,
) -> str | None:
    """Re-check the document's indexes now that the fingerprint is known, before any provider call.

    ``None``: the active index holds exactly what this request would build, so nothing is rebuilt.
    Otherwise this request takes its index row now, unless another index started since the first
    check (it is running, or it already replaced the active index): that one wins, as it would have
    superseded this request's running row.
    """
    current = await index_repository.load_active_index(pool, row, embedding_model, request.embedding_dimensions)
    if current is not None and not current["index_running"]:
        if current["content_sha256"] == fingerprint and current["chunk_count"] == chunk_count:
            return None
        if str(current["index_id"]) == reusable_index_id:
            return await index_repository.start_index_row(pool, row, embedding_model)
    raise ValueError(SUPERSEDED_INDEX_MESSAGE)


def chunk_rows(
    document: Mapping[str, Any], index_id: str, stored_chunks: Sequence[StoredChunk],
    embeddings: Sequence[Sequence[float]],
) -> list[tuple[Any, ...]]:
    """The rag_chunks rows of one index, in insert order; CPU work (768 rounded values per chunk), run off the loop."""
    return [
        (document["tenant_id"], document["kb_id"], document["id"], index_id, *stored, vector_values(embedding))
        for stored, embedding in zip(stored_chunks, embeddings, strict=False)
    ]


async def _mark_cancelled_index(pool: asyncpg.Pool, index_id: str | None, request: RagIndexRequest) -> None:
    """Close the index row of a cancelled request: shielded, so a second cancellation cannot stop the write."""

    if index_id is None:
        return
    try:
        # Bounded: a hung database must not hold the cancelled request (and pool.close() at shutdown) for the full
        # command timeout; a TimeoutError is reported below as INDEX_CANCEL_NOT_RECORDED.
        await asyncio.shield(asyncio.wait_for(
            index_repository.mark_index_error(pool, index_id, INDEX_CANCELLED_REASON, request.tenant_id,
                                              request.kb_id),
            CANCEL_RECORD_TIMEOUT_SECONDS,
        ))
    except Exception as error:  # the cancellation goes on either way; the failed write is counted and logged
        report_handled_exception(logger, error, stage="index", outcome="INDEX_CANCEL_NOT_RECORDED",
                                 tenant_id=request.tenant_id, document_id=request.document_id, index_id=index_id)
        return
    logger.warning(
        "rag_index_cancelled",
        extra={"event": "rag_index_cancelled", "tenant_id": request.tenant_id,
               "document_id": request.document_id, "index_id": index_id},
    )


async def index_document(request: RagIndexRequest, pool: asyncpg.Pool) -> dict[str, Any]:
    # Bounded concurrency: a saturated indexer answers SERVICE_BUSY (503) so the
    # backend's durable KB worker retries later instead of piling up memory.
    async with service_runtime.concurrency.index.slot():
        return await _index_document(request, pool)


async def _index_document(request: RagIndexRequest, pool: asyncpg.Pool) -> dict[str, Any]:
    if request.embedding_dimensions != 768:  # noqa: PLR2004 - dimensions of the vector column
        raise ServiceError(status_code=400, detail="RAG hiện chỉ hỗ trợ embedding 768 chiều.")
    row = await index_repository.load_document(pool, request.tenant_id, request.kb_id, request.document_id)
    if row is None:
        raise ServiceError(status_code=404, detail="Không tìm thấy tài liệu Knowledge Base.")
    # F4: whitespace-only content has nothing to learn; reject it before any index row exists.
    if row["file_path"] is None and not clean_text(str(row["content"] or "")):
        raise ServiceError(status_code=400, detail="Tài liệu không có file nguồn hoặc nội dung để học.")

    index_id: str | None = None
    # The active index this request may turn out to match. When it is set the request takes no index
    # row before the fingerprint shows a rebuild is needed (_claim_deferred_index); a failure before
    # that leaves the document's indexes untouched (the error code is still logged and answered).
    reusable_index_id: str | None = None
    raw_bytes: int | None = None
    effective_embedding_model = normalize_embedding_model(request.embedding_model)
    deadline = asyncio.timeout(deadline_seconds(settings.index_deadline_ms))
    try:
        async with deadline:
            logger.info(
                "rag_index_started",
                extra={"event": "rag_index_started", "tenant_id": request.tenant_id,
                       "kb_id": request.kb_id, "document_id": request.document_id},
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                # F1/F3: ownership, type, size and the download are settled before the index row is
                # created, so a rejected request never supersedes the document's running index.
                source_path: Path | None = None
                if row["file_path"]:
                    storage_path = str(row["file_path"])
                    assert_tenant_storage_path(storage_path, request.tenant_id)
                    # Routed by the stored object's extension; unknown types are rejected (F1).
                    suffix = document_source_suffix(storage_path)
                    source_path = index_document_temp_path(temp_dir, str(row["id"]), f"source{suffix}")
                    raw_bytes = await fetch_index_source(request, storage_path, source_path)
                else:
                    raw_bytes = len(str(row["content"] or "").encode("utf-8"))
                    assert_document_size(raw_bytes, maximum=settings.max_document_bytes)
                if not request.force:
                    reusable_index_id = await _reusable_index_id(pool, row, request, effective_embedding_model)
                if reusable_index_id is None:
                    # Nothing to reuse (or forced): take the index row now, superseding a running one.
                    index_id = await index_repository.start_index_row(pool, row, effective_embedding_model)
                if source_path is not None:
                    sections = await extract_isolated.extract_document(service_runtime.concurrency, source_path)
                else:
                    sections = [ExtractedSection(text=clean_text(row["content"] or ""))]
            logger.info(
                "rag_index_extracted",
                extra={"event": "rag_index_extracted", "tenant_id": request.tenant_id,
                       "document_id": request.document_id, "section_count": len(sections)},
            )
            structure = await service_runtime.concurrency.run_cpu(analyze_source_structure, sections)
            chunks = await service_runtime.concurrency.run_cpu(build_chunks, sections, structure)
            if not chunks:
                raise ValueError("Không tạo được đoạn kiến thức nào từ tài liệu.")
            index_diagnostics = build_index_diagnostics(
                sections,
                chunks,
                raw_bytes=raw_bytes,
                file_name=row["name"] or None,
            )
            logger.info(
                "rag_index_chunked",
                extra={"event": "rag_index_chunked", "tenant_id": request.tenant_id,
                       "document_id": request.document_id, "chunk_count": len(chunks)},
            )
            learned: dict[str, Any] = {
                "status": "learned",
                "chunk_count": len(chunks),
                "structure_source": structure.get("structure_source"),
                "structure_confidence": structure.get("confidence"),
                "structure_node_count": len(structure.get("nodes", [])),
                "diagnostics": index_diagnostics,
            }
            stored_chunks = stored_chunk_values(row, chunks)
            fingerprint = await service_runtime.concurrency.run_cpu(
                index_fingerprint,
                embedding_model=effective_embedding_model,
                embedding_dimensions=request.embedding_dimensions,
                structure=structure,
                chunks=stored_chunks,
            )
            if index_id is None:  # deferred: reusable_index_id is set
                index_id = await _claim_deferred_index(
                    pool, row, request, effective_embedding_model,
                    reusable_index_id=reusable_index_id, fingerprint=fingerprint, chunk_count=len(chunks),
                )
                if index_id is None:
                    logger.info(
                        "rag_index_skipped",
                        extra={"event": "rag_index_skipped", "tenant_id": request.tenant_id,
                               "document_id": request.document_id, "index_id": reusable_index_id,
                               "chunk_count": len(chunks), "reason": INDEX_SKIP_REASON_UNCHANGED},
                    )
                    return {**learned, "usage": AiUsage().model_dump(), "skipped": True,
                            "skip_reason": INDEX_SKIP_REASON_UNCHANGED}

            with metrics.provider_stage("index_embedding"):
                embeddings, embedding_usage = await provider.embed_texts(
                    request.api_key,
                    effective_embedding_model,
                    [chunk["content"] for chunk in chunks],
                    task_type=EMBEDDING_TASK_TYPE,
                    output_dimensionality=request.embedding_dimensions,
                )
            if len(embeddings) != len(chunks):
                raise ValueError("Số lượng embedding không khớp số đoạn kiến thức.")
            logger.info(
                "rag_index_embedded",
                extra={"event": "rag_index_embedded", "tenant_id": request.tenant_id,
                       "document_id": request.document_id, "embedding_count": len(embeddings)},
            )

            rows = await service_runtime.concurrency.run_cpu(chunk_rows, row, index_id, stored_chunks, embeddings)
            async with pool.acquire() as conn:  # noqa: SIM117 - the transaction is the inner block
                async with conn.transaction():
                    index_status = await index_repository.lock_index_status(conn, index_id, row["id"],
                                                                            row["tenant_id"], row["kb_id"])
                    if index_status != "running":
                        raise ValueError(SUPERSEDED_INDEX_MESSAGE)
                    # Multi-row INSERTs inside the same transaction: one statement, one round trip and one run of
                    # the statement-level quota trigger per batch of chunks (SEP-1 #7, audit 2026-10-24 P1).
                    await index_repository.insert_chunks(conn, rows)
                    await persist_structure_nodes_if_available(conn, row, index_id, structure)
                    await delete_previous_structure_nodes_if_available(conn, row, index_id)
                    await index_repository.deactivate_other_indexes(conn, row["id"], index_id, row["tenant_id"],
                                                                    row["kb_id"])
                    # content_sha256 holds the index fingerprint (fingerprint.py) a later re-index compares.
                    await index_repository.mark_index_learned(conn, index_id, row["id"], fingerprint, len(chunks),
                                                              row["tenant_id"], row["kb_id"])
                    await index_repository.delete_inactive_indexes(conn, row["id"], index_id, row["tenant_id"],
                                                                   row["kb_id"])

            logger.info(
                "rag_index_completed",
                extra={"event": "rag_index_completed", "tenant_id": request.tenant_id,
                       "document_id": request.document_id, "chunk_count": len(chunks)},
            )
            return {**learned, "usage": embedding_usage.model_dump()}
    except asyncio.CancelledError:
        # The caller disconnected or the service is stopping. ``except Exception`` below never sees a
        # cancellation, so the row stayed "running" and the next request embedded everything again (audit O2).
        await _mark_cancelled_index(pool, index_id, request)
        raise
    except Exception as exc:
        retryable = retryable_provider_failure(exc)
        if isinstance(exc, TimeoutError) and deadline.expired():
            error_code = "INDEX_DEADLINE_EXCEEDED"
            metrics.DEADLINE_EXCEEDED.labels(route="/v1/kb/documents/index").inc()
        elif retryable is not None:
            error_code = retryable.code
        else:
            error_code = exc.code if isinstance(exc, AppError) else "INDEX_DOCUMENT_FAILED"
        if isinstance(exc, (DocumentLimitError, UnsupportedDocumentTypeError)):
            metrics.DOCUMENT_LIMIT_REJECTIONS.labels(code=exc.code).inc()
        # rag_index_failed below carries the traceback (logger.exception).
        count_handled_exception(exc, stage="index", outcome=error_code)
        await index_repository.mark_index_error(pool, index_id, error_code, row["tenant_id"], row["kb_id"])
        logger.exception(
            "rag_index_failed",
            extra={"event": "rag_index_failed", "tenant_id": request.tenant_id,
                   "document_id": request.document_id, "error_code": error_code,
                   "error_type": exc.error_type if isinstance(exc, IsolatedExtractionError) else type(exc).__name__},
        )
        if isinstance(exc, DocumentLimitError):
            raise
        if index_id is None and isinstance(exc, AppError):
            # Rejected before any index row (type, ownership, storage): keep the HTTP status.
            raise
        if retryable is not None:
            # F5: a transient provider failure is a retryable 503 carrying the provider code, so the
            # backend's durable worker retries it instead of reading HTTP 200 + status "error".
            raise retryable from None
        if isinstance(exc, AppError) and exc.http_status >= 500:  # noqa: PLR2004 - server error status
            raise  # e.g. SERVICE_BUSY from the provider limiter: retryable as well
        return {
            "status": "error",
            "chunk_count": 0,
            "usage": AiUsage().model_dump(),
            "error_reason": error_code,
        }
